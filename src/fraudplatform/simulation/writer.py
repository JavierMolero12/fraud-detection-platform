"""Write a simulated dataset to disk with a manifest of row counts and content hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from fraudplatform.simulation.generator import SimulationResult
from fraudplatform.simulation.summary import build_summary


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_dir_name(result: SimulationResult) -> str:
    return f"v{result.simulator_version}-{result.scale}"


def write_dataset(result: SimulationResult, out_root: Path) -> Path:
    out = out_root / dataset_dir_name(result)
    groups: dict[str, dict[str, pd.DataFrame]] = {
        "raw": result.raw,
        "reference": result.reference,
        "sim": {"ground_truth": result.ground_truth, "dq_defects": result.dq_defects},
    }
    files = {}
    for group, tables in groups.items():
        (out / group).mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            path = out / group / f"{name}.csv"
            # Fixed line terminator: identical bytes on Windows and Linux.
            df.to_csv(path, index=False, lineterminator="\n")
            files[f"{group}/{name}.csv"] = {"rows": len(df), "sha256": _sha256(path)}

    dataset_hash = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    manifest = {
        "simulator_version": result.simulator_version,
        "scale": result.scale,
        "seed": result.seed,
        "dataset_hash": dataset_hash,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(build_summary(result, dataset_hash), encoding="utf-8")
    return out
