"""MLflow tracking: local SQLite store in mlruns/ (git-ignored) plus lineage tags.

Every run records which data it saw (dataset hash, data quality run, feature build) and which
code produced it (git commit, dirty flag), so any number in a report can be traced back.
"""

from __future__ import annotations

import os
import subprocess

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import psycopg  # noqa: E402

from fraudplatform.config import PROJECT_ROOT  # noqa: E402

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def tracking_uri() -> str:
    default = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
    return os.environ.get("MLFLOW_TRACKING_URI", default)


def experiment_id(name: str) -> str:
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifacts = (MLRUNS_DIR / "artifacts" / name).resolve()
    return mlflow.create_experiment(name, artifact_location=artifacts.as_uri())


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lineage_tags(conn: psycopg.Connection) -> dict[str, str]:
    dataset_hash, simulator_version = conn.execute(
        "SELECT dataset_hash, simulator_version FROM raw.load_log ORDER BY load_id DESC LIMIT 1"
    ).fetchone()
    build_id, feature_version, dq_run_id = conn.execute(
        "SELECT build_id, feature_set_version, dq_run_id FROM features.builds "
        "ORDER BY build_id DESC LIMIT 1"
    ).fetchone()
    return {
        "dataset_hash": dataset_hash,
        "simulator_version": simulator_version,
        "dq_run_id": str(dq_run_id),
        "feature_build_id": str(build_id),
        "feature_set_version": feature_version,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))),
    }
