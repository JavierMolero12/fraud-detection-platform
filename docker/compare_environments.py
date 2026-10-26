"""Row-by-row comparison of the frozen bundle's scores in two environments (e.g. Windows
training venv vs the Linux `app` image). `fraud serve verify-bundle` only compares sorted
probabilities; this shows WHERE differences come from: tree output, sigmoid, or decisions.

    python docker/compare_environments.py score OUT.parquet      # once in each environment
    python docker/compare_environments.py compare A.parquet B.parquet [--json OUT.json]

In the container (the script is not in the image, mount the repository):
    docker compose run --rm -v "$PWD:/repo" --entrypoint python api \
        /repo/docker/compare_environments.py score /repo/scores_linux.parquet
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def score(out: Path) -> None:
    from fraudplatform.config import load_serving_config
    from fraudplatform.db import connect
    from fraudplatform.eda.data import load_features
    from fraudplatform.models.pipelines import prepare
    from fraudplatform.serving.scorer import Bundle, Scorer

    cfg = load_serving_config()
    scorer = Scorer(Bundle.load(cfg.bundle_dir), cfg.risk_levels)
    start, end = (date.fromisoformat(d) for d in scorer.bundle.policy["chosen_on"])
    with connect() as conn:
        df = prepare(load_features(conn, start, end, with_label=False))
    x = df[scorer.bundle.columns]
    q = scorer.probability(df)
    day = df["ts"].dt.date.to_numpy()
    frame = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"],
            "tree_output": scorer.bundle.model.decision_function(x),  # log-odds, before sigmoid
            "p_raw": scorer.bundle.model.predict_proba(x)[:, 1],
            "q": q,
            "action": scorer.batch_actions(q, df["amount_eur"].to_numpy(float), day),
        }
    )
    frame.to_parquet(out)
    print(f"{len(frame)} rows -> {out} ({platform.system()}, Python {platform.python_version()})")


def compare(a_path: Path, b_path: Path) -> dict:
    a = pd.read_parquet(a_path).set_index("transaction_id")
    b = pd.read_parquet(b_path).set_index("transaction_id").loc[a.index]
    result = {"rows": len(a)}
    for col in ("tree_output", "p_raw", "q"):
        x, y = a[col].to_numpy(), b[col].to_numpy()
        ulps = np.abs(x.view(np.int64) - y.view(np.int64))
        result[col] = {
            "rows_differing": int((x != y).sum()),
            "max_abs_difference": float(np.abs(x - y).max()),
            "max_ulps": int(ulps.max()),
        }
    result["decisions_identical"] = int((a["action"] == b["action"]).sum())
    return result


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(Path(sys.argv[2]))
    else:
        res = compare(Path(sys.argv[2]), Path(sys.argv[3]))
        print(json.dumps(res, indent=2))
        if "--json" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--json") + 1])
            out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
