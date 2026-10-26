"""A stand-in bundle for smoke tests and demos. It is NOT the Phase 8 model.

The real bundle (data/processed/phase8) comes out of phases 6-8 on the full dataset and is not
in git. This trains a small HistGradientBoosting on whatever data is loaded (the tiny dataset
in CI, the small one in the tests) and saves it in the same format, so that the API and the
batch job can run end to end anywhere. Its scores mean nothing about model quality.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import psycopg
import skops.io as sio

from fraudplatform.config import load_cost_config
from fraudplatform.decision.costs import CostModel
from fraudplatform.eda.data import load_features
from fraudplatform.models.pipelines import build_hist_gradient_boosting, model_columns, prepare


def build_demo_bundle(
    conn: psycopg.Connection, out_dir: Path, max_iter: int = 40, train_share: float = 0.6
) -> Path:
    """Fit on the first `train_share` of the loaded period; write model.skops + policy.json."""
    df = prepare(load_features(conn, date(2000, 1, 1), date(2100, 1, 1)))
    if df.empty or not df["is_fraud"].any():
        raise ValueError("no feature rows with fraud: run `fraud features build` first")
    cutoff = df["ts"].quantile(train_share)
    train = df[df["ts"] < cutoff]
    columns = model_columns("tree")
    params = {"max_iter": max_iter, "learning_rate": 0.1}
    model = build_hist_gradient_boosting(params).fit(train[columns], train["is_fraud"])

    costs = load_cost_config()
    policy = {
        "policy": {"type": "expected_cost"},
        "policy_name": "P2 expected cost (DEMO bundle, not the Phase 8 model)",
        "calibration": {"method": "platt", "coef": 1.0, "intercept": 0.0},  # identity
        "costs": asdict(CostModel.from_config(costs)),
        "review_capacity_per_day": costs.review_capacity_per_day,
        "model": {
            "family": "hist_gradient_boosting",
            "params": params,
            "columns": columns,
            "trained_on": [str(train["ts"].min().date()), str(cutoff.date())],
        },
        "chosen_on": [str(cutoff.date()), str(df["ts"].max().date())],
        "lineage": {"demo": "trained by `fraud serve demo-bundle`"},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.dump(model, out_dir / "model.skops")
    with open(out_dir / "policy.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(policy, fh, indent=2)
        fh.write("\n")
    return out_dir
