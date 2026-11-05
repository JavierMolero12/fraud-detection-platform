"""The ONE evaluation on the test period (Oct-Dec 2025).

Everything was frozen on validation: model, calibrator, policy and cost assumptions
(`data/processed/phase8/policy.json` + `model.skops`). This module only scores and reports.

To make "the test set is used once" enforceable rather than a promise, the result is written
to reports/policy/test_evaluation.json, which is committed. If that file exists, the command
refuses to run again. A deliberate re-run (e.g. after a bug fix) needs --force and should be
explained in docs/decisions.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import psycopg
import skops.io as sio
from sklearn.metrics import average_precision_score

from fraudplatform.config import PROJECT_ROOT, ModelConfig, Splits
from fraudplatform.decision.calibration import apply_calibration_dict
from fraudplatform.decision.costs import APPROVE, CostModel, evaluate_actions, threshold_actions
from fraudplatform.decision.optimise import EXPERIMENT, decide, probability_metrics
from fraudplatform.eda.data import load_features
from fraudplatform.models import baselines, pipelines
from fraudplatform.models.comparison import prepare
from fraudplatform.models.metrics import evaluate
from fraudplatform.models.tracking import experiment_id, lineage_tags

log = logging.getLogger(__name__)
MARKER = PROJECT_ROOT / "reports" / "policy" / "test_evaluation.json"


class TestAlreadyEvaluated(RuntimeError):
    pass


def evaluate_on_test(
    conn: psycopg.Connection,
    model_cfg: ModelConfig,
    splits: Splits,
    bundle_dir: Path,
    force: bool = False,
) -> dict:
    if MARKER.exists() and not force:
        raise TestAlreadyEvaluated(
            f"{MARKER} exists: the test period has already been evaluated. "
            "Re-running it would turn the test set into another validation set."
        )
    policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
    model = sio.load(bundle_dir / "model.skops", trusted=pipelines.SKOPS_TRUSTED_TYPES)
    test = prepare(load_features(conn, splits.test.start, splits.test.end))
    cols = policy["model"]["columns"]
    y = test["is_fraud"].to_numpy()
    amount = test["amount_eur"].to_numpy()
    day = test["ts"].dt.date.to_numpy()
    p_raw = model.predict_proba(test[cols])[:, 1]
    q = apply_calibration_dict(policy["calibration"], p_raw)
    ev = model_cfg.evaluation

    costs = CostModel(**policy["costs"])
    capacity = policy["review_capacity_per_day"]
    realized = costs.realized(y, amount)
    rules_score = baselines.rules_score(test, model_cfg.rules)
    p1 = json.loads((bundle_dir / "results.json").read_text(encoding="utf-8"))["p1"]
    actions = {
        "approve_all": np.full(len(y), APPROVE),
        "rules (1 rule: review, 2+: block)": threshold_actions(rules_score, day, 1, 2, capacity),
        "P1 thresholds (frozen)": decide(p1, q, amount, day, costs, capacity),
        "P2 expected cost (frozen, deployed)": decide(
            policy["policy"], q, amount, day, costs, capacity
        ),
    }
    results = {
        "period": [str(splits.test.start), str(splits.test.end)],
        "transactions": int(len(y)),
        "frauds": int(y.sum()),
        "base_rate": float(y.mean()),
        "model": evaluate(
            y, q, amount, day, ev.alert_rate, True, ev.bootstrap_reps, ev.bootstrap_seed
        ),
        "rules_pr_auc": float(average_precision_score(y, rules_score)),
        "calibration_raw": probability_metrics(p_raw, y),
        "calibration_shipped": probability_metrics(q, y),
        "policies": {
            name: evaluate_actions(a, y, amount, realized, day, costs.reviewer_recall)
            for name, a in actions.items()
        },
        "bundle": {
            "policy_name": policy["policy_name"],
            "calibration": policy["calibration"]["method"],
            "chosen_on": policy["chosen_on"],
            "lineage": policy["lineage"],
        },
        "evaluated_with": lineage_tags(conn),
    }
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8", newline="\n")

    with mlflow.start_run(experiment_id=experiment_id(EXPERIMENT), run_name="TEST_evaluation"):
        mlflow.set_tags(
            {**results["evaluated_with"], "test_evaluation": "true", "forced": str(force)}
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in results["model"].items()})
        for name, m in results["policies"].items():
            mlflow.log_metric(f"test_cost_{name.split(' ')[0].lower()}", m["total_cost_eur"])
        for path in (bundle_dir / "model.skops", bundle_dir / "policy.json", MARKER):
            mlflow.log_artifact(str(path))
    log.info(
        "TEST: PR-AUC %.4f, deployed policy cost %.0f EUR vs approve-all %.0f EUR",
        results["model"]["pr_auc"],
        results["policies"]["P2 expected cost (frozen, deployed)"]["total_cost_eur"],
        results["policies"]["approve_all"]["total_cost_eur"],
    )
    return results
