"""Phase 8: calibrate the selected model and choose the decision policy on VALIDATION, then
freeze everything (model, calibrator, policy) for a single evaluation on the test period.

Calibration is assessed honestly inside validation:
- choice of method: fit on Jul-Aug, score on Sep (forward in time);
- probabilities used to optimise the policy: month-wise cross-fitting (each validation month
  calibrated by a calibrator fitted on the other two), so the policy is not tuned on
  calibrated probabilities that saw their own labels.
The calibrator shipped with the model is then refitted on all of validation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg
import skops.io as sio
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from fraudplatform.config import CostConfig, FeatureConfig, ModelConfig, Splits
from fraudplatform.decision.calibration import (
    METHODS,
    apply_calibrator,
    calibrator_to_dict,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from fraudplatform.decision.costs import (
    APPROVE,
    CostModel,
    bayes_actions,
    evaluate_actions,
    threshold_actions,
    threshold_grid,
)
from fraudplatform.eda.data import load_features
from fraudplatform.models import baselines, pipelines
from fraudplatform.models.comparison import prepare, training_start
from fraudplatform.models.tracking import experiment_id, lineage_tags

log = logging.getLogger(__name__)
EXPERIMENT = "phase8-cost-calibration"
REVIEW_GRID = np.unique(np.round(np.geomspace(0.001, 0.6, 45), 5))
BLOCK_GRID = np.append(np.round(np.linspace(0.05, 0.99, 48), 4), 1.01)  # 1.01 = never block


def probability_metrics(p, y) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(p, y),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def cross_fitted(method: str, p, y, month) -> np.ndarray:
    """Calibrated p where each month is calibrated by a calibrator fitted on the others."""
    out = np.empty(len(p))
    for m in np.unique(month):
        held = month == m
        cal = fit_calibrator(method, p[~held], y[~held])
        out[held] = apply_calibrator(method, cal, p[held])
    return out


def decide(policy: dict, score, amount, day, costs: CostModel, capacity: int) -> np.ndarray:
    if policy["type"] == "thresholds":
        return threshold_actions(score, day, policy["t_review"], policy["t_block"], capacity)
    return bayes_actions(score, amount, day, costs, capacity)


def optimise_policies(
    score,
    y,
    amount,
    day,
    costs: CostModel,
    capacity: int,
    review_grid=REVIEW_GRID,
    block_grid=BLOCK_GRID,
):
    """Best global thresholds (P1) and the per-transaction policy (P2) for one cost setting."""
    realized = costs.realized(y, amount)
    grid = threshold_grid(score, realized, day, capacity, review_grid, block_grid)
    best = grid.loc[grid["total_cost_eur"].idxmin()]
    p1 = {
        "type": "thresholds",
        "t_review": float(best["t_review"]),
        "t_block": float(best["t_block"]),
    }
    p2 = {"type": "expected_cost"}
    return grid, p1, p2, realized


def sensitivity_analysis(
    deployed: dict, score, y, amount, day, costs: CostModel, capacity: int, factors
) -> pd.DataFrame:
    """Move one cost assumption at a time and compare, under the CHANGED costs:

    - the deployed policy, which still decides with the ORIGINAL assumptions (what happens if
      we were wrong and did not know), and
    - the policy re-optimised with the changed assumptions (what we would deploy if we knew).

    regret = deployed - re-optimised >= 0: the price of a wrong assumption.
    A smaller review capacity is a physical limit, so the deployed policy is given it.
    """
    rows = []
    idx = np.arange(len(score))
    base = asdict(costs) | {"review_capacity_per_day": capacity}
    for key in base:
        for factor in factors:
            changed = dict(base)
            value = base[key] * factor
            if key == "reviewer_recall":
                value = min(value, 1.0)
            if key == "review_capacity_per_day":
                value = int(round(value))
            changed[key] = value
            cap = changed.pop("review_capacity_per_day")
            alt_costs = CostModel(**changed)
            alt_grid, alt_p1, _, alt_realized = optimise_policies(
                score, y, amount, day, alt_costs, cap, REVIEW_GRID[::3], BLOCK_GRID[::3]
            )
            reoptimised = {
                "P1": float(alt_grid["total_cost_eur"].min()),
                "P2": float(
                    alt_realized[idx, bayes_actions(score, amount, day, alt_costs, cap)].sum()
                ),
            }
            kept_actions = decide(deployed, score, amount, day, costs, cap)  # ORIGINAL costs
            kept = float(alt_realized[idx, kept_actions].sum())
            best_name = min(reoptimised, key=reoptimised.get)
            rows.append(
                {
                    "assumption": key,
                    "factor": factor,
                    "value": value,
                    "approve_all_eur": float(alt_realized[:, APPROVE].sum()),
                    "deployed_policy_eur": kept,
                    "best_reoptimised_eur": reoptimised[best_name],
                    "best_reoptimised_policy": best_name,
                    "regret_eur": kept - reoptimised[best_name],
                    "p1_t_review": alt_p1["t_review"],
                    "p1_t_block": alt_p1["t_block"],
                }
            )
    return pd.DataFrame(rows)


def run_optimisation(
    conn: psycopg.Connection,
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    cost_cfg: CostConfig,
    splits: Splits,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selected = model_cfg.selected_model
    train = prepare(load_features(conn, training_start(splits, feature_cfg), splits.train.end))
    valid = prepare(load_features(conn, splits.validation.start, splits.validation.end))
    cols = pipelines.model_columns("tree")
    model = pipelines.build_hist_gradient_boosting(
        selected["params"], random_state=model_cfg.random_state
    ).fit(train[cols], train["is_fraud"])
    y = valid["is_fraud"].to_numpy()
    amount = valid["amount_eur"].to_numpy()
    day = valid["ts"].dt.date.to_numpy()
    month = valid["ts"].dt.strftime("%Y-%m").to_numpy()
    p_raw = model.predict_proba(valid[cols])[:, 1]
    log.info("model fitted; validation PR-AUC %.4f", average_precision_score(y, p_raw))

    # --- 1. calibration: choose the method forward in time (fit Jul-Aug, score Sep) ---------
    last = month == month.max()
    calibration = {}
    for method in METHODS:
        cal = fit_calibrator(method, p_raw[~last], y[~last])
        calibration[method] = probability_metrics(
            apply_calibrator(method, cal, p_raw[last]), y[last]
        )
    chosen = min(METHODS, key=lambda m: calibration[m]["log_loss"])
    log.info(
        "calibration (held-out %s): %s -> %s",
        month.max(),
        {m: round(v["log_loss"], 5) for m, v in calibration.items()},
        chosen,
    )
    q = cross_fitted(chosen, p_raw, y, month)  # probabilities used to choose the policy
    final_cal = fit_calibrator(chosen, p_raw, y)  # shipped with the model
    reliability = {
        m: reliability_table(cross_fitted(m, p_raw, y, month), y, bins=15).reset_index()
        for m in METHODS
    }

    # --- 2. policies on validation -----------------------------------------------------------
    costs = CostModel.from_config(cost_cfg)
    capacity = cost_cfg.review_capacity_per_day
    grid, p1, p2, realized = optimise_policies(q, y, amount, day, costs, capacity)
    rules_score = baselines.rules_score(valid, model_cfg.rules)
    candidates = {
        "approve_all": np.full(len(y), APPROVE),
        "rules (1 rule: review, 2+: block)": threshold_actions(rules_score, day, 1, 2, capacity),
        "P1 thresholds": decide(p1, q, amount, day, costs, capacity),
        "P2 expected cost": decide(p2, q, amount, day, costs, capacity),
    }
    policies = {
        name: evaluate_actions(a, y, amount, realized, day, costs.reviewer_recall)
        for name, a in candidates.items()
    }
    deployed = min(
        ("P1 thresholds", p1),
        ("P2 expected cost", p2),
        key=lambda kv: policies[kv[0]]["total_cost_eur"],
    )
    log.info(
        "validation cost: %s -> deploy %s",
        {k: round(v["total_cost_eur"]) for k, v in policies.items()},
        deployed[0],
    )

    # --- 3. sensitivity: what if an assumption is wrong? ----------------------------------------
    sensitivity = sensitivity_analysis(
        deployed[1], q, y, amount, day, costs, capacity, cost_cfg.sensitivity_factors
    )

    # --- 4. freeze the bundle ----------------------------------------------------------------
    tags = lineage_tags(conn)
    sio.dump(model, out_dir / "model.skops")
    policy = {
        "policy": deployed[1],
        "policy_name": deployed[0],
        "calibration": calibrator_to_dict(chosen, final_cal),
        "costs": asdict(costs),
        "review_capacity_per_day": capacity,
        "model": {
            "family": selected["family"],
            "params": selected["params"],
            "columns": cols,
            "trained_on": [str(training_start(splits, feature_cfg)), str(splits.train.end)],
        },
        "chosen_on": [str(splits.validation.start), str(splits.validation.end)],
        "lineage": tags,
    }
    (out_dir / "policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8", newline="\n"
    )
    grid.to_csv(out_dir / "threshold_grid.csv", index=False, lineterminator="\n")
    sensitivity.to_csv(out_dir / "sensitivity.csv", index=False, lineterminator="\n")
    pd.concat([t.assign(method=m) for m, t in reliability.items()]).to_csv(
        out_dir / "reliability.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(
        {"ts": valid["ts"], "amount_eur": amount, "is_fraud": y, "p_raw": p_raw, "p_calibrated": q}
    ).to_parquet(out_dir / "validation_scores.parquet")
    results = {
        "calibration_heldout_month": calibration,
        "calibration_chosen": chosen,
        "validation_raw_pr_auc": float(average_precision_score(y, p_raw)),
        "policies_validation": policies,
        "p1": p1,
        "deployed": deployed[0],
        "breakeven_examples": {a: costs.breakeven(a) for a in (5, 20, 100, 200, 1000)},
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8", newline="\n"
    )

    with mlflow.start_run(
        experiment_id=experiment_id(EXPERIMENT), run_name="optimise_on_validation"
    ):
        mlflow.set_tags(
            {**tags, "step": "optimise", "deployed_policy": deployed[0], "calibration": chosen}
        )
        mlflow.log_params(
            {
                **{f"cost_{k}": v for k, v in asdict(costs).items()},
                "review_capacity_per_day": capacity,
                **p1,
            }
        )
        for name, metrics in policies.items():
            key = name.split(" ")[0].lower()
            mlflow.log_metric(f"{key}_total_cost_eur", metrics["total_cost_eur"])
        for path in ("policy.json", "results.json", "sensitivity.csv", "threshold_grid.csv"):
            mlflow.log_artifact(str(out_dir / path))
    log.info("phase 8 optimisation done in %.0fs", time.perf_counter() - started)
    return results
