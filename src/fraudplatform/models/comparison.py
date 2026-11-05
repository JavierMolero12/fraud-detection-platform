"""Phase 6: fit baselines and first models on the train period, evaluate on validation.

Hyperparameters are fixed defaults from configs/model.yaml. Nothing is chosen by looking at
validation results here (tuning with temporal folds is Phase 7), and the test split is not
loaded at all.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
import psycopg

from fraudplatform.config import FeatureConfig, ModelConfig, Splits
from fraudplatform.eda.data import load_features
from fraudplatform.features.catalog import CATEGORICAL
from fraudplatform.models import baselines, pipelines
from fraudplatform.models.metrics import evaluate
from fraudplatform.models.tracking import experiment_id, lineage_tags

log = logging.getLogger(__name__)
EXPERIMENT = "phase6-baselines-and-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    family: str  # key in model.yaml `models`
    kind: str  # "linear" or "tree": decides the input columns
    builder: Callable
    class_weight: str | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "M1_logistic_regression",
        "Logistic regression",
        "logistic_regression",
        "linear",
        pipelines.build_logistic_regression,
    ),
    ModelSpec(
        "M1b_logistic_regression_balanced",
        "Logistic regression, class_weight=balanced",
        "logistic_regression",
        "linear",
        pipelines.build_logistic_regression,
        "balanced",
    ),
    ModelSpec(
        "M2_random_forest", "Random forest", "random_forest", "tree", pipelines.build_random_forest
    ),
    ModelSpec(
        "M3_hist_gradient_boosting",
        "Histogram gradient boosting",
        "hist_gradient_boosting",
        "tree",
        pipelines.build_hist_gradient_boosting,
    ),
    ModelSpec(
        "M3b_hist_gradient_boosting_balanced",
        "Histogram gradient boosting, class_weight=balanced",
        "hist_gradient_boosting",
        "tree",
        pipelines.build_hist_gradient_boosting,
        "balanced",
    ),
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as plain strings so that encoders see the same values in every split."""
    df = df.copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df


def training_start(splits: Splits, feature_cfg: FeatureConfig):
    """Train period minus the warm-up (decision 016)."""
    return splits.train.start + timedelta(days=feature_cfg.warmup_days)


def run_comparison(
    conn: psycopg.Connection, cfg: ModelConfig, feature_cfg: FeatureConfig, splits: Splits
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metrics per candidate, validation scores per candidate)."""
    start = training_start(splits, feature_cfg)
    train = prepare(load_features(conn, start, splits.train.end))
    valid = prepare(load_features(conn, splits.validation.start, splits.validation.end))
    log.info(
        "train %s rows (%s .. %s), validation %s rows",
        len(train),
        start,
        splits.train.end,
        len(valid),
    )

    ev = cfg.evaluation
    y_valid = valid["is_fraud"].to_numpy()
    days = valid["ts"].dt.date.to_numpy()
    tags = lineage_tags(conn)
    exp_id = experiment_id(EXPERIMENT)
    common_params = {
        "train_start": str(start),
        "train_end": str(splits.train.end),
        "valid_start": str(splits.validation.start),
        "valid_end": str(splits.validation.end),
        "train_rows": len(train),
        "train_frauds": int(train["is_fraud"].sum()),
        "valid_rows": len(valid),
    }
    scores = pd.DataFrame(
        {
            "transaction_id": valid["transaction_id"],
            "ts": valid["ts"],
            "amount_eur": valid["amount_eur"],
            "is_fraud": y_valid,
        }
    )
    results = []

    def record(name, description, score, probabilistic, params, fit_seconds, model=None):
        metrics = evaluate(
            y_valid,
            score,
            valid["amount_eur"],
            days,
            ev.alert_rate,
            probabilistic,
            ev.bootstrap_reps,
            ev.bootstrap_seed,
        )
        metrics["fit_seconds"] = fit_seconds
        with mlflow.start_run(experiment_id=exp_id, run_name=name):
            mlflow.set_tags({**tags, "candidate": name, "description": description})
            mlflow.log_params({**common_params, **params})
            mlflow.log_metrics(metrics)
            if model is not None:
                # skops (not pickle): loading cannot execute arbitrary code; our own types
                # must be listed explicitly as trusted.
                mlflow.sklearn.log_model(
                    model, name="model", skops_trusted_types=pipelines.SKOPS_TRUSTED_TYPES
                )
        scores[name] = score
        results.append({"candidate": name, "description": description, **metrics})
        log.info(
            "%-38s PR-AUC %.4f  ROC-AUC %.4f  recall@%.1f%% %.3f",
            name,
            metrics["pr_auc"],
            metrics["roc_auc"],
            100 * ev.alert_rate,
            metrics["recall_at_alert_rate"],
        )

    record(
        "B0_approve_all",
        "Same score for everything",
        baselines.approve_all_score(valid),
        False,
        {},
        0.0,
    )
    record("B1_amount", "Rank by amount", baselines.amount_score(valid), False, {}, 0.0)
    rule_params = cfg.rules.model_dump()
    record(
        "B2_expert_rules",
        "Number of expert rules fired",
        baselines.rules_score(valid, cfg.rules),
        False,
        {f"rule_{k}": str(v) for k, v in rule_params.items()},
        0.0,
    )

    for spec in MODEL_SPECS:
        params = cfg.models[spec.family]
        model = spec.builder(params, class_weight=spec.class_weight, random_state=cfg.random_state)
        columns = pipelines.model_columns(spec.kind)
        started = time.perf_counter()
        model.fit(train[columns], train["is_fraud"])
        fit_seconds = round(time.perf_counter() - started, 1)
        score = model.predict_proba(valid[columns])[:, 1]
        record(
            spec.name,
            spec.description,
            score,
            True,
            {**params, "class_weight": str(spec.class_weight), "n_features_in": len(columns)},
            fit_seconds,
            model,
        )

    return pd.DataFrame(results), scores


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    t = results.copy()
    t["pr_auc_95ci"] = [
        f"{lo:.3f} – {hi:.3f}" if np.isfinite(lo) else ""
        for lo, hi in zip(t["pr_auc_ci_low"], t["pr_auc_ci_high"], strict=True)
    ]
    cols = [
        "candidate",
        "pr_auc",
        "pr_auc_95ci",
        "roc_auc",
        "precision_at_alert_rate",
        "recall_at_alert_rate",
        "amount_recall_at_alert_rate",
        "brier",
        "mean_predicted",
        "fit_seconds",
    ]
    return t.reindex(columns=cols)
