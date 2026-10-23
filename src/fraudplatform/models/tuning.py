"""Phase 7: temporal cross-validation, hyperparameter search and two leakage experiments.

Everything that chooses something (hyperparameters) only sees the training window, split
into time-ordered folds. Validation is used once at the end to confirm the choice; the test
period is not loaded.

Experiments
- H7: the same model scored with shuffled K-fold vs temporal folds.
- H8: a model trained on the deliberately leaky features, scored (a) on validation with leaky
  values, the number an offline report would show, and (b) with the values that exist at
  decision time, which is what the model would actually receive in production.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg
from scipy.stats import loguniform
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from fraudplatform.config import FeatureConfig, ModelConfig, Splits
from fraudplatform.eda.data import load_features
from fraudplatform.features.catalog import LEAKY_FEATURES
from fraudplatform.markdown import md_table
from fraudplatform.models import pipelines
from fraudplatform.models.comparison import prepare, training_start
from fraudplatform.models.metrics import bootstrap_pr_auc_difference, evaluate
from fraudplatform.models.tracking import experiment_id, lineage_tags
from fraudplatform.models.validation import fold_indices, random_folds, temporal_folds

log = logging.getLogger(__name__)
EXPERIMENT = "phase7-temporal-cv-tuning"


def parse_space(space: dict) -> dict:
    """YAML search space -> sklearn distributions ({'loguniform': [a, b]} or a list)."""
    out = {}
    for name, spec in space.items():
        if isinstance(spec, dict) and "loguniform" in spec:
            out[f"model__{name}"] = loguniform(*spec["loguniform"])
        else:
            out[f"model__{name}"] = list(spec)
    return out


def cv_scores(factory, X: pd.DataFrame, y: np.ndarray, splits) -> list[float]:
    """PR-AUC of a freshly fitted model on each fold's test rows."""
    scores = []
    for train_idx, test_idx in splits:
        model = factory().fit(X.iloc[train_idx], y[train_idx])
        scores.append(
            average_precision_score(y[test_idx], model.predict_proba(X.iloc[test_idx])[:, 1])
        )
    return scores


def with_leaky_values(df: pd.DataFrame) -> pd.DataFrame:
    """Same column names, leaky values: the model pipeline does not change at all."""
    out = df.copy()
    for leaky, clean in LEAKY_FEATURES.items():
        out[clean] = out[leaky]
    return out


def _cv_table(search) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    keep = (
        [c for c in res.columns if c.startswith("param_")]
        + [c for c in res.columns if c.startswith("split") and c.endswith("_test_score")]
        + ["mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
    )
    return res[keep].sort_values("rank_test_score").reset_index(drop=True)


def run_phase7(
    conn: psycopg.Connection,
    cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    splits: Splits,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_cfg, tuning, seed = cfg.cross_validation, cfg.tuning, cfg.random_state
    start = training_start(splits, feature_cfg)
    train = prepare(load_features(conn, start, splits.train.end, include_leaky=True))
    valid = prepare(
        load_features(conn, splits.validation.start, splits.validation.end, include_leaky=True)
    )
    y, y_valid = train["is_fraud"].to_numpy(), valid["is_fraud"].to_numpy()
    days_valid = valid["ts"].dt.date.to_numpy()
    folds = temporal_folds(start, cv_cfg.test_months, cv_cfg.gap_days)
    tsplits = fold_indices(train["ts"], folds)
    rsplits = random_folds(y, n_splits=len(folds), seed=seed)
    tree_cols, lin_cols = pipelines.model_columns("tree"), pipelines.model_columns("linear")
    hgb_default = cfg.models["hist_gradient_boosting"]
    lr_default = cfg.models["logistic_regression"]
    ev = cfg.evaluation
    exp_id = experiment_id(EXPERIMENT)
    tags = lineage_tags(conn)
    fold_info = [
        {
            "fold": f.name,
            "train_end": str(f.train_end),
            "train_rows": len(tr),
            "test_rows": len(te),
            "test_frauds": int(y[te].sum()),
        }
        for f, (tr, te) in zip(folds, tsplits, strict=True)
    ]
    log.info("train window %s rows, %s temporal folds", len(train), len(folds))
    results: dict = {"folds": fold_info}

    def hgb(params=None):
        return lambda: pipelines.build_hist_gradient_boosting(
            params or hgb_default, random_state=seed
        )

    def run(name, params, metrics, artifacts=()):
        with mlflow.start_run(experiment_id=exp_id, run_name=name):
            mlflow.set_tags({**tags, "step": name})
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            for path in artifacts:
                mlflow.log_artifact(str(path))

    # --- H7: random vs temporal cross-validation (same model, same rows) -------------------
    t0 = time.perf_counter()
    temporal = cv_scores(hgb(), train[tree_cols], y, tsplits)
    shuffled = cv_scores(hgb(), train[tree_cols], y, rsplits)
    default_model = hgb()().fit(train[tree_cols], y)
    p_default = default_model.predict_proba(valid[tree_cols])[:, 1]
    out_of_time = average_precision_score(y_valid, p_default)
    results["h7"] = {
        "temporal_folds": temporal,
        "random_folds": shuffled,
        "validation_out_of_time": out_of_time,
    }
    run(
        "H7_random_vs_temporal_cv",
        hgb_default,
        {
            "cv_temporal_mean": float(np.mean(temporal)),
            "cv_random_mean": float(np.mean(shuffled)),
            "validation_pr_auc": out_of_time,
        },
    )
    log.info(
        "H7 temporal %.4f random %.4f validation %.4f (%.0fs)",
        np.mean(temporal),
        np.mean(shuffled),
        out_of_time,
        time.perf_counter() - t0,
    )

    # --- logistic regression: grid over C ---------------------------------------------------
    t0 = time.perf_counter()
    lr_search = GridSearchCV(
        pipelines.build_logistic_regression(lr_default, random_state=seed),
        {f"model__{k}": v for k, v in tuning["logistic_regression"]["grid"].items()},
        scoring=tuning["scoring"],
        cv=tsplits,
        n_jobs=4,
        refit=True,
    ).fit(train[lin_cols], y)
    lr_table = _cv_table(lr_search)
    lr_table.to_csv(out_dir / "lr_cv_results.csv", index=False)
    p_lr = lr_search.predict_proba(valid[lin_cols])[:, 1]
    log.info(
        "LR best %s cv %.4f (%.0fs)",
        lr_search.best_params_,
        lr_search.best_score_,
        time.perf_counter() - t0,
    )

    # --- gradient boosting: randomized search ----------------------------------------------
    t0 = time.perf_counter()
    hgb_cfg = tuning["hist_gradient_boosting"]
    hgb_search = RandomizedSearchCV(
        pipelines.build_hist_gradient_boosting(hgb_default, random_state=seed),
        parse_space(hgb_cfg["space"]),
        n_iter=hgb_cfg["n_iter"],
        scoring=tuning["scoring"],
        cv=tsplits,
        n_jobs=1,
        refit=True,
        random_state=seed,
    ).fit(train[tree_cols], y)
    hgb_table = _cv_table(hgb_search)
    hgb_table.to_csv(out_dir / "hgb_cv_results.csv", index=False)
    p_hgb = hgb_search.predict_proba(valid[tree_cols])[:, 1]
    log.info(
        "HGB best %s cv %.4f (%.0fs)",
        hgb_search.best_params_,
        hgb_search.best_score_,
        time.perf_counter() - t0,
    )

    # default HGB through the same temporal CV, so tuned vs default is like-for-like
    default_cv = float(np.mean(temporal))
    for name, search, table, p in [
        ("logistic_regression", lr_search, lr_table, p_lr),
        ("hist_gradient_boosting", hgb_search, hgb_table, p_hgb),
    ]:
        metrics = evaluate(
            y_valid,
            p,
            valid["amount_eur"],
            days_valid,
            ev.alert_rate,
            True,
            ev.bootstrap_reps,
            ev.bootstrap_seed,
        )
        vs_default = bootstrap_pr_auc_difference(y_valid, p, p_default, days_valid, 300, 11)
        best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        results[name] = {
            "best_params": best,
            "cv_best": float(search.best_score_),
            "validation": metrics,
            "vs_default_hgb_on_validation": vs_default,
        }
        path = out_dir / f"{name}_cv_results.md"
        path.write_text(md_table(table.round(4)), encoding="utf-8", newline="\n")
        run(
            f"{name}_tuned",
            best,
            {
                "cv_best_pr_auc": float(search.best_score_),
                **metrics,
                "validation_diff_vs_default_hgb": vs_default["difference"],
            },
        )
    results["default_hgb"] = {
        "cv_temporal_mean": default_cv,
        "validation": evaluate(
            y_valid,
            p_default,
            valid["amount_eur"],
            days_valid,
            ev.alert_rate,
            True,
            ev.bootstrap_reps,
            ev.bootstrap_seed,
        ),
    }

    # --- H8: leaky features -------------------------------------------------------------------
    t0 = time.perf_counter()
    train_leaky = with_leaky_values(train)
    leaky_cv = cv_scores(hgb(), train_leaky[tree_cols], y, tsplits)
    leaky_model = hgb()().fit(train_leaky[tree_cols], y)
    p_offline = leaky_model.predict_proba(with_leaky_values(valid)[tree_cols])[:, 1]
    p_served = leaky_model.predict_proba(valid[tree_cols])[:, 1]  # what production can compute
    results["h8"] = {
        "clean_cv_temporal_mean": default_cv,
        "leaky_cv_temporal_mean": float(np.mean(leaky_cv)),
        "clean_validation": out_of_time,
        "leaky_validation_offline": average_precision_score(y_valid, p_offline),
        "leaky_validation_served": average_precision_score(y_valid, p_served),
        "served_vs_clean": bootstrap_pr_auc_difference(
            y_valid, p_served, p_default, days_valid, 300, 11
        ),
    }
    run(
        "H8_leaky_features",
        hgb_default,
        {
            "leaky_cv_temporal_mean": results["h8"]["leaky_cv_temporal_mean"],
            "leaky_validation_offline": results["h8"]["leaky_validation_offline"],
            "leaky_validation_served": results["h8"]["leaky_validation_served"],
        },
    )
    log.info(
        "H8 offline %.4f served %.4f clean %.4f (%.0fs)",
        results["h8"]["leaky_validation_offline"],
        results["h8"]["leaky_validation_served"],
        out_of_time,
        time.perf_counter() - t0,
    )

    pd.DataFrame(
        {
            "transaction_id": valid["transaction_id"],
            "ts": valid["ts"],
            "amount_eur": valid["amount_eur"],
            "is_fraud": y_valid,
            "hgb_default": p_default,
            "hgb_tuned": p_hgb,
            "lr_tuned": p_lr,
            "hgb_leaky_offline": p_offline,
            "hgb_leaky_served": p_served,
        }
    ).to_parquet(out_dir / "validation_scores.parquet", index=False)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    return results
