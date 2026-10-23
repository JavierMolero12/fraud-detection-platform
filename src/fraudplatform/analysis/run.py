"""Phase 9 job: explanations, error analysis and segment metrics of the frozen bundle on
validation. Writes everything the notebook and the model card need to data/processed/phase9."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import skops.io as sio

from fraudplatform.analysis.errors import (
    add_outcomes,
    error_profile,
    load_context,
    scenario_breakdown,
    segment_metrics,
)
from fraudplatform.analysis.explain import (
    additivity_error,
    grouped_permutation_importance,
    shap_values,
    single_permutation_importance,
)
from fraudplatform.config import ModelConfig, Splits
from fraudplatform.decision.calibration import apply_calibration_dict
from fraudplatform.decision.costs import BLOCK, CostModel
from fraudplatform.decision.optimise import decide
from fraudplatform.eda.data import load_features
from fraudplatform.features.catalog import FEATURES
from fraudplatform.models import pipelines
from fraudplatform.models.comparison import prepare

log = logging.getLogger(__name__)

PROFILE_COLUMNS = [
    "merchant_category",
    "payment_method",
    "cross_border",
    "amount_band",
    "customer_segment",
    "device_type",
    "hour_band",
    "account_age_band",
    "customer_age_band",
    "home_country",
    "card_type",
]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_band"] = pd.cut(
        out["amount_eur"],
        [0, 5, 20, 100, 500, np.inf],
        labels=["< 5", "5-20", "20-100", "100-500", ">= 500"],
        right=False,
    ).astype(str)
    out["hour_band"] = pd.cut(
        out["local_hour"],
        [0, 6, 12, 18, 24],
        labels=["00-05", "06-11", "12-17", "18-23"],
        right=False,
    ).astype(str)
    out["account_age_band"] = pd.cut(
        out["account_age_days"],
        [-1, 30, 365, 3 * 365, np.inf],
        labels=["<= 30 d", "1-12 m", "1-3 y", "> 3 y"],
    ).astype(str)
    out["customer_age_band"] = pd.cut(
        out["customer_age"],
        [0, 25, 35, 50, 65, 120],
        labels=["18-25", "26-35", "36-50", "51-65", "66+"],
    ).astype(str)
    out["cross_border"] = out["cross_border"].map({1.0: "cross-border", 0.0: "domestic"})
    out["channel"] = np.where(out["is_card_present"] == 1.0, "card present", "card not present")
    return out


def run_phase9(
    conn: psycopg.Connection,
    model_cfg: ModelConfig,
    splits: Splits,
    bundle_dir: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed = model_cfg.random_state
    policy = json.loads((bundle_dir / "policy.json").read_text(encoding="utf-8"))
    model = sio.load(bundle_dir / "model.skops", trusted=pipelines.SKOPS_TRUSTED_TYPES)
    cols = policy["model"]["columns"]
    valid = prepare(load_features(conn, splits.validation.start, splits.validation.end))
    context = load_context(conn, splits.validation.start, splits.validation.end)
    valid = valid.merge(context, on="transaction_id", how="left", validate="one_to_one")

    costs = CostModel(**policy["costs"])
    capacity = policy["review_capacity_per_day"]
    p = apply_calibration_dict(policy["calibration"], model.predict_proba(valid[cols])[:, 1])
    day = valid["ts"].dt.date.to_numpy()
    valid["p"] = p
    valid["action"] = decide(
        policy["policy"], p, valid["amount_eur"].to_numpy(), day, costs, capacity
    )
    df = add_bands(add_outcomes(valid))
    y = df["is_fraud"].to_numpy()
    log.info("validation decisions: %s", df["outcome"].value_counts().to_dict())

    # --- global importance --------------------------------------------------------------------
    t0 = time.perf_counter()
    single = single_permutation_importance(model, df[cols], y, n_repeats=5, seed=seed)
    groups = {}
    for f in FEATURES:
        if f.name in cols:
            groups.setdefault(f.group, []).append(f.name)
    grouped = grouped_permutation_importance(model, df[cols], y, groups, n_repeats=5, seed=seed)
    log.info("permutation importance done (%.0fs)", time.perf_counter() - t0)

    # --- SHAP: a random background and a sample with every kind of decision -------------------
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    background = df.iloc[rng.choice(len(df), 200, replace=False)][cols]
    picks = []
    for outcome, n in [
        ("fraud blocked", 250),
        ("fraud approved (FN)", 150),
        ("legit blocked (FP)", 150),
        ("legit approved", 250),
        ("fraud reviewed", 50),
        ("legit reviewed", 50),
    ]:
        idx = np.flatnonzero(df["outcome"].to_numpy() == outcome)
        picks.extend(rng.choice(idx, min(n, len(idx)), replace=False))
    sample = df.iloc[np.sort(picks)]
    attributions, base, log_odds = shap_values(model, sample[cols], background, seed=seed)
    err = additivity_error(attributions, base, log_odds)
    if err > 1e-6:
        raise RuntimeError(f"SHAP values do not add up to the model output (max error {err})")
    log.info(
        "SHAP on %s rows, additivity error %.1e (%.0fs)", len(sample), err, time.perf_counter() - t0
    )

    # --- errors and segments ------------------------------------------------------------------
    profiles = {c: error_profile(df, c) for c in PROFILE_COLUMNS}
    segments = {
        c: segment_metrics(df, c, costs.reviewer_recall)
        for c in [
            "channel",
            "payment_method",
            "merchant_category",
            "customer_segment",
            "home_country",
            "account_age_band",
            "customer_age_band",
            "amount_band",
            "cross_border",
        ]
    }
    scenarios = scenario_breakdown(df)

    # --- save -------------------------------------------------------------------------------
    keep = [
        "transaction_id",
        "ts",
        "amount_eur",
        "is_fraud",
        "is_fraud_true",
        "scenario",
        "p",
        "action",
        "outcome",
        "label_noise",
        *PROFILE_COLUMNS,
        "channel",
    ]
    df[keep].to_parquet(out_dir / "validation_decisions.parquet", index=False)
    single.to_csv(out_dir / "permutation_single.csv", lineterminator="\n")
    grouped.to_csv(out_dir / "permutation_grouped.csv", index=False, lineterminator="\n")
    shap_frame = attributions.add_prefix("shap__")
    shap_frame["base"] = base
    shap_frame["log_odds"] = log_odds
    sample[["transaction_id", "outcome", "p", "amount_eur", *cols]].join(shap_frame).to_parquet(
        out_dir / "shap_sample.parquet", index=False
    )
    for name, table in profiles.items():
        table.to_csv(out_dir / f"profile_{name}.csv", lineterminator="\n")
    for name, table in segments.items():
        table.to_csv(out_dir / f"segment_{name}.csv", lineterminator="\n")
    scenarios.to_csv(out_dir / "scenarios.csv", index=False, lineterminator="\n")
    summary = {
        "outcomes": df["outcome"].value_counts().to_dict(),
        "label_noise_by_outcome": pd.crosstab(df["outcome"], df["label_noise"]).to_dict(),
        "shap_rows": len(sample),
        "shap_additivity_error": err,
        "blocked": int((df["action"] == BLOCK).sum()),
        "seconds": round(time.perf_counter() - started),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    log.info("phase 9 analysis done in %.0fs", time.perf_counter() - started)
    return summary
