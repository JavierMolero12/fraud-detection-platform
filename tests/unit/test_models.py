"""Baselines, pipelines and metrics on small synthetic frames (no database needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudplatform.config import load_model_config
from fraudplatform.features.catalog import BOOLEAN, CATEGORICAL, NUMERIC
from fraudplatform.models import baselines, pipelines
from fraudplatform.models.metrics import (
    alert_metrics,
    bootstrap_pr_auc,
    bootstrap_pr_auc_difference,
    evaluate,
    top_k_mask,
)

LEVELS = {
    "payment_method": ["chip_pin", "contactless", "ecommerce", "mobile_wallet"],
    "merchant_category": ["grocery", "electronics", "gift_cards", "luxury"],
    "merchant_risk_level": ["low", "medium", "high"],
    "card_type": ["debit", "credit"],
    "customer_segment": ["standard", "student"],
    "device_type": ["mobile", "desktop", "none"],
    "operating_system": ["ios", "android", "none"],
}


def make_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Random rows with every model column; fraud depends on two features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.exponential(3, n) for c in NUMERIC})
    df["local_hour"] = rng.integers(0, 24, n).astype(float)
    df["log_amount"] = np.log(df["amount_eur"] + 1)
    for c in BOOLEAN:
        df[c] = rng.choice([0.0, 1.0, np.nan], n, p=[0.6, 0.3, 0.1])
    for c in CATEGORICAL:
        df[c] = rng.choice(LEVELS[c], n)
    df.loc[rng.random(n) < 0.1, "device_age_days"] = np.nan  # NULL = no device
    logit = -4 + 0.8 * df["card_txn_count_1h"] + 2 * (df["merchant_category"] == "gift_cards")
    df["is_fraud"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return df


@pytest.fixture(scope="module")
def cfg():
    return load_model_config()


# ---------------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------------
def test_rules_fire_on_the_intended_rows(cfg):
    rows = make_frame(4, seed=1)
    for c in BOOLEAN:
        rows[c] = 0.0
    rows[["card_txn_count_1h", "card_micro_count_1h", "device_unique_cards_24h"]] = 0
    rows["amount_eur"] = 50.0
    rows.loc[0, "card_txn_count_1h"] = 3  # burst
    rows.loc[1, ["card_micro_count_1h", "is_card_present"]] = [1, 0.0]  # card testing
    rows.loc[2, ["device_new_for_customer", "amount_eur"]] = [1.0, 500.0]  # new device, high EUR
    rows.loc[3, "device_new_for_customer"] = np.nan  # NULL never fires
    flags = baselines.rule_flags(rows, cfg.rules)
    assert flags.loc[0, "burst"] and flags.loc[1, "card_testing"]
    assert flags.loc[2, "new_device_high_amount"]
    assert not flags.loc[3].any()
    assert list(baselines.rules_score(rows, cfg.rules)) == [1, 1, 1, 0]


def test_approve_all_has_base_rate_pr_auc():
    y = np.array([True] * 10 + [False] * 990)
    metrics = evaluate(y, np.zeros(1000), np.ones(1000), np.zeros(1000), 0.01, False)
    assert metrics["pr_auc"] == pytest.approx(0.01)
    assert metrics["roc_auc"] == 0.5


# ---------------------------------------------------------------------------------------------
# pipelines
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("builder", "kind", "params"),
    [
        (pipelines.build_logistic_regression, "linear", {"max_iter": 500}),
        (pipelines.build_random_forest, "tree", {"n_estimators": 20, "min_samples_leaf": 5}),
        (pipelines.build_hist_gradient_boosting, "tree", {"max_iter": 30}),
    ],
)
def test_pipelines_learn_and_output_probabilities(builder, kind, params):
    train, test = make_frame(4000, seed=2), make_frame(1500, seed=3)
    test.loc[0, "merchant_category"] = "never_seen_before"  # unseen category must not crash
    columns = pipelines.model_columns(kind)
    model = builder(params, random_state=0).fit(train[columns], train["is_fraud"])
    p = model.predict_proba(test[columns])[:, 1]
    assert p.shape == (len(test),)
    assert np.all((p >= 0) & (p <= 1))
    assert (
        evaluate(test["is_fraud"], p, test["amount_eur"], np.zeros(len(test)), 0.05, True)[
            "roc_auc"
        ]
        > 0.7
    )


@pytest.mark.parametrize(
    ("builder", "kind", "params"),
    [
        (pipelines.build_logistic_regression, "linear", {"max_iter": 300}),
        (pipelines.build_random_forest, "tree", {"n_estimators": 5}),
        (pipelines.build_hist_gradient_boosting, "tree", {"max_iter": 10}),
    ],
)
def test_models_round_trip_with_skops_and_the_trusted_list(builder, kind, params, tmp_path):
    """Models are saved with skops (no arbitrary code on load); the trusted list must cover
    every type a pipeline needs, and the reloaded model must predict identically."""
    import skops.io as sio

    df = make_frame(1500, seed=5)
    columns = pipelines.model_columns(kind)
    model = builder(params).fit(df[columns], df["is_fraud"])
    path = tmp_path / "model.skops"
    sio.dump(model, path)
    assert set(sio.get_untrusted_types(file=path)) <= set(pipelines.SKOPS_TRUSTED_TYPES)
    reloaded = sio.load(path, trusted=pipelines.SKOPS_TRUSTED_TYPES)
    np.testing.assert_allclose(
        reloaded.predict_proba(df[columns]), model.predict_proba(df[columns])
    )


def test_linear_and_tree_inputs_avoid_duplicate_amounts():
    assert "amount_eur" not in pipelines.model_columns("linear")
    assert "log_amount" not in pipelines.model_columns("tree")
    assert "local_hour" in pipelines.LINEAR_CATEGORICAL  # U-shape: one-hot for the linear model


def test_signed_log1p_keeps_sign_and_order():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    out = pipelines.signed_log1p(x)
    assert np.all(np.diff(out) > 0) and out[2] == 0


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def test_top_k_breaks_ties_at_random_but_reproducibly():
    score = np.array([1, 1, 1, 1, 0.0])
    a, b = top_k_mask(score, 2, seed=1), top_k_mask(score, 2, seed=1)
    assert a.sum() == 2 and not a[4] and np.array_equal(a, b)


def test_alert_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    score = np.array([0.9, 0.1, 0.8, 0, 0, 0, 0, 0, 0, 0.7])
    amount = np.array([100, 50, 10, 1, 1, 1, 1, 1, 1, 30.0])
    m = alert_metrics(y, score, amount, alert_rate=0.3)  # 3 alerts: rows 0, 2, 9
    assert m["precision_at_alert_rate"] == pytest.approx(2 / 3)
    assert m["recall_at_alert_rate"] == pytest.approx(2 / 3)
    assert m["amount_recall_at_alert_rate"] == pytest.approx(130 / 180)


def test_day_bootstrap_is_reproducible_and_brackets_the_estimate():
    df = make_frame(5000, seed=4)
    score = df["card_txn_count_1h"].to_numpy()
    days = np.repeat(np.arange(50), 100)
    lo1, hi1 = bootstrap_pr_auc(df["is_fraud"], score, days, reps=50, seed=3)
    lo2, hi2 = bootstrap_pr_auc(df["is_fraud"], score, days, reps=50, seed=3)
    point = evaluate(df["is_fraud"], score, df["amount_eur"], days, 0.05, False)["pr_auc"]
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < point < hi1


def test_paired_bootstrap_detects_a_better_score():
    df = make_frame(6000, seed=6)
    days = np.repeat(np.arange(60), 100)
    good = df["card_txn_count_1h"] + 3 * (df["merchant_category"] == "gift_cards")
    weak = df["card_txn_count_1h"].to_numpy()
    out = bootstrap_pr_auc_difference(df["is_fraud"], good, weak, days, reps=60, seed=1)
    assert out["difference"] > 0 and out["ci_low"] > 0 and out["share_a_better"] > 0.95
    same = bootstrap_pr_auc_difference(df["is_fraud"], weak, weak, days, reps=20, seed=1)
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0


def test_probability_metrics_only_for_probabilistic_scores():
    y = np.array([True, False, False, True])
    assert "brier" not in evaluate(y, [5, 1, 2, 9], [1] * 4, [0] * 4, 0.5, False)
    assert "brier" in evaluate(y, [0.9, 0.1, 0.2, 0.8], [1] * 4, [0] * 4, 0.5, True)


def test_code_version_from_git_in_a_checkout():
    from fraudplatform.models.tracking import code_version

    commit, dirty = code_version()
    assert len(commit) == 40 and dirty in ("True", "False")


def test_code_version_without_git_uses_the_commit_baked_into_the_image(monkeypatch):
    from fraudplatform.models import tracking

    monkeypatch.setattr(tracking, "_git", lambda *args: None)  # a Docker image: no git
    monkeypatch.setenv("FRAUD_GIT_COMMIT", "a" * 40)
    assert tracking.code_version() == ("a" * 40, "unknown")
    monkeypatch.delenv("FRAUD_GIT_COMMIT")
    assert tracking.code_version() == ("unknown", "unknown")  # never a guess, e.g. dirty=True
