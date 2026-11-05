"""Explanations and error analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_models import make_frame

from fraudplatform.analysis.errors import add_outcomes, error_profile, segment_metrics
from fraudplatform.analysis.explain import (
    additivity_error,
    grouped_permutation_importance,
    shap_values,
    single_permutation_importance,
)
from fraudplatform.decision.costs import APPROVE, BLOCK, REVIEW
from fraudplatform.models import pipelines


@pytest.fixture(scope="module")
def fitted():
    """Gradient boosting with NATIVE categorical splits: the case shap.TreeExplainer gets wrong."""
    df = make_frame(4000, seed=7)
    cols = pipelines.model_columns("tree")
    model = pipelines.build_hist_gradient_boosting({"max_iter": 40}).fit(df[cols], df["is_fraud"])
    return model, df, cols


def test_shap_values_add_up_to_the_model_output(fitted):
    model, df, cols = fitted
    attributions, base, log_odds = shap_values(model, df[cols].iloc[:25], df[cols].iloc[100:160])
    assert attributions.shape == (25, len(cols))
    assert additivity_error(attributions, base, log_odds) < 1e-6


def test_the_informative_feature_gets_the_largest_attribution(fitted):
    model, df, cols = fitted
    risky = df[df["merchant_category"] == "gift_cards"].iloc[:20]
    attributions, _, _ = shap_values(model, risky[cols], df[cols].iloc[200:300])
    assert attributions.abs().mean().idxmax() in {"merchant_category", "card_txn_count_1h"}


def test_grouped_permutation_ranks_signal_above_noise_and_matches_single_for_one_feature():
    df = make_frame(4000, seed=8)
    cols = pipelines.model_columns("tree")
    model = pipelines.build_hist_gradient_boosting({"max_iter": 40}).fit(df[cols], df["is_fraud"])
    single = single_permutation_importance(model, df[cols], df["is_fraud"], n_repeats=3, seed=0)
    grouped = grouped_permutation_importance(
        model,
        df[cols],
        df["is_fraud"],
        {"signal": ["card_txn_count_1h"], "noise": ["customer_age"]},
        n_repeats=3,
        seed=0,
    ).set_index("group")
    assert grouped.loc["signal", "importance"] > 0.1
    assert abs(grouped.loc["noise", "importance"]) < 0.02
    # a one-feature group is the same experiment as single-feature permutation
    assert grouped.loc["signal", "importance"] == pytest.approx(
        single.loc["card_txn_count_1h", "importance"], abs=0.03
    )


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(8)],
            "is_fraud": [True, True, True, False, False, False, False, False],
            "is_fraud_true": [True, True, False, False, True, False, False, False],
            "scenario": ["ato", "stealth", "legit", "legit", "ring", "legit", "legit", "legit"],
            "action": [BLOCK, APPROVE, REVIEW, BLOCK, APPROVE, APPROVE, APPROVE, REVIEW],
            "p": [0.9, 0.01, 0.3, 0.8, 0.02, 0.001, 0.002, 0.2],
            "amount_eur": [500, 50, 100, 200, 80, 10, 20, 30.0],
            "channel": [
                "online",
                "store",
                "online",
                "online",
                "store",
                "store",
                "online",
                "online",
            ],
        }
    )


def test_outcomes_and_label_noise():
    out = add_outcomes(_decisions())
    assert list(out["outcome"]) == [
        "fraud blocked",
        "fraud approved (FN)",
        "fraud reviewed",
        "legit blocked (FP)",
        "legit approved",
        "legit approved",
        "legit approved",
        "legit reviewed",
    ]
    # t2 is labelled fraud but true legit (friendly fraud); t4 is unreported fraud
    assert out.loc[2, "label_noise"] == "friendly fraud"
    assert out.loc[4, "label_noise"] == "unreported fraud"


def test_error_profile_over_representation():
    out = add_outcomes(_decisions())
    table = error_profile(out, "channel")
    # 5 legit: 3 online, 2 store; the only FP is online -> share 1.0 vs 0.6 -> 1.67x
    assert table.loc["online", "fp_over_representation"] == pytest.approx(1 / 0.6)
    # 3 fraud: 2 online, 1 store; the only FN is in store -> share 1.0 vs 1/3 -> 3x
    assert table.loc["store", "fn_over_representation"] == pytest.approx(3.0)


def test_segment_metrics_counts():
    out = add_outcomes(_decisions())
    seg = segment_metrics(out, "channel", reviewer_recall=0.9, min_frauds=1)
    online = seg.loc["online"]
    assert online["frauds"] == 2 and online["fraud_stopped"] == 1.0
    assert online["block_precision"] == pytest.approx(0.5)  # 2 blocked, 1 fraud
    assert online["false_decline_rate"] == pytest.approx(1 / 3)
    assert np.isclose(online["fraud_eur_stopped"], (500 + 0.9 * 100) / 600)
