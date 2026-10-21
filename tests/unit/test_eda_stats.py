"""Statistics helpers checked against hand-computed or textbook values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudplatform.eda.stats import (
    benjamini_hochberg,
    cramers_v,
    precision_from_rates,
    rate_table,
    tukey_fences,
    univariate_auc,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.2775),  # normal interval would give [0, 0]: Wilson does not
        (5, 10, 0.2366, 0.7634),
        (10, 10, 0.7225, 1.0),
    ],
)
def test_wilson_interval_known_values(k, n, low, high):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=1e-4)
    assert hi == pytest.approx(high, abs=1e-4)


def test_wilson_interval_narrows_with_sample_size():
    lo_small, hi_small = wilson_interval(1, 100)
    lo_big, hi_big = wilson_interval(100, 10_000)
    assert hi_big - lo_big < hi_small - lo_small


def test_rate_table_lift_and_counts():
    df = pd.DataFrame(
        {"cat": ["a"] * 90 + ["b"] * 10, "is_fraud": [False] * 89 + [True] + [True] * 10}
    )
    t = rate_table(df, "cat").set_index("cat")
    assert t.loc["a", "transactions"] == 90 and t.loc["a", "frauds"] == 1
    assert t.loc["b", "fraud_rate"] == 1.0
    assert t.loc["b", "lift"] == pytest.approx(1.0 / 0.11)


def test_cramers_v_extremes():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.integers(0, 3, 20_000))
    assert cramers_v(x, x, bias_correction=False) == pytest.approx(1.0)
    independent = pd.Series(rng.integers(0, 3, 20_000))
    assert cramers_v(x, independent) < 0.02


def test_benjamini_hochberg_matches_reference():
    # Hand-computed: sorted p * n / rank = [.02, .02, .04, .04], then running minimum.
    p = [0.01, 0.04, 0.03, 0.005]
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_univariate_auc():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert univariate_auc([1, 2, 3, 4], labels) == 1.0
    assert univariate_auc([4, 3, 2, 1], labels) == 0.0
    assert univariate_auc([1, 1, 1, 1], labels) == 0.5
    assert univariate_auc([1, np.nan, 3, 4], labels) == 1.0  # NaN ignored


def test_tukey_fences():
    low, high = tukey_fences(np.arange(1, 101))  # Q1 = 25.75, Q3 = 75.25
    assert (low, high) == pytest.approx((25.75 - 1.5 * 49.5, 75.25 + 1.5 * 49.5))


def test_precision_from_rates_base_rate_effect():
    # 90% recall and only 1% false positives still give < 50% precision at 0.9% prevalence.
    assert precision_from_rates(0.009, 0.9, 0.01) == pytest.approx(0.4497, abs=1e-4)
