from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fraudplatform.config import load_model_config
from fraudplatform.models.validation import fold_indices, random_folds, temporal_folds

START = date(2024, 10, 1)


def test_folds_respect_order_gap_and_expand():
    folds = temporal_folds(START, ["2025-03", "2025-04", "2025-05"], gap_days=14)
    assert [f.test_start for f in folds] == [date(2025, 3, 1), date(2025, 4, 1), date(2025, 5, 1)]
    assert folds[0].train_end == date(2025, 2, 15)  # 14 days before the test month
    assert folds[2].test_end == date(2025, 6, 1)
    for f in folds:
        assert f.train_start == START and f.train_end < f.test_start
    assert folds[0].train_end < folds[1].train_end < folds[2].train_end  # expanding


def test_december_month_bounds():
    (fold,) = temporal_folds(START, ["2024-12"], gap_days=0)
    assert (fold.test_start, fold.test_end) == (date(2024, 12, 1), date(2025, 1, 1))


def test_fold_without_training_data_is_rejected():
    with pytest.raises(ValueError):
        temporal_folds(START, ["2024-10"], gap_days=14)


def test_fold_indices_never_train_on_or_after_the_test_period():
    ts = pd.Series(pd.date_range("2024-10-01", "2025-07-01", freq="6h", tz="UTC", inclusive="left"))
    folds = temporal_folds(START, ["2025-03", "2025-06"], gap_days=14)
    for (train, test), f in zip(fold_indices(ts, folds), folds, strict=True):
        assert ts.iloc[train].max() < pd.Timestamp(f.test_start, tz="UTC") - pd.Timedelta(days=13)
        assert ts.iloc[test].min() >= pd.Timestamp(f.test_start, tz="UTC")
        assert ts.iloc[test].max() < pd.Timestamp(f.test_end, tz="UTC")
        assert not set(train) & set(test)


def test_random_folds_cover_every_row_once_and_keep_the_fraud_rate():
    y = np.array([True] * 50 + [False] * 950)
    folds = random_folds(y, n_splits=4, seed=0)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests) == list(range(1000))
    assert all(abs(y[t].mean() - 0.05) < 0.01 for _, t in folds)


def test_configured_folds_fit_inside_the_train_window():
    cv = load_model_config().cross_validation
    folds = temporal_folds(START, cv.test_months, cv.gap_days)
    assert folds[-1].test_end <= date(2025, 7, 1)  # never reaches validation


def test_search_space_parsing_and_leaky_substitution():
    from fraudplatform.features.catalog import LEAKY_FEATURES
    from fraudplatform.models.tuning import parse_space, with_leaky_values

    space = parse_space({"learning_rate": {"loguniform": [0.01, 0.1]}, "max_iter": [100, 200]})
    assert space["model__max_iter"] == [100, 200]
    draws = space["model__learning_rate"].rvs(size=200, random_state=0)
    assert draws.min() >= 0.01 and draws.max() <= 0.1

    df = pd.DataFrame(
        {**{k: [1.0] for k in LEAKY_FEATURES}, **{v: [0.0] for v in LEAKY_FEATURES.values()}}
    )
    leaky = with_leaky_values(df)
    assert all(leaky[clean].iloc[0] == 1.0 for clean in LEAKY_FEATURES.values())
    assert list(leaky.columns) == list(df.columns)  # same names: the pipeline is unchanged
