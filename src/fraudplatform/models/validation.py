"""Cross-validation splits: temporal (expanding window with a gap) and random (for contrast).

Why not sklearn's TimeSeriesSplit? It splits by row position and its `gap` is a number of
rows. Our folds need calendar boundaries (whole test months) and a gap in days, so the
splitter works on timestamps. The output is a list of (train_idx, test_idx) arrays, which
any sklearn search accepts as `cv`.

    fold 1  [=========== train ===========]  gap  [ test: 2025-03 ]
    fold 2  [============= train =============]  gap  [ test: 2025-04 ]
    ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: date
    train_end: date  # exclusive
    test_start: date
    test_end: date  # exclusive


def _month_bounds(month: str) -> tuple[date, date]:
    start = date.fromisoformat(f"{month}-01")
    end = date(start.year + start.month // 12, start.month % 12 + 1, 1)
    return start, end


def temporal_folds(window_start: date, test_months: list[str], gap_days: int) -> list[Fold]:
    """Expanding-window folds: train from `window_start` up to `gap_days` before each test
    month, test on that month."""
    folds = []
    for month in test_months:
        test_start, test_end = _month_bounds(month)
        train_end = test_start - timedelta(days=gap_days)
        if train_end <= window_start:
            raise ValueError(f"fold {month}: no training data before the gap")
        folds.append(Fold(f"test {month}", window_start, train_end, test_start, test_end))
    return folds


def fold_indices(ts: pd.Series, folds: list[Fold]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row positions (not labels) of each fold's train and test rows."""
    day = ts.dt.tz_convert("UTC").dt.date if ts.dt.tz is not None else ts.dt.date
    day = day.to_numpy()
    splits = []
    for f in folds:
        train = np.flatnonzero((day >= f.train_start) & (day < f.train_end))
        test = np.flatnonzero((day >= f.test_start) & (day < f.test_end))
        splits.append((train, test))
    return splits


def random_folds(y, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled stratified K-fold over the same rows: the 'what not to do' comparison (H7)."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
