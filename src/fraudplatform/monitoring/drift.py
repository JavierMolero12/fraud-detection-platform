"""Distribution drift: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS).

Bins are fitted once on the reference period and reused for every monitoring window, so that a
window is always compared on the same grid.

- numeric: reference deciles as upper-inclusive cut points (ties merged). A zero-inflated count
  (most velocity features) keeps 0 in its own bin instead of hiding it inside a decile;
- categorical / boolean: one bin per reference value, plus "other" for values never seen;
- missing values: always their own bin ("no device" is information, not noise).

PSI = sum((cur - ref) * ln(cur / ref)) over bins; shares are smoothed with EPS so that an empty
bin does not give an infinite PSI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


@dataclass(frozen=True)
class Binning:
    numeric: bool
    edges: np.ndarray | None = None  # numeric: upper-inclusive cut points
    categories: tuple[str, ...] = ()  # categorical: reference values

    @property
    def n_bins(self) -> int:
        # numeric: len(edges) + 1 value bins; categorical: values + "other"; both: + missing
        return (len(self.edges) + 1 if self.numeric else len(self.categories) + 1) + 1


def fit_binning(reference: pd.Series, numeric: bool, bins: int = 10) -> Binning:
    if numeric:
        values = reference.to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        quantiles = np.linspace(0, 1, bins + 1)[1:-1]
        edges = np.unique(np.quantile(values, quantiles)) if len(values) else np.array([])
        return Binning(numeric=True, edges=edges)
    categories = tuple(sorted(reference.dropna().astype(str).unique()))
    return Binning(numeric=False, categories=categories)


def bin_counts(values: pd.Series, binning: Binning) -> np.ndarray:
    counts = np.zeros(binning.n_bins)
    missing = values.isna().to_numpy()
    counts[-1] = missing.sum()
    present = values[~missing]
    if binning.numeric:
        idx = np.searchsorted(binning.edges, present.to_numpy(dtype=float), side="left")
    else:
        lookup = {c: i for i, c in enumerate(binning.categories)}
        other = len(binning.categories)
        idx = present.astype(str).map(lookup).fillna(other).to_numpy(dtype=int)
    counts[: binning.n_bins - 1] += np.bincount(idx, minlength=binning.n_bins - 1)
    return counts


def shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else counts


def psi(reference_shares: np.ndarray, current_shares: np.ndarray, eps: float = EPS) -> float:
    ref = (reference_shares + eps) / (1 + eps * len(reference_shares))
    cur = (current_shares + eps) / (1 + eps * len(current_shares))
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Largest gap between the two empirical CDFs (missing values dropped).

    Reported, not alerted on: with tens of thousands of rows every difference is "significant",
    so only the size of the gap (and PSI) is informative.
    """
    ref, cur = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    return float(ks_2samp(ref, cur).statistic)
