"""Performance monitoring with delayed labels.

A fraud is only known once it is reported, a median of ~2 weeks after the payment and up to ~4
months. Two consequences:

1. Early signal, delay-adjusted calibration: by date D the model "expects" sum(p_i * F(D - ts_i))
   reported frauds among recent payments, where F is the share of frauds reported within a given
   age. Observed / expected far from 1 means the model's probabilities no longer match reality
   (more fraud than predicted: a pattern it does not see). Usable weeks before labels mature.
2. Late but direct signal: recall on cohorts old enough for most labels to be in ("mature").
   On young cohorts recall is biased: the fraud that is reported first (card testing) is also
   the easiest to catch.

F is estimated once, at deployment, from frauds old enough to have all their labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, poisson


@dataclass(frozen=True)
class DelayDistribution:
    delays_days: np.ndarray  # sorted reporting delays of reported frauds

    @classmethod
    def fit(cls, delays_days) -> DelayDistribution:
        delays = np.sort(np.asarray(delays_days, dtype=float))
        if len(delays) == 0:
            raise ValueError("no reported frauds to estimate the reporting delay from")
        return cls(delays)

    def cdf(self, age_days) -> np.ndarray:
        """Share of eventually reported frauds already reported at this age."""
        age = np.asarray(age_days, dtype=float)
        return np.searchsorted(self.delays_days, age, side="right") / len(self.delays_days)

    def age_for_share(self, share: float) -> float:
        return float(np.quantile(self.delays_days, share))


def expected_reported(p, age_days, delay: DelayDistribution) -> float:
    """Frauds the model expects to be reported by now among these payments."""
    return float(np.sum(np.asarray(p, dtype=float) * delay.cdf(age_days)))


def poisson_two_sided_p(observed: int, expected: float) -> float:
    if expected <= 0:
        return float("nan")
    low = poisson.cdf(observed, expected)
    high = poisson.sf(observed - 1, expected)
    return float(min(1.0, 2 * min(low, high)))


def recall_drop_p(caught: int, frauds: int, reference_recall: float) -> float:
    """One-sided: probability of catching this few if recall were still the reference one."""
    if frauds == 0:
        return float("nan")
    return float(binom.cdf(caught, frauds, reference_recall))
