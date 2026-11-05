"""Reproducible randomness and time helpers shared by the simulator components."""

from __future__ import annotations

import zlib
from datetime import date

import numpy as np
import pandas as pd

DAY = 86_400
HOUR = 3_600


class RngFactory:
    """Named, independent random streams derived from a single seed.

    Each component asks for its own stream by name. Adding or removing draws in one component
    therefore does not shift the random numbers another component sees, which keeps the dataset
    stable when the simulator evolves.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def get(self, name: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, zlib.crc32(name.encode())])


def lognormal_mean_one(rng: np.random.Generator, sigma: float, size: int) -> np.ndarray:
    """Lognormal multipliers with expected value 1 (so they rescale without biasing totals)."""
    return rng.lognormal(-(sigma**2) / 2, sigma, size)


def weighted_choice(rng: np.random.Generator, weights: np.ndarray, size: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return rng.choice(len(weights), size=size, p=weights / weights.sum())


def random_ids(rng: np.random.Generator, n: int, prefix: str, hex_digits: int) -> np.ndarray:
    """Unique random hexadecimal IDs.

    Random rather than sequential so that an ID never encodes creation order: a model must not
    be able to learn "high IDs are recent" (temporal leakage through identifiers).
    """
    draws = rng.integers(0, 16**hex_digits, size=int(n * 1.05) + 16, dtype=np.uint64)
    unique = draws[np.sort(np.unique(draws, return_index=True)[1])][:n]
    if len(unique) < n:
        raise RuntimeError("ID space too small for the requested number of IDs")
    return np.array([f"{prefix}{x:0{hex_digits}x}" for x in unique])


class Calendar:
    """Simulation period with day-level seasonality, working in integer UTC epoch seconds."""

    def __init__(self, start: date, end: date):
        self.days = np.arange(
            np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D")
        )
        self.n_days = len(self.days)
        self.day_start_s = self.days.astype("datetime64[s]").astype(np.int64)
        self.start_s = int(self.day_start_s[0])
        self.end_s = int(self.day_start_s[-1]) + DAY - 1
        self.weights = self._day_weights(self.days)
        self._cdf = np.cumsum(self.weights) / self.weights.sum()

    @staticmethod
    def _day_weights(days: np.ndarray) -> np.ndarray:
        weekday = (days.astype(np.int64) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
        weekday_factor = np.array([0.95, 0.97, 1.00, 1.02, 1.12, 1.15, 0.85])[weekday]
        day_of_month = (days - days.astype("datetime64[M]")).astype(np.int64) + 1
        payday_factor = np.where(day_of_month <= 3, 1.10, np.where(day_of_month >= 25, 1.05, 1.0))
        month = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
        december_factor = np.where(month == 12, 1.15, 1.0)
        return weekday_factor * payday_factor * december_factor

    def day_index(self, ts_s: np.ndarray | int) -> np.ndarray:
        return np.clip((np.asarray(ts_s) - self.start_s) // DAY, 0, self.n_days - 1)

    def sample_days(self, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Sample one day index per row in the inclusive range [lo, hi], seasonality-weighted."""
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        cdf = self._cdf
        c_lo = np.where(lo > 0, cdf[np.maximum(lo - 1, 0)], 0.0)
        u = c_lo + rng.random(len(lo)) * (cdf[hi] - c_lo)
        return np.clip(np.searchsorted(cdf, u, side="left"), lo, hi)

    def sample_times(
        self,
        rng: np.random.Generator,
        lo_s: np.ndarray,
        hi_s: np.ndarray,
        hour_weights: np.ndarray,
        utc_offset_h: np.ndarray,
    ) -> np.ndarray:
        """Sample UTC timestamps in [lo_s, hi_s] with day seasonality and a LOCAL hour curve."""
        lo_s = np.asarray(lo_s, dtype=np.int64)
        hi_s = np.asarray(hi_s, dtype=np.int64)
        days = self.sample_days(rng, self.day_index(lo_s), self.day_index(hi_s))
        local = weighted_choice(rng, hour_weights, len(days)) * HOUR + rng.integers(
            0, HOUR, len(days)
        )
        ts = self.day_start_s[days] + local - np.asarray(utc_offset_h, dtype=np.int64) * HOUR
        return np.clip(ts, lo_s, hi_s)


def to_iso(ts_s: np.ndarray) -> np.ndarray:
    """Epoch seconds -> ISO-8601 UTC strings ('2025-04-10T14:32:11Z')."""
    return np.char.add(np.datetime_as_string(np.asarray(ts_s).astype("datetime64[s]")), "Z")


def iso_to_epoch(values) -> np.ndarray:
    """ISO-8601 strings -> epoch seconds, independent of the datetime resolution pandas picks."""
    dt = pd.to_datetime(pd.Series(values), utc=True)
    return ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()


def to_date(ts_s: np.ndarray) -> np.ndarray:
    return np.datetime_as_string(np.asarray(ts_s).astype("datetime64[s]").astype("datetime64[D]"))
