"""Non-ML baselines. A model is only worth deploying if it beats these.

B0 approve-all : every transaction gets the same score (the "do nothing" reference)
B1 amount      : rank by amount (the classic single rule "block if amount > X")
B2 rules       : number of expert rules fired, written from the EDA on the train period
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fraudplatform.config import RuleParams


def approve_all_score(df: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(df))


def amount_score(df: pd.DataFrame) -> np.ndarray:
    return df["amount_eur"].to_numpy(dtype=float)


def _flag(series: pd.Series) -> pd.Series:
    """Boolean features arrive as 1.0 / 0.0 / NaN; NULL never fires a rule."""
    return series.eq(1.0)


def rule_definitions(p: RuleParams) -> dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    """Name -> (description, function returning a boolean Series)."""
    not_present = lambda d: ~_flag(d["is_card_present"])  # noqa: E731
    return {
        "burst": (
            f"card used >= {p.burst_card_txn_1h} times in the previous hour",
            lambda d: d["card_txn_count_1h"] >= p.burst_card_txn_1h,
        ),
        "card_testing": (
            f">= {p.card_testing_micro_1h} micro-charge(s) in the previous hour, card not present",
            lambda d: (d["card_micro_count_1h"] >= p.card_testing_micro_1h) & not_present(d),
        ),
        "new_device_high_amount": (
            f"first use of this device by the customer and amount > {p.new_device_amount_eur} EUR",
            lambda d: (
                _flag(d["device_new_for_customer"]) & (d["amount_eur"] > p.new_device_amount_eur)
            ),
        ),
        "cross_border_online": (
            "card-not-present payment from another country than the customer's",
            lambda d: _flag(d["cross_border"]) & not_present(d),
        ),
        "impossible_travel": (
            "faster than a plane since the previous payment",
            lambda d: _flag(d["impossible_travel"]),
        ),
        "device_many_cards": (
            f">= {p.device_many_cards_24h} distinct cards on the device in 24 h",
            lambda d: d["device_unique_cards_24h"] >= p.device_many_cards_24h,
        ),
        "night_risky_category": (
            f"night hours and category in {p.risky_categories}",
            lambda d: _flag(d["is_night"]) & d["merchant_category"].isin(p.risky_categories),
        ),
    }


def rule_flags(df: pd.DataFrame, params: RuleParams) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: fn(df).fillna(False).astype(bool)
            for name, (_, fn) in rule_definitions(params).items()
        },
        index=df.index,
    )


def rules_score(df: pd.DataFrame, params: RuleParams) -> np.ndarray:
    return rule_flags(df, params).sum(axis=1).to_numpy(dtype=float)
