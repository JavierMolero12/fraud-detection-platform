"""Point-in-time features for ONE incoming transaction (the API path).

The offline build (sql/features/10..40) computes the same 50 features for every stored
transaction with window functions; this module computes them for a single new one with
lookups (sql/features/online/single_transaction.sql). tests/integration/test_feature_parity.py
checks that both give the same values (decision 017).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import psycopg

from fraudplatform.config import SQL_DIR, FeatureConfig
from fraudplatform.eda.data import query_features
from fraudplatform.features.catalog import FEATURE_NAMES

ONLINE_SQL = SQL_DIR / "features" / "online" / "single_transaction.sql"
MAX_FX_STALENESS = timedelta(days=7)


class UnknownEntity(ValueError):
    """Customer, card (of that customer), merchant or country not in core, or no usable FX rate."""


@dataclass(frozen=True)
class IncomingTransaction:
    customer_id: str
    card_id: str
    merchant_id: str
    ts: datetime  # timezone-aware
    amount: float
    currency: str
    country: str  # payment origin: terminal (card present) or IP (online) country
    payment_method: str
    device_id: str | None = None
    device_type: str | None = None  # only needed for a device never seen before
    operating_system: str | None = None


def amount_in_eur(conn: psycopg.Connection, txn: IncomingTransaction) -> float:
    """Same conversion as the data quality stage, rounded to cents.

    The rate is the latest one published on or before the payment's UTC day: identical to the
    data quality stage for stored data (a rate exists for every day), and still usable when the
    feed is a few days late. Older than MAX_FX_STALENESS, the payment is refused.
    """
    rate = Decimal(1)
    if txn.currency != "EUR":
        day = txn.ts.astimezone(UTC).date()
        row = conn.execute(
            "SELECT eur_per_unit, rate_date FROM core.fx_rates "
            "WHERE currency_code = %s AND rate_date <= %s ORDER BY rate_date DESC LIMIT 1",
            (txn.currency, day),
        ).fetchone()
        if row is None:
            raise UnknownEntity(f"no exchange rate for {txn.currency} on or before {day}")
        rate, rate_date = row
        if day - rate_date > MAX_FX_STALENESS:
            raise UnknownEntity(
                f"latest {txn.currency} exchange rate is from {rate_date}, "
                f"more than {MAX_FX_STALENESS.days} days before {day}"
            )
    eur = (Decimal(str(txn.amount)) * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return float(max(eur, Decimal("0.01")))


class OnlineFeatureBuilder:
    def __init__(self, cfg: FeatureConfig):
        self.sql = ONLINE_SQL.read_text(encoding="utf-8")
        self.settings = {
            "micro_amount_eur": cfg.micro_amount_eur,
            "night_start": cfg.night_hours[0],
            "night_end": cfg.night_hours[1],
            "impossible_speed_kmh": cfg.impossible_speed_kmh,
            "merchant_rate_prior": cfg.merchant_rate_prior,
            "merchant_rate_prior_strength": cfg.merchant_rate_prior_strength,
        }

    def frame(
        self, conn: psycopg.Connection, txn: IncomingTransaction, amount_eur: float | None = None
    ) -> pd.DataFrame:
        """One-row frame of the 50 features, typed exactly like load_features (same parser)."""
        if amount_eur is None:
            amount_eur = amount_in_eur(conn, txn)
        params = {
            **self.settings,
            "ts": txn.ts,
            "amount_eur": amount_eur,
            "payment_method": txn.payment_method,
            "txn_country": txn.country,
            "customer_id": txn.customer_id,
            "card_id": txn.card_id,
            "merchant_id": txn.merchant_id,
            "device_id": txn.device_id,
            "device_type": txn.device_type,
            "operating_system": txn.operating_system,
        }
        df = query_features(conn, self.sql, params)
        if df.empty:
            raise UnknownEntity(
                "unknown customer, card of that customer, merchant or country: "
                f"{txn.customer_id} / {txn.card_id} / {txn.merchant_id} / {txn.country}"
            )
        if tuple(df.columns) != FEATURE_NAMES:
            raise RuntimeError("online SQL columns do not match the feature catalog")
        return df
