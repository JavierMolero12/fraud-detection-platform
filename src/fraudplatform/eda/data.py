"""Data access for the exploratory analysis."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
import psycopg
from psycopg import sql

from fraudplatform.config import SQL_DIR, Splits

FRAME_SQL = SQL_DIR / "eda" / "transactions_frame.sql"
CATEGORICAL = [
    "payment_method",
    "txn_country",
    "merchant_category",
    "merchant_channel",
    "merchant_risk_level",
    "merchant_country",
    "customer_segment",
    "home_country",
    "card_type",
    "device_type",
    "operating_system",
    "label_source",
]


def load_transactions(conn: psycopg.Connection, start: date, end: date) -> pd.DataFrame:
    """Transactions in [start, end) as a typed DataFrame.

    Streams the result with COPY ... TO STDOUT (CSV) instead of fetching Python tuples:
    for ~1.4M rows this is several times faster and uses far less memory.
    """
    query = sql.SQL(FRAME_SQL.read_text(encoding="utf-8")).format(
        start=sql.Literal(start), end=sql.Literal(end)
    )
    buffer = io.BytesIO()
    with conn.cursor() as cur:
        copy_sql = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(query)
        with cur.copy(copy_sql) as copy:
            for chunk in copy:
                buffer.write(chunk)
    buffer.seek(0)
    df = pd.read_csv(buffer, dtype={c: "category" for c in CATEGORICAL})
    for col in ("ts", "reported_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    df["local_ts"] = pd.to_datetime(df["local_ts"], format="ISO8601")
    df["is_fraud"] = df["is_fraud"].map({"t": True, "f": False}).astype(bool)
    return df


def split_sizes(conn: psycopg.Connection, splits: Splits) -> pd.DataFrame:
    """Row count per split. Deliberately no fraud statistics outside the train period."""
    rows = []
    for name, period in splits.items():
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM core.transactions WHERE ts >= %s AND ts < %s",
            (period.start, period.end),
        ).fetchone()
        rows.append({"split": name, "start": period.start, "end": period.end, "transactions": n})
    return pd.DataFrame(rows)
