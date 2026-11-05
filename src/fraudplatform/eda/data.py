"""Data access for the exploratory analysis."""

from __future__ import annotations

import io
from collections.abc import Iterable
from datetime import date

import pandas as pd
import psycopg
from psycopg import sql

from fraudplatform.config import SQL_DIR, Splits
from fraudplatform.features import catalog

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


def _copy_to_frame(
    conn: psycopg.Connection,
    query: sql.Composable,
    categorical: Iterable[str],
    params: dict | None = None,
) -> pd.DataFrame:
    """Run a query with COPY ... TO STDOUT (CSV) and parse it with pandas.

    For ~1.4M rows this is several times faster than fetching Python tuples and uses far
    less memory. `params` are bound client-side: COPY cannot take server-side parameters.
    """
    buffer = io.BytesIO()
    with psycopg.ClientCursor(conn) if params else conn.cursor() as cur:
        copy_sql = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(query)
        with cur.copy(copy_sql, params) as copy:
            for chunk in copy:
                buffer.write(chunk)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype={c: "category" for c in categorical})


def _to_bool(series: pd.Series) -> pd.Series:
    """PostgreSQL CSV booleans ('t'/'f'/empty) -> float 1/0/NaN, keeping NULL visible."""
    return series.map({"t": 1.0, "f": 0.0})


def load_transactions(conn: psycopg.Connection, start: date, end: date) -> pd.DataFrame:
    """Transactions in [start, end) as a typed DataFrame."""
    query = sql.SQL(FRAME_SQL.read_text(encoding="utf-8")).format(
        start=sql.Literal(start), end=sql.Literal(end)
    )
    df = _copy_to_frame(conn, query, CATEGORICAL)
    for col in ("ts", "reported_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    df["local_ts"] = pd.to_datetime(df["local_ts"], format="ISO8601")
    df["is_fraud"] = df["is_fraud"].map({"t": True, "f": False}).astype(bool)
    return df


def load_features(
    conn: psycopg.Connection,
    start: date,
    end: date,
    include_leaky: bool = False,
    with_label: bool = True,
) -> pd.DataFrame:
    """Feature rows with ts in [start, end) plus the eventual label (retrospective view).

    Booleans come back as 1.0 / 0.0 / NaN so that "unknown" (no device, no history) stays
    distinguishable from False. Scoring jobs pass with_label=False: they never read labels.
    """
    leaky_cols = list(catalog.LEAKY_FEATURES) if include_leaky else []
    query = sql.SQL(
        """
        SELECT f.* {leaky} {label}
        FROM features.transaction_features f
        {label_join}
        {leaky_join}
        WHERE f.ts >= {start} AND f.ts < {end}
        """
    ).format(
        leaky=sql.SQL("").join(sql.SQL(", x.{}").format(sql.Identifier(c)) for c in leaky_cols),
        label=sql.SQL(", coalesce(l.is_fraud, false) AS is_fraud" if with_label else ""),
        label_join=sql.SQL(
            "LEFT JOIN core.fraud_labels l USING (transaction_id)" if with_label else ""
        ),
        leaky_join=sql.SQL(
            "JOIN features.transaction_features_leaky x USING (transaction_id)"
            if include_leaky
            else ""
        ),
        start=sql.Literal(start),
        end=sql.Literal(end),
    )
    df = _copy_to_frame(conn, query, catalog.CATEGORICAL)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    for col in catalog.BOOLEAN:
        df[col] = _to_bool(df[col])
    if with_label:
        df["is_fraud"] = df["is_fraud"].map({"t": True, "f": False}).astype(bool)
    return df


def query_features(conn: psycopg.Connection, query: str, params: dict) -> pd.DataFrame:
    """Rows of another feature query (the API's single-transaction SQL), parsed exactly like
    load_features.

    The models were trained on frames from this parser. pandas' default CSV float parser is
    not correctly rounded (it can land 1 ulp away, e.g. 0.002707006369426752 ->
    0.0027070063694267), and a tree split can sit exactly on a repeated training value, so the
    API must reproduce the training input bit for bit, not just "within 1e-9".
    """
    df = _copy_to_frame(conn, sql.SQL(query), catalog.CATEGORICAL, params)
    for col in catalog.BOOLEAN:
        df[col] = _to_bool(df[col])
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
