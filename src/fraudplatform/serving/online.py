"""Scoring of one payment at authorisation time: features -> probability -> decision -> log.

Review capacity online: decisions are made one at a time, so the day's review slots are given
first come, first served. Two concurrent requests could both see "59 of 60 used" and both
send a review; an advisory lock per UTC day makes the count-then-insert atomic. Only payments
whose best action is a review take the lock, so the common path (approve) never waits.

Idempotency: a retried call with the same transaction_id returns the decision already logged
(unique index on api transaction ids) instead of scoring, and counting a review, twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from fraudplatform.decision.costs import ACTION_NAMES, REVIEW
from fraudplatform.features.catalog import BOOLEAN, CATEGORICAL, FEATURE_NAMES
from fraudplatform.models.pipelines import prepare
from fraudplatform.serving.online_features import (
    IncomingTransaction,
    OnlineFeatureBuilder,
    amount_in_eur,
)
from fraudplatform.serving.scorer import Scorer

REVIEWS_TODAY = """
SELECT count(*) FROM scoring.predictions
WHERE source = 'api' AND decision = 'review'
  AND (transaction_ts AT TIME ZONE 'UTC')::date = %s
"""
INSERT = """
INSERT INTO scoring.predictions (transaction_id, source, model_version, policy_version,
    fraud_probability, risk_level, decision, amount_eur, transaction_ts, scored_at, latency_ms,
    features)
VALUES (%s, 'api', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id) WHERE source = 'api' DO NOTHING
RETURNING {returned}
"""
LOGGED = """
SELECT {returned} FROM scoring.predictions WHERE source = 'api' AND transaction_id = %s
"""
RETURNED = (
    "transaction_id, fraud_probability, risk_level, decision, model_version, policy_version, "
    "scored_at"
)
INSERT, LOGGED = INSERT.format(returned=RETURNED), LOGGED.format(returned=RETURNED)


@dataclass(frozen=True)
class Prediction:
    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    model_version: str
    policy_version: str
    scoring_timestamp: datetime


def logged_features(row: pd.Series) -> dict:
    """Model inputs as JSON values: booleans back to true/false, NaN (unknown) to null."""
    values = {}
    for name in FEATURE_NAMES:
        value = row[name]
        if name in CATEGORICAL:
            values[name] = str(value)
        elif pd.isna(value):
            values[name] = None
        elif name in BOOLEAN:
            values[name] = bool(value)
        else:
            values[name] = value.item() if hasattr(value, "item") else value
    return values


def _logged(conn: psycopg.Connection, transaction_id: str) -> Prediction | None:
    row = conn.execute(LOGGED, (transaction_id,)).fetchone()
    return Prediction(*row) if row else None


def score_payment(
    conn: psycopg.Connection,
    scorer: Scorer,
    features: OnlineFeatureBuilder,
    transaction_id: str,
    txn: IncomingTransaction,
    log_features: bool = True,
    timings: dict[str, float] | None = None,
) -> Prediction:
    """`conn` must be in autocommit mode: the decision and its log are one explicit transaction.

    latency_ms in the log = time to the decision, without writing the log itself. `timings`,
    if given, receives the milliseconds of each step (fraud serve profile).
    """
    start = last = time.perf_counter()

    def step(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        if timings is not None:
            timings[name] = (now - last) * 1000
        last = now

    if (previous := _logged(conn, transaction_id)) is not None:
        return previous
    step("idempotency_check")
    amount_eur = amount_in_eur(conn, txn)
    step("exchange_rate")
    raw = features.frame(conn, txn, amount_eur)
    step("features_sql")
    frame = prepare(raw)  # same frame as batch scoring
    step("prepare")
    bundle = scorer.bundle
    q = float(scorer.probability(frame)[0])
    step("predict_calibrate")
    best, fallback = scorer.preferred_actions(q, amount_eur)
    day = txn.ts.astimezone(UTC).date()
    logged = Jsonb(logged_features(frame.iloc[0])) if log_features else None

    with conn.transaction():
        action = best
        if best == REVIEW:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('review-capacity'), %s)", (day.toordinal(),)
            )
            used = conn.execute(REVIEWS_TODAY, (day,)).fetchone()[0]
            if used >= bundle.capacity:
                action = fallback
        step("decide")
        decided_ms = (time.perf_counter() - start) * 1000
        row = conn.execute(
            INSERT,
            (
                transaction_id,
                bundle.model_version,
                bundle.policy_version,
                q,
                str(scorer.risk_level([q])[0]),
                str(ACTION_NAMES[action]),
                amount_eur,
                txn.ts,
                datetime.now(UTC),
                decided_ms,
                logged,
            ),
        ).fetchone()
    step("write_log")
    # no row: a concurrent call with the same id won the insert, return its decision
    return Prediction(*row) if row else _logged(conn, transaction_id)
