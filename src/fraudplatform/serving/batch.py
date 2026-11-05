"""Batch scoring of a period with the frozen bundle.

Reads the offline features (features.transaction_features, the same table the model was
trained from), never the labels. Writes one CSV row per transaction and appends the
decisions to scoring.predictions (source = 'batch') for monitoring (Phase 12).

Batch differs from the API in one point only: it sees a whole day at once, so the day's review
slots go to the largest review benefits (bayes_actions), while the API hands them out first
come, first served. With the Phase 8 policy the capacity has not bound so far (busiest
production day: 41 of 60 reviews), so both give the same decisions.
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import psycopg

from fraudplatform.decision.costs import ACTION_NAMES, REVIEW
from fraudplatform.eda.data import load_features
from fraudplatform.models.pipelines import prepare
from fraudplatform.serving.scorer import Scorer

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "transaction_id",
    "fraud_probability",
    "risk_level",
    "decision",
    "model_version",
    "scoring_timestamp",
]


def score_period(
    conn: psycopg.Connection, scorer: Scorer, start: date, end: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(output rows, rows for scoring.predictions) for transactions with ts in [start, end)."""
    df = prepare(load_features(conn, start, end, with_label=False))
    if df.empty:
        raise ValueError(f"no feature rows between {start} and {end}")
    df = df.sort_values(["ts", "transaction_id"], ignore_index=True)
    q = scorer.probability(df)
    amount = df["amount_eur"].to_numpy(dtype=float)
    actions = scorer.batch_actions(q, amount, df["ts"].dt.date.to_numpy())
    scored_at = datetime.now(UTC)
    out = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"],
            "fraud_probability": q,
            "risk_level": scorer.risk_level(q),
            "decision": ACTION_NAMES[actions],
            "model_version": scorer.bundle.model_version,
            "scoring_timestamp": scored_at.isoformat(),
        }
    )
    log_rows = out.assign(
        policy_version=scorer.bundle.policy_version,
        amount_eur=amount,
        transaction_ts=df["ts"],
    )
    return out, log_rows


def write_predictions(conn: psycopg.Connection, rows: pd.DataFrame, start: date, end: date) -> None:
    """Batch decisions -> scoring.predictions with COPY.

    A re-run of the same period with the same model replaces the earlier rows (one
    transaction) instead of duplicating them; other model versions are kept for comparison.
    """
    frame = pd.DataFrame(
        {
            "transaction_id": rows["transaction_id"],
            "source": "batch",
            "model_version": rows["model_version"],
            "policy_version": rows["policy_version"],
            "fraud_probability": rows["fraud_probability"].map(repr),  # exact float
            "risk_level": rows["risk_level"],
            "decision": rows["decision"],
            "amount_eur": rows["amount_eur"].map("{:.2f}".format),
            "transaction_ts": rows["transaction_ts"].map(lambda t: t.isoformat()),
            "scored_at": rows["scoring_timestamp"],
        }
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, lineterminator="\n")
    copy = "COPY scoring.predictions ({}) FROM STDIN WITH (FORMAT csv)".format(
        ", ".join(frame.columns)
    )
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "DELETE FROM scoring.predictions WHERE source = 'batch' AND model_version = %s "
            "AND transaction_ts >= %s AND transaction_ts < %s",
            (str(rows["model_version"].iloc[0]), start, end),
        )
        with cur.copy(copy) as cp:
            cp.write(buffer.getvalue())
    conn.commit()  # conn.transaction() is only a savepoint if a transaction was already open


def summarise(out: pd.DataFrame, rows: pd.DataFrame, seconds: float, start, end) -> dict:
    day = rows["transaction_ts"].dt.date
    reviews_per_day = (out["decision"] == ACTION_NAMES[REVIEW]).groupby(day).sum()
    return {
        "period": [str(start), str(end)],
        "rows": int(len(out)),
        "seconds": round(seconds, 1),
        "rows_per_second": round(len(out) / seconds),
        "model_version": str(out["model_version"].iloc[0]),
        "policy_version": str(rows["policy_version"].iloc[0]),
        "decisions": out["decision"].value_counts().to_dict(),
        "risk_levels": out["risk_level"].value_counts().to_dict(),
        "reviews_per_day": {
            "mean": round(float(reviews_per_day.mean()), 2),
            "max": int(reviews_per_day.max()),
        },
        "fraud_probability_quantiles": {
            str(k): float(v)
            for k, v in out["fraud_probability"].quantile([0.5, 0.9, 0.99, 0.999]).items()
        },
    }


def run_batch(
    conn: psycopg.Connection,
    scorer: Scorer,
    start: date,
    end: date,
    output: Path,
    summary_path: Path | None = None,
    write_db: bool = True,
) -> dict:
    began = time.perf_counter()
    out, rows = score_period(conn, scorer, start, end)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, lineterminator="\n")
    if write_db:
        write_predictions(conn, rows, start, end)
    summary = summarise(out, rows, time.perf_counter() - began, start, end)
    summary["capacity_per_day"] = scorer.bundle.capacity
    summary["output"] = output.name
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
    log.info("scored %d transactions in %.1f s", len(out), summary["seconds"])
    return summary
