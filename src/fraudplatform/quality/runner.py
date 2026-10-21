"""Run the SQL data quality pipeline (sql/data_quality/*.sql) as one atomic transaction."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psycopg

from fraudplatform.config import SQL_DIR

log = logging.getLogger(__name__)

PIPELINE_DIR = SQL_DIR / "data_quality"


def pipeline_files() -> list[Path]:
    return sorted(PIPELINE_DIR.glob("*.sql"))


def run_data_quality(conn: psycopg.Connection) -> int:
    """Validate raw, quarantine bad rows, rebuild core. Returns the run id.

    The run row is committed first so that a failed run stays visible in dq.runs; the rules
    and the core load then run in a single transaction, so core is either fully replaced or
    untouched.
    """
    row = conn.execute(
        "SELECT dataset_hash FROM raw.load_log ORDER BY load_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("raw is empty: run `fraud db load <dataset>` first")
    (run_id,) = conn.execute(
        "INSERT INTO dq.runs (dataset_hash, status) VALUES (%s, 'running') RETURNING run_id",
        (row[0],),
    ).fetchone()
    conn.commit()

    try:
        with conn.transaction():
            # Local to this transaction: every SQL file reads it through dq.current_run().
            conn.execute("SELECT set_config('dq.run_id', %s, true)", (str(run_id),))
            for path in pipeline_files():
                started = time.perf_counter()
                conn.execute(path.read_text(encoding="utf-8"))
                log.info("%-22s %6.1fs", path.name, time.perf_counter() - started)
            conn.execute(
                "UPDATE dq.runs SET status = 'succeeded', finished_at = clock_timestamp() "
                "WHERE run_id = %s",
                (run_id,),
            )
    except Exception:
        conn.rollback()
        conn.execute(
            "UPDATE dq.runs SET status = 'failed', finished_at = clock_timestamp() "
            "WHERE run_id = %s",
            (run_id,),
        )
        conn.commit()
        raise
    return run_id
