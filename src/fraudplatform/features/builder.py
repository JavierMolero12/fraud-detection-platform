"""Build the feature tables from core (sql/features/*.sql) in one transaction."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psycopg

from fraudplatform.config import SQL_DIR, FeatureConfig

log = logging.getLogger(__name__)

FEATURE_SQL_DIR = SQL_DIR / "features"


def feature_files() -> list[Path]:
    return sorted(FEATURE_SQL_DIR.glob("*.sql"))


def build_features(conn: psycopg.Connection, cfg: FeatureConfig) -> int:
    """Rebuild features.transaction_features (and the leaky variant). Returns the build id."""
    started = time.perf_counter()
    with conn.transaction():
        for name, value in cfg.sql_settings().items():
            conn.execute("SELECT set_config(%s, %s, true)", (name, value))
        for path in feature_files():
            step = time.perf_counter()
            conn.execute(path.read_text(encoding="utf-8"))
            log.info("%-18s %6.1fs", path.name, time.perf_counter() - step)
        (rows,) = conn.execute("SELECT COUNT(*) FROM features.transaction_features").fetchone()
        (build_id,) = conn.execute(
            """
            INSERT INTO features.builds
                (feature_set_version, dataset_hash, dq_run_id, rows, seconds)
            VALUES (
                %s,
                (SELECT dataset_hash FROM raw.load_log ORDER BY load_id DESC LIMIT 1),
                (SELECT max(run_id) FROM dq.runs WHERE status = 'succeeded'),
                %s, %s
            )
            RETURNING build_id
            """,
            (cfg.feature_set_version, rows, round(time.perf_counter() - started, 1)),
        ).fetchone()
    log.info("feature build %s: %s rows in %.1fs", build_id, rows, time.perf_counter() - started)
    return build_id
