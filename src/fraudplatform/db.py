"""PostgreSQL access: connection, schema creation and bulk loading of a simulated dataset."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import psycopg

from fraudplatform.config import SQL_DIR

log = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql://fraud:fraud@localhost:5433/fraud"  # local docker compose

# CSV file (relative to the dataset directory) -> target table. Raw tables receive the source
# extract as text; reference data is trusted master data and goes straight to core.
LOAD_PLAN: tuple[tuple[str, str], ...] = (
    ("reference/currencies.csv", "core.currencies"),
    ("reference/countries.csv", "core.countries"),
    ("reference/fx_rates.csv", "core.fx_rates"),
    ("raw/customers.csv", "raw.customers"),
    ("raw/cards.csv", "raw.cards"),
    ("raw/merchants.csv", "raw.merchants"),
    ("raw/devices.csv", "raw.devices"),
    ("raw/transactions.csv", "raw.transactions"),
    ("raw/fraud_labels.csv", "raw.fraud_labels"),
    ("sim/ground_truth.csv", "sim.ground_truth"),
    ("sim/dq_defects.csv", "sim.dq_defects"),
)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or database_url())


def schema_files() -> list[Path]:
    return sorted((SQL_DIR / "schema").glob("*.sql"))


def init_schema(conn: psycopg.Connection) -> list[str]:
    """Run every file in sql/schema in order. The files are idempotent (IF NOT EXISTS)."""
    applied = []
    for path in schema_files():
        conn.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
        log.info("applied %s", path.name)
    conn.commit()
    return applied


def _header(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.readline().rstrip("\n").split(",")


def load_dataset(conn: psycopg.Connection, dataset_dir: Path) -> dict[str, int]:
    """Replace the content of raw, sim and reference tables with a dataset written by the CLI.

    Everything runs in one transaction: either the whole dataset is loaded or nothing changes.
    The TRUNCATE cascades to core tables that reference the reference data: they are derived
    from raw by the data quality stage and must be rebuilt after a reload anyway.
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    with conn.transaction(), conn.cursor() as cur:
        targets = ", ".join(table for _, table in LOAD_PLAN)
        cur.execute(f"TRUNCATE {targets} RESTART IDENTITY CASCADE")
        for rel_path, table in LOAD_PLAN:
            path = dataset_dir / rel_path
            columns = ", ".join(_header(path))
            sql = f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)"
            with cur.copy(sql) as copy, open(path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    copy.write(chunk)
            counts[table] = cur.rowcount
            expected = manifest["files"][rel_path]["rows"]
            if cur.rowcount != expected:
                raise RuntimeError(f"{table}: loaded {cur.rowcount} rows, manifest says {expected}")
            log.info("loaded %-22s %10s rows", table, f"{cur.rowcount:,}")
        cur.execute(
            """
            INSERT INTO raw.load_log (dataset_hash, simulator_version, scale, seed, row_counts)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                manifest["dataset_hash"],
                manifest["simulator_version"],
                manifest["scale"],
                manifest["seed"],
                json.dumps(counts),
            ),
        )
    return counts
