"""Schema and loader tests against a real PostgreSQL.

They run in a separate database (``<db>_test``) so they never touch the development data.
Run with:  docker compose up -d db  &&  pytest -m integration
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from fraudplatform.db import database_url, init_schema, load_dataset
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration


def _with_db(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture(scope="module")
def conn():
    base = database_url()
    test_db = urlsplit(base).path.lstrip("/") + "_test"
    try:
        admin = psycopg.connect(_with_db(base, "postgres"), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {test_db} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {test_db}")
    with psycopg.connect(_with_db(base, test_db)) as connection:
        init_schema(connection)
        yield connection


@pytest.fixture(scope="module")
def loaded(conn, tiny, tmp_path_factory):
    path = write_dataset(tiny, tmp_path_factory.mktemp("dataset"))
    return load_dataset(conn, path)


def test_schema_is_idempotent(conn):
    assert init_schema(conn)  # second run must not fail


def test_load_matches_dataset(conn, tiny, loaded):
    assert loaded["raw.transactions"] == len(tiny.raw["transactions"])
    assert loaded["sim.ground_truth"] == len(tiny.ground_truth)
    (n_logs,) = conn.execute("SELECT count(*) FROM raw.load_log").fetchone()
    assert n_logs == 1


def test_reload_replaces_instead_of_appending(conn, tiny, tmp_path):
    load_dataset(conn, write_dataset(tiny, tmp_path))
    (n,) = conn.execute("SELECT count(*) FROM raw.transactions").fetchone()
    assert n == len(tiny.raw["transactions"])


# ------------------------------------------------------------------------------------------
# The core schema rejects rows that break business rules
# ------------------------------------------------------------------------------------------
@pytest.fixture
def core_rows(conn, loaded):
    """A minimal valid customer, two cards (one of another customer), merchant and device."""
    conn.execute(
        """
        INSERT INTO core.customers VALUES
          ('C1', '1990-01-01', 'ES', '2020-01-01', 'standard'),
          ('C2', '1985-01-01', 'ES', '2020-01-01', 'premium');
        INSERT INTO core.cards VALUES ('K1', 'C1', 'debit', '2020-01-02'),
                                      ('K2', 'C2', 'debit', '2020-01-02');
        INSERT INTO core.merchants VALUES ('M1', 'grocery', 'physical', 'ES', 'low', '2019-01-01');
        INSERT INTO core.devices VALUES ('D1', 'mobile', 'ios', 'ES', '2020-01-03');
        """
    )
    yield
    conn.rollback()


VALID = ("T1", "C1", "K1", "M1", "D1", "2025-01-01T10:00:00Z", 10, "EUR", 10, "ES", "ecommerce")
INSERT = "INSERT INTO core.transactions VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"


def test_valid_transaction_is_accepted(conn, core_rows):
    conn.execute(INSERT, VALID)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({6: -5}, psycopg.errors.CheckViolation),  # amount must be > 0
        ({2: "K2"}, psycopg.errors.ForeignKeyViolation),  # card belongs to another customer
        ({3: "M404"}, psycopg.errors.ForeignKeyViolation),  # unknown merchant
        ({4: None}, psycopg.errors.CheckViolation),  # e-commerce without device
        ({10: "chip_pin"}, psycopg.errors.CheckViolation),  # card present with a device
        ({7: "XXX"}, psycopg.errors.ForeignKeyViolation),  # unknown currency
    ],
)
def test_invalid_transaction_is_rejected(conn, core_rows, change, error):
    row = list(VALID)
    for i, value in change.items():
        row[i] = value
    with pytest.raises(error):
        conn.execute(INSERT, row)
