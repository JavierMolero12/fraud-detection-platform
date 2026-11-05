from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from fraudplatform.db import database_url, init_schema


def _with_db(url: str, name: str) -> str:
    return urlunsplit(urlsplit(url)._replace(path=f"/{name}"))


@pytest.fixture(scope="module")
def db_url() -> str:
    """URL of the test database, for code that opens its own connections (the API pool)."""
    base = database_url()
    return _with_db(base, urlsplit(base).path.lstrip("/") + "_test")


@pytest.fixture(scope="module")
def conn():
    """A fresh, empty test database per test module (never the development database)."""
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
