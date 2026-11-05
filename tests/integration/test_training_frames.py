"""Training frames come in a fixed row order, whatever plan the database picks.

Model results depend on row order (gradient boosting bins a 200k-row subsample; shuffled K-fold
assigns rows by position). Phase 7 loaded its frames through a parallel join that returned a
new order on every load, so its numbers moved in the third decimal from run to run.
"""

from __future__ import annotations

from datetime import date

import pytest

from fraudplatform.config import load_feature_config
from fraudplatform.db import load_dataset
from fraudplatform.eda.data import load_features
from fraudplatform.features.builder import build_features
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration

ALL = (date(2000, 1, 1), date(2100, 1, 1))


@pytest.fixture(scope="module")
def shuffled(conn, small, tmp_path_factory):
    """Features built, then half the rows physically moved to the end of their tables."""
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    build_features(conn, load_feature_config())
    # An UPDATE writes a new row version at the end of the heap: same values, new position.
    for table in ("transaction_features", "transaction_features_leaky"):
        conn.execute(
            f"UPDATE features.{table} SET transaction_id = transaction_id "
            "WHERE hashtext(transaction_id) % 2 = 0"
        )
    # Sequential scans read in physical order: an unordered query now returns rows unsorted.
    conn.execute("SET enable_indexscan = off")
    conn.execute("SET enable_bitmapscan = off")
    yield
    conn.execute("RESET enable_indexscan")
    conn.execute("RESET enable_bitmapscan")


def test_the_heap_is_really_out_of_order(conn, shuffled):
    rows = conn.execute("SELECT ts, transaction_id FROM features.transaction_features").fetchall()
    assert rows != sorted(rows)


@pytest.mark.parametrize("include_leaky", [False, True])
def test_rows_are_ordered_by_time_then_id(conn, shuffled, include_leaky):
    df = load_features(conn, *ALL, include_leaky=include_leaky)
    assert len(df) > 1000
    expected = df.sort_values(["ts", "transaction_id"], kind="stable")
    assert df["transaction_id"].tolist() == expected["transaction_id"].tolist()
