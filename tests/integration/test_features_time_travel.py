"""Time-travel test: deleting the future must not change any feature of the past.

Build features, then delete every transaction after a cutoff and every label reported after
it, rebuild, and compare the features of transactions before the cutoff. A point-in-time
feature cannot change; a leaky one does. The leaky table is checked too, to prove the test
would catch leakage (a test that cannot fail proves nothing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudplatform.config import load_feature_config
from fraudplatform.db import load_dataset
from fraudplatform.features.builder import build_features
from fraudplatform.features.catalog import FEATURE_NAMES, LEAKY_FEATURES
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration

CUTOFF = "2025-04-01"  # middle of the `small` period (2025-01-01 .. 2025-06-30)


def _frame(conn, table: str) -> pd.DataFrame:
    cur = conn.execute(
        f"SELECT f.* FROM features.{table} f JOIN core.transactions t USING (transaction_id) "
        "WHERE t.ts < %s ORDER BY transaction_id",
        (CUTOFF,),
    )
    columns = [d.name for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=columns).set_index("transaction_id")


@pytest.fixture(scope="module")
def before_and_after(conn, small, tmp_path_factory):
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    cfg = load_feature_config()
    build_features(conn, cfg)
    before = _frame(conn, "transaction_features"), _frame(conn, "transaction_features_leaky")

    with conn.transaction():
        conn.execute(
            "DELETE FROM core.fraud_labels l USING core.transactions t "
            "WHERE l.transaction_id = t.transaction_id AND (t.ts >= %s OR l.reported_at >= %s)",
            (CUTOFF, CUTOFF),
        )
        conn.execute("DELETE FROM core.transactions WHERE ts >= %s", (CUTOFF,))
    build_features(conn, cfg)
    after = _frame(conn, "transaction_features"), _frame(conn, "transaction_features_leaky")
    return before, after


def _changed_columns(a: pd.DataFrame, b: pd.DataFrame, columns) -> list[str]:
    changed = []
    for col in columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            same = np.allclose(
                x.astype(float).to_numpy(), y.astype(float).to_numpy(), equal_nan=True, rtol=1e-9
            )
        else:
            same = x.astype(object).equals(y.astype(object))
        if not same:
            changed.append(col)
    return changed


def test_past_features_ignore_the_future(before_and_after):
    (clean_before, _), (clean_after, _) = before_and_after
    assert len(clean_before) == len(clean_after) > 10_000
    assert _changed_columns(clean_before, clean_after, FEATURE_NAMES) == []


def test_leaky_features_are_caught(before_and_after):
    (_, leaky_before), (_, leaky_after) = before_and_after
    changed = _changed_columns(leaky_before, leaky_after, LEAKY_FEATURES)
    # Every deliberately leaky feature must be detected by this test.
    assert sorted(changed) == sorted(LEAKY_FEATURES)
