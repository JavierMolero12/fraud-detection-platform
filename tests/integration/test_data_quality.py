"""Data quality pipeline against the `small` dataset (~48k transactions, every defect type).

Because the simulator logged each injected defect, the rules can be tested for recall
(every defect caught) and for false alarms (no clean record flagged).
"""

from __future__ import annotations

import pytest

from fraudplatform.config import SQL_DIR
from fraudplatform.db import load_dataset
from fraudplatform.quality.report import EVALUATION_DIR, build_report
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration

# Rules allowed to fire on records without an injected defect.
EXPECTED_WITHOUT_DEFECT = {
    "CRD_PARENT_REJECTED",  # cards of a rejected customer
    "TXN_PARENT_REJECTED",  # transactions of a rejected customer or card
    "LBL_PARENT_REJECTED",  # labels of a rejected transaction
    "TXN_AMOUNT_OUTLIER",  # warning
    "TXN_SHARED_DEVICE",  # warning
}


@pytest.fixture(scope="module")
def run_id(conn, small, tmp_path_factory):
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    return run_data_quality(conn)


def _scalar(conn, sql, params=None):
    return conn.execute(sql, params).fetchone()[0]


def test_run_succeeds(conn, run_id):
    assert _scalar(conn, "SELECT status FROM dq.runs WHERE run_id = %s", (run_id,)) == "succeeded"


def test_every_injected_defect_is_detected(conn, run_id):
    sql = (EVALUATION_DIR / "dq_detection.sql").read_text(encoding="utf-8")
    rows = conn.execute(sql, {"run_id": run_id}).fetchall()
    assert rows, "the small dataset should contain injected defects"
    missed = [(table, defect, n - caught) for table, defect, n, caught, _ in rows if caught < n]
    assert not missed, f"defects not detected: {missed}"


def test_no_false_alarms_on_clean_records(conn, run_id):
    sql = (EVALUATION_DIR / "dq_false_alarms.sql").read_text(encoding="utf-8")
    rules = {rule for rule, _, _ in conn.execute(sql, {"run_id": run_id}).fetchall()}
    assert rules <= EXPECTED_WITHOUT_DEFECT, rules - EXPECTED_WITHOUT_DEFECT


def test_every_raw_row_is_loaded_or_excluded(conn, run_id):
    rows = conn.execute(
        "SELECT table_name, raw_rows, excluded_rows, loaded_rows FROM dq.table_stats "
        "WHERE run_id = %s",
        (run_id,),
    ).fetchall()
    assert len(rows) == 6
    for table, raw, excluded, loaded in rows:
        assert raw == excluded + loaded, table


def test_clean_transactions_are_loaded_unchanged(conn, run_id):
    """A transaction with no defect and no rejected parent must reach core untouched."""
    missing = _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM dq.stg_transactions s   -- typed copy of raw (never cast raw text in a filter:
                                     -- SQL does not guarantee the order predicates run in)
        WHERE NOT EXISTS (SELECT 1 FROM sim.dq_defects d
                          WHERE d.table_name = 'transactions' AND d.record_key = s.transaction_id)
          AND NOT EXISTS (SELECT 1 FROM dq.violations v
                          WHERE v.run_id = %(run)s AND v.rule_id = 'TXN_PARENT_REJECTED'
                            AND v.raw_id = s._raw_id)
          AND NOT EXISTS (SELECT 1 FROM core.transactions t
                          WHERE t.transaction_id = s.transaction_id
                            AND t.ts = s.ts
                            AND t.amount = s.amount
                            AND t.customer_id = s.customer_id
                            AND t.device_id IS NOT DISTINCT FROM s.device_id)
        """,
        {"run": run_id},
    )
    assert missing == 0


def test_sentinel_devices_become_null(conn, run_id):
    leftovers = _scalar(
        conn, "SELECT COUNT(*) FROM core.transactions WHERE dq.is_sentinel_device(device_id)"
    )
    loaded, repaired = conn.execute(
        """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE t.device_id IS NULL)
        FROM core.transactions t
        JOIN sim.dq_defects d ON d.record_key = t.transaction_id
        WHERE d.defect_type = 'sentinel_device'
        """
    ).fetchone()
    # Sentinel rows that never reached core must have been excluded by a cascade
    # (e.g. their customer was rejected), not by the sentinel itself.
    not_loaded_without_cascade = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM sim.dq_defects d
        JOIN dq.stg_transactions s ON s.transaction_id = d.record_key
        WHERE d.defect_type = 'sentinel_device'
          AND NOT EXISTS (SELECT 1 FROM core.transactions t WHERE t.transaction_id = d.record_key)
          AND NOT EXISTS (SELECT 1 FROM dq.violations v
                          WHERE v.run_id = %s AND v.raw_id = s._raw_id
                            AND v.rule_id = 'TXN_PARENT_REJECTED')
        """,
        (run_id,),
    )
    assert leftovers == 0
    assert loaded > 0 and repaired == loaded
    assert not_loaded_without_cascade == 0


def test_missing_customer_is_recovered_from_card_owner(conn, run_id):
    wrong = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM core.transactions t
        JOIN core.cards k USING (card_id)
        JOIN sim.dq_defects d ON d.record_key = t.transaction_id
        WHERE d.defect_type = 'missing_customer_id' AND t.customer_id <> k.customer_id
        """,
    )
    recovered = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM core.transactions t
        JOIN sim.dq_defects d ON d.record_key = t.transaction_id
        WHERE d.defect_type = 'missing_customer_id'
        """,
    )
    assert wrong == 0
    assert recovered > 0


def test_card_present_country_equals_merchant_country(conn, run_id):
    mismatches = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM core.transactions t JOIN core.merchants m USING (merchant_id)
        WHERE t.payment_method IN ('chip_pin', 'contactless')
          AND t.country_code <> m.country_code
        """,
    )
    assert mismatches == 0


def test_outliers_are_kept(conn, run_id):
    not_loaded = _scalar(
        conn,
        """
        SELECT COUNT(*) FROM dq.violations v
        WHERE v.run_id = %s AND v.rule_id = 'TXN_AMOUNT_OUTLIER'
          AND NOT EXISTS (SELECT 1 FROM core.transactions t
                          WHERE t.transaction_id = v.record_key)
        """,
        (run_id,),
    )
    assert not_loaded == 0


def test_rerun_produces_the_same_core(conn, run_id):
    before = _scalar(conn, "SELECT COUNT(*) FROM core.transactions")
    second = run_data_quality(conn)
    assert second != run_id
    assert _scalar(conn, "SELECT COUNT(*) FROM core.transactions") == before


def test_report_renders(conn, run_id):
    report = build_report(conn, run_id)
    assert "# Data Quality Report" in report
    assert "Duplicates:" in report
    assert "recall" in report


@pytest.mark.parametrize("path", sorted((SQL_DIR / "analysis").glob("*.sql")), ids=lambda p: p.name)
def test_analysis_queries_run(conn, run_id, path):
    rows = conn.execute(path.read_text(encoding="utf-8")).fetchall()
    assert rows
