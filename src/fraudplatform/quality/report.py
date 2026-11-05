"""Markdown Data Quality Report for one run of the pipeline."""

from __future__ import annotations

import psycopg

from fraudplatform.config import SQL_DIR
from fraudplatform.markdown import rows_table

EVALUATION_DIR = SQL_DIR / "evaluation"

# Headline categories, in the order of the report template agreed in the design.
CATEGORY_LABELS = {
    "duplicate": "Duplicates",
    "missing": "Missing values",
    "invalid": "Invalid values",
    "broken_relationship": "Broken relationships",
    "inconsistent": "Inconsistencies",
    "outlier": "Outliers (warnings, kept)",
}


def _query(conn: psycopg.Connection, sql: str, params: dict | None = None):
    cur = conn.execute(sql, params)
    return [d.name for d in cur.description], cur.fetchall()


def _evaluation(conn, name: str, run_id: int):
    sql = (EVALUATION_DIR / f"{name}.sql").read_text(encoding="utf-8")
    return _query(conn, sql, {"run_id": run_id})


def has_simulation_truth(conn: psycopg.Connection) -> bool:
    (n,) = conn.execute("SELECT COUNT(*) FROM sim.dq_defects").fetchone()
    return n > 0


def build_report(conn: psycopg.Connection, run_id: int) -> str:
    run = conn.execute(
        """
        SELECT r.run_id, r.dataset_hash, r.status, r.started_at,
               round(extract(epoch FROM r.finished_at - r.started_at)::numeric, 1),
               l.simulator_version, l.scale, l.seed
        FROM dq.runs r
        LEFT JOIN raw.load_log l USING (dataset_hash)
        WHERE r.run_id = %s
        ORDER BY l.load_id DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"run {run_id} not found")
    _, dataset_hash, status, started_at, seconds, version, scale, seed = run

    out = [
        "# Data Quality Report",
        "",
        f"- Run: `{run_id}` ({status}, {seconds}s, started {started_at:%Y-%m-%d %H:%M} UTC)",
        f"- Dataset: v{version} `{scale}` seed {seed}, hash `{dataset_hash[:16]}…`",
        "",
    ]

    # Headline numbers (transactions), in the format agreed in the design.
    (raw_rows,) = conn.execute("SELECT COUNT(*) FROM raw.transactions").fetchone()
    by_category = dict(
        conn.execute(
            """
            SELECT r.category, COUNT(DISTINCT v.raw_id)
            FROM dq.violations v JOIN dq.rules r USING (rule_id)
            WHERE v.run_id = %s AND v.table_name = 'transactions'
            GROUP BY r.category
            """,
            (run_id,),
        ).fetchall()
    )
    out += ["## Transactions at a glance", "", "```", f"Rows: {raw_rows:,}"]
    for category, label in CATEGORY_LABELS.items():
        out.append(f"{label}: {by_category.get(category, 0):,}")
    out += ["```", "", "Counts are raw rows with at least one violation of that category.", ""]

    cols, rows = _query(
        conn,
        """
        SELECT table_name, raw_rows, excluded_rows, repaired_rows, warned_rows, loaded_rows,
               round(100.0 * excluded_rows / NULLIF(raw_rows, 0), 3) AS excluded_pct
        FROM dq.table_stats WHERE run_id = %s
        ORDER BY array_position(ARRAY['customers', 'cards', 'merchants', 'devices',
                                      'transactions', 'fraud_labels'], table_name)
        """,
        (run_id,),
    )
    out += ["## Rows per table", "", rows_table(cols, rows), ""]

    cols, rows = _query(
        conn,
        """
        SELECT r.rule_id, r.table_name, r.category, r.action,
               COUNT(v.raw_id) AS rows_flagged, r.description
        FROM dq.rules r
        LEFT JOIN dq.violations v ON v.rule_id = r.rule_id AND v.run_id = %s
        GROUP BY r.rule_id
        ORDER BY r.table_name, rows_flagged DESC, r.rule_id
        """,
        (run_id,),
    )
    out += [
        "## Rules",
        "",
        "Actions: `reject` (not loaded), `deduplicate` (first copy loaded), `repair` (loaded "
        "after a deterministic fix), `warn` (loaded, only reported).",
        "",
        rows_table(cols, rows),
        "",
    ]

    # detail = "device <id> used by <n> customers"
    cols, rows = _query(
        conn,
        """
        SELECT split_part(detail, ' ', 2) AS device_id,
               split_part(detail, ' ', 5)::int AS distinct_customers,
               COUNT(*) AS transactions
        FROM dq.violations
        WHERE run_id = %s AND rule_id = 'TXN_SHARED_DEVICE'
        GROUP BY 1, 2
        ORDER BY 2 DESC
        """,
        (run_id,),
    )
    if rows:
        out += [
            "## Devices shared by many customers (warning)",
            "",
            "This warning is how placeholder device ids are discovered: no real device is "
            "used by thousands of customers.",
            "",
            rows_table(cols, rows),
            "",
        ]

    if has_simulation_truth(conn):
        out += _simulation_sections(conn, run_id)
    return "\n".join(out) + "\n"


def _simulation_sections(conn: psycopg.Connection, run_id: int) -> list[str]:
    out = [
        "## Evaluation against injected defects (simulation only)",
        "",
        "The simulator logged every defect it injected, so each rule can be measured. "
        "This is not possible with real data.",
        "",
    ]
    cols, rows = _evaluation(conn, "dq_detection", run_id)
    rows = [(*r, f"{100 * r[3] / r[2]:.1f}%" if r[2] else "n/a") for r in rows]
    out += [rows_table([*cols, "recall"], rows), ""]

    cols, rows = _evaluation(conn, "dq_false_alarms", run_id)
    out += [
        "### Violations on records without an injected defect",
        "",
        "Expected: warnings, and cascades (clean rows whose customer, card or transaction was "
        "rejected). Anything else would be a false alarm.",
        "",
        rows_table(cols, rows) if rows else "None.",
        "",
    ]

    cols, rows = _evaluation(conn, "dq_fraud_impact", run_id)
    out += [
        "### Fraud removed by cleaning",
        "",
        "If excluded rows were much more (or less) fraudulent than kept rows, cleaning would "
        "bias the training data.",
        "",
        rows_table(cols, rows),
        "",
    ]

    cols, rows = _evaluation(conn, "dq_outlier_fraud", run_id)
    out += [
        "### Are amount outliers errors or signal?",
        "",
        rows_table(cols, rows),
        "",
    ]
    return out
