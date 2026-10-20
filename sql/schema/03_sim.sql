-- Simulator-only tables. They exist because the data is synthetic, and they let us MEASURE
-- things that are unknowable in real life:
--   ground_truth : true label and fraud scenario of every clean transaction (error analysis)
--   dq_defects   : every defect injected into the raw extract (recall of data quality rules)
--
-- Feature and model code must never read this schema (enforced by a test in Phase 5).

CREATE TABLE IF NOT EXISTS sim.ground_truth (
    transaction_id TEXT    PRIMARY KEY,
    is_fraud_true  BOOLEAN NOT NULL,
    scenario       TEXT    NOT NULL,
    incident_id    BIGINT  NOT NULL   -- -1 for legit transactions
);

CREATE TABLE IF NOT EXISTS sim.dq_defects (
    defect_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    table_name  TEXT NOT NULL,
    record_key  TEXT NOT NULL,
    defect_type TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS ix_dq_defects_key ON sim.dq_defects (table_name, record_key);
