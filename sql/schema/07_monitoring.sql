-- Monitoring results (Phase 12): one row per monitoring date (as_of) and subject.
-- A run on date D only used payments with ts < D and labels with reported_at < D.

CREATE TABLE IF NOT EXISTS monitoring.feature_drift (
    as_of     DATE             NOT NULL,
    feature   TEXT             NOT NULL,
    payments  INTEGER          NOT NULL,   -- in the drift window
    psi       DOUBLE PRECISION NOT NULL,
    ks        DOUBLE PRECISION,            -- numeric features only
    PRIMARY KEY (as_of, feature)
);

-- prediction drift (score PSI, decision rates) and delay-adjusted calibration
CREATE TABLE IF NOT EXISTS monitoring.metrics (
    as_of     DATE             NOT NULL,
    metric    TEXT             NOT NULL,
    value     DOUBLE PRECISION,
    reference DOUBLE PRECISION,
    PRIMARY KEY (as_of, metric)
);

-- weekly payment cohorts as known on as_of: labels in so far, expected so far, recall
CREATE TABLE IF NOT EXISTS monitoring.label_cohorts (
    as_of            DATE             NOT NULL,
    cohort_start     DATE             NOT NULL,
    payments         INTEGER          NOT NULL,
    known_frauds     INTEGER          NOT NULL,
    expected_by_now  DOUBLE PRECISION NOT NULL,
    caught_known     INTEGER          NOT NULL,
    mature           BOOLEAN          NOT NULL,
    PRIMARY KEY (as_of, cohort_start)
);

CREATE TABLE IF NOT EXISTS monitoring.alerts (
    as_of     DATE             NOT NULL,
    rule      TEXT             NOT NULL,
    subject   TEXT             NOT NULL,
    severity  TEXT             NOT NULL CHECK (severity IN ('warning', 'alert', 'action')),
    value     DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    detail    TEXT,
    PRIMARY KEY (as_of, rule, subject)
);
