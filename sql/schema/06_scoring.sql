-- Scoring log: every decision made by the API or the batch job.
--
-- No foreign key to core.transactions on purpose: the API scores a payment at authorisation
-- time, BEFORE the ingestion pipeline has written it to core.
-- `features` keeps the model inputs of each decision, which monitoring (Phase 12) needs to
-- measure input drift without recomputing anything.

CREATE TABLE IF NOT EXISTS scoring.predictions (
    prediction_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id    TEXT             NOT NULL,
    source            TEXT             NOT NULL CHECK (source IN ('api', 'batch')),
    model_version     TEXT             NOT NULL,
    policy_version    TEXT             NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_level        TEXT             NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    decision          TEXT             NOT NULL CHECK (decision IN ('approve', 'review', 'block')),
    amount_eur        NUMERIC(14,2)    NOT NULL,
    transaction_ts    TIMESTAMPTZ      NOT NULL,
    scored_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    latency_ms        DOUBLE PRECISION,  -- API: request to decision, without the log write
    features          JSONB
);
CREATE INDEX IF NOT EXISTS ix_predictions_source_ts ON scoring.predictions (source, transaction_ts);
CREATE INDEX IF NOT EXISTS ix_predictions_txn ON scoring.predictions (transaction_id);
-- live review capacity: "how many reviews have we already sent today?"
CREATE INDEX IF NOT EXISTS ix_predictions_reviews
    ON scoring.predictions (((transaction_ts AT TIME ZONE 'UTC')::date)) WHERE decision = 'review';
-- idempotency: a retried API call (same transaction_id) gets the decision already made instead
-- of being scored, and counted against the review capacity, twice
CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_api_txn
    ON scoring.predictions (transaction_id) WHERE source = 'api';
