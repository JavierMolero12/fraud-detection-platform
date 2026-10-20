-- Raw layer: one table per source file, every business column as TEXT.
--
-- Why no types or constraints here? Because the source is dirty. If raw.transactions had
-- `amount NUMERIC CHECK (amount > 0)`, a negative or malformed amount would make the whole load
-- fail and we could never see, count or quarantine the bad rows. Validation happens when rows
-- move to core (Phase 3).
--
-- _raw_id gives every physical row an identity, so exact duplicates can still be told apart.

CREATE TABLE IF NOT EXISTS raw.customers (
    _raw_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id           TEXT,
    date_of_birth         TEXT,
    country               TEXT,
    account_creation_date TEXT,
    customer_segment      TEXT,
    _loaded_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.cards (
    _raw_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    card_id            TEXT,
    customer_id        TEXT,
    card_type          TEXT,
    card_creation_date TEXT,
    _loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.merchants (
    _raw_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    merchant_id       TEXT,
    merchant_category TEXT,
    channel           TEXT,
    country           TEXT,
    risk_level        TEXT,
    onboarding_date   TEXT,
    _loaded_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.devices (
    _raw_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    device_id        TEXT,
    device_type      TEXT,
    operating_system TEXT,
    country          TEXT,
    first_seen_at    TEXT,
    _loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    _raw_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id TEXT,
    customer_id    TEXT,
    card_id        TEXT,
    merchant_id    TEXT,
    device_id      TEXT,
    "timestamp"    TEXT,
    amount         TEXT,
    currency       TEXT,
    country        TEXT,
    payment_method TEXT,
    _loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.fraud_labels (
    _raw_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    transaction_id TEXT,
    is_fraud       TEXT,
    reported_at    TEXT,
    label_source   TEXT,
    _loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Lineage: which dataset version is currently loaded.
CREATE TABLE IF NOT EXISTS raw.load_log (
    load_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_hash      TEXT        NOT NULL,
    simulator_version TEXT        NOT NULL,
    scale             TEXT        NOT NULL,
    seed              BIGINT      NOT NULL,
    row_counts        JSONB       NOT NULL,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
