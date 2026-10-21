-- Core layer: the validated relational model (see docs/phase1_design.md, section 4).
-- Every constraint here encodes a business rule; rows that break one are quarantined in Phase 3
-- instead of being loaded.

-- ------------------------------------------------------------------------------------------
-- Reference data
-- ------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.currencies (
    currency_code CHAR(3) PRIMARY KEY CHECK (currency_code ~ '^[A-Z]{3}$')
);

CREATE TABLE IF NOT EXISTS core.countries (
    country_code     CHAR(2)      PRIMARY KEY CHECK (country_code ~ '^[A-Z]{2}$'),
    country_name     TEXT         NOT NULL,
    region           TEXT         NOT NULL,
    currency_code    CHAR(3)      NOT NULL REFERENCES core.currencies,
    -- Country centroid: geography is only known at country level. Distances derived from it
    -- are coarse by construction (same country = 0 km).
    centroid_lat     NUMERIC(7,4) NOT NULL CHECK (centroid_lat BETWEEN -90 AND 90),
    centroid_lon     NUMERIC(7,4) NOT NULL CHECK (centroid_lon BETWEEN -180 AND 180),
    utc_offset_hours SMALLINT     NOT NULL CHECK (utc_offset_hours BETWEEN -12 AND 14)
);

CREATE TABLE IF NOT EXISTS core.fx_rates (
    currency_code CHAR(3)        NOT NULL REFERENCES core.currencies,
    rate_date     DATE           NOT NULL,
    eur_per_unit  NUMERIC(14,6)  NOT NULL CHECK (eur_per_unit > 0),
    PRIMARY KEY (currency_code, rate_date)
);

-- ------------------------------------------------------------------------------------------
-- Entities. Attributes are immutable: a mutable column (e.g. a "status" updated after a fraud
-- case) would leak the future into historical features.
-- ------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.customers (
    customer_id        TEXT        PRIMARY KEY,
    date_of_birth      DATE        NOT NULL,   -- not "age": age depends on the transaction date
    country_code       CHAR(2)     NOT NULL REFERENCES core.countries,
    account_created_at TIMESTAMPTZ NOT NULL,
    customer_segment   TEXT        NOT NULL
        CHECK (customer_segment IN ('student', 'standard', 'premium', 'small_business')),
    CHECK (account_created_at >= date_of_birth + INTERVAL '18 years')
);

CREATE TABLE IF NOT EXISTS core.cards (
    card_id     TEXT        PRIMARY KEY,
    customer_id TEXT        NOT NULL REFERENCES core.customers,
    card_type   TEXT        NOT NULL CHECK (card_type IN ('debit', 'credit', 'prepaid', 'virtual')),
    issued_at   TIMESTAMPTZ NOT NULL,
    -- Target of the composite FK below: guarantees a card is used by its own customer.
    UNIQUE (card_id, customer_id)
);

CREATE TABLE IF NOT EXISTS core.merchants (
    merchant_id       TEXT        PRIMARY KEY,
    merchant_category TEXT        NOT NULL,
    channel           TEXT        NOT NULL CHECK (channel IN ('online', 'physical')),
    country_code      CHAR(2)     NOT NULL REFERENCES core.countries,
    -- External rating fixed at onboarding; never recomputed from our own fraud labels.
    risk_level        TEXT        NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    onboarded_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS core.devices (
    device_id        TEXT        PRIMARY KEY,
    device_type      TEXT        NOT NULL CHECK (device_type IN ('mobile', 'desktop', 'tablet')),
    operating_system TEXT        NOT NULL,
    country_code     CHAR(2)     NOT NULL REFERENCES core.countries,  -- country of FIRST use
    first_seen_at    TIMESTAMPTZ NOT NULL
);

-- ------------------------------------------------------------------------------------------
-- Facts
-- ------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS core.transactions (
    transaction_id TEXT          PRIMARY KEY,
    customer_id    TEXT          NOT NULL REFERENCES core.customers,
    card_id        TEXT          NOT NULL,
    merchant_id    TEXT          NOT NULL REFERENCES core.merchants,
    device_id      TEXT          REFERENCES core.devices,   -- NULL for card-present payments
    ts             TIMESTAMPTZ   NOT NULL,
    amount         NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    currency_code  CHAR(3)       NOT NULL REFERENCES core.currencies,
    amount_eur     NUMERIC(14,2) NOT NULL CHECK (amount_eur > 0),
    -- Where the payment originated: terminal country (card present) or IP country (online).
    country_code   CHAR(2)       NOT NULL REFERENCES core.countries,
    payment_method TEXT          NOT NULL
        CHECK (payment_method IN ('chip_pin', 'contactless', 'ecommerce', 'mobile_wallet')),
    FOREIGN KEY (card_id, customer_id) REFERENCES core.cards (card_id, customer_id),
    -- Card-present payments never have a customer device. Online and wallet payments always
    -- come from one, but the source sometimes sends a placeholder ("unknown"): the data
    -- quality stage keeps those transactions with device_id = NULL (device unknown) rather
    -- than dropping real payments.
    CONSTRAINT ck_txn_no_device_when_card_present
        CHECK (device_id IS NULL OR payment_method NOT IN ('chip_pin', 'contactless'))
);

-- One row per labelled transaction. A transaction without a row is "not known to be fraud",
-- which is NOT the same as "known to be legit" until the label window (120 days) has passed.
CREATE TABLE IF NOT EXISTS core.fraud_labels (
    transaction_id TEXT        PRIMARY KEY REFERENCES core.transactions,
    is_fraud       BOOLEAN     NOT NULL,
    reported_at    TIMESTAMPTZ NOT NULL,
    label_source   TEXT        NOT NULL
        CHECK (label_source IN ('chargeback', 'customer_report', 'investigation'))
);

-- ------------------------------------------------------------------------------------------
-- Indexes, designed for point-in-time features: "history of entity X before time t" is
-- always an index range scan on (entity, ts).
-- ------------------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_txn_card_ts     ON core.transactions (card_id, ts);
CREATE INDEX IF NOT EXISTS ix_txn_customer_ts ON core.transactions (customer_id, ts);
CREATE INDEX IF NOT EXISTS ix_txn_device_ts   ON core.transactions (device_id, ts)
    WHERE device_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_txn_merchant_ts ON core.transactions (merchant_id, ts);
-- BRIN: tiny index for date-range scans on an append-only, time-ordered table.
CREATE INDEX IF NOT EXISTS brin_txn_ts        ON core.transactions USING brin (ts);
CREATE INDEX IF NOT EXISTS ix_labels_reported ON core.fraud_labels (reported_at);
CREATE INDEX IF NOT EXISTS ix_cards_customer  ON core.cards (customer_id);
