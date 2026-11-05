-- Transaction rules.
-- 1. Parse the raw text once into a typed staging table, without failing on bad values
--    (pg_input_is_valid, PostgreSQL 16+, tests a cast without raising an error).
-- 2. Flag violations.
-- 3. Build dq.clean_transactions applying the repairs.

DROP TABLE IF EXISTS dq.stg_transactions;
CREATE UNLOGGED TABLE dq.stg_transactions AS
SELECT _raw_id,
       transaction_id,
       customer_id,
       card_id,
       merchant_id,
       device_id,
       "timestamp" AS ts_raw,
       CASE WHEN pg_input_is_valid("timestamp", 'timestamptz')
            THEN "timestamp"::timestamptz END AS ts,
       amount AS amount_raw,
       CASE WHEN pg_input_is_valid(amount, 'numeric') THEN amount::numeric END AS amount,
       currency AS currency_raw,
       upper(trim(currency)) AS currency_code,
       country AS country_code,
       payment_method
FROM raw.transactions;
ALTER TABLE dq.stg_transactions ADD PRIMARY KEY (_raw_id);
CREATE INDEX ON dq.stg_transactions (card_id);
CREATE INDEX ON dq.stg_transactions (device_id);
ANALYZE dq.stg_transactions;

-- ------------------------------------------------------------------------------------------
-- duplicates and missing ids
-- ------------------------------------------------------------------------------------------
SELECT dq.flag_duplicates('transactions', 'transaction_id', 'TXN_DUP_EXACT', 'TXN_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_MISSING_ID', 'transactions', _raw_id, NULL, 'transaction_id is null'
FROM dq.stg_transactions
WHERE transaction_id IS NULL;

-- ------------------------------------------------------------------------------------------
-- invalid values
-- ------------------------------------------------------------------------------------------
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_INVALID_AMOUNT', 'transactions', _raw_id, transaction_id,
       'amount = ' || coalesce(amount_raw, 'null')
FROM dq.stg_transactions
WHERE amount IS NULL OR amount <= 0;

-- Same currency, wrong format ("eur", " EUR"): deterministic fix, so repair instead of reject.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_CURRENCY_FORMAT', 'transactions', s._raw_id, s.transaction_id,
       quote_literal(s.currency_raw) || ' -> ' || s.currency_code
FROM dq.stg_transactions s
JOIN core.currencies c USING (currency_code)
WHERE s.currency_raw <> s.currency_code;

-- "EURO" also probably means EUR, but guessing is not a deterministic rule: reject.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_INVALID_CURRENCY', 'transactions', _raw_id, transaction_id,
       'currency = ' || coalesce(quote_literal(currency_raw), 'null')
FROM dq.stg_transactions s
WHERE NOT EXISTS (SELECT 1 FROM core.currencies c WHERE c.currency_code = s.currency_code);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_INVALID_TIMESTAMP', 'transactions', _raw_id, transaction_id,
       'timestamp = ' || coalesce(quote_literal(ts_raw), 'null')
FROM dq.stg_transactions
WHERE ts IS NULL;

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_FUTURE_TIMESTAMP', 'transactions', _raw_id, transaction_id,
       'timestamp ' || ts_raw || ' is after the extraction'
FROM dq.stg_transactions
WHERE ts > (SELECT started_at FROM dq.runs WHERE run_id = dq.current_run());

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_INVALID_VALUE', 'transactions', _raw_id, transaction_id, problems
FROM (
    SELECT _raw_id, transaction_id,
           concat_ws('; ',
               CASE WHEN payment_method IS NULL OR payment_method NOT IN
                        ('chip_pin', 'contactless', 'ecommerce', 'mobile_wallet')
                    THEN 'unknown payment_method ' || coalesce(payment_method, 'null') END,
               CASE WHEN country_code IS NULL
                     OR country_code NOT IN (SELECT country_code FROM core.countries)
                    THEN 'unknown country ' || coalesce(country_code, 'null') END
           ) AS problems
    FROM dq.stg_transactions
) checked
WHERE problems <> '';

-- ------------------------------------------------------------------------------------------
-- relationships
-- ------------------------------------------------------------------------------------------
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_UNKNOWN_CUSTOMER', 'transactions', _raw_id, transaction_id,
       'customer ' || customer_id
FROM dq.stg_transactions s
WHERE customer_id IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM raw.customers c WHERE c.customer_id = s.customer_id);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_UNKNOWN_CARD', 'transactions', _raw_id, transaction_id,
       'card ' || coalesce(card_id, 'null')
FROM dq.stg_transactions s
WHERE NOT EXISTS (SELECT 1 FROM raw.cards k WHERE k.card_id = s.card_id);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_UNKNOWN_MERCHANT', 'transactions', _raw_id, transaction_id,
       'merchant ' || coalesce(merchant_id, 'null')
FROM dq.stg_transactions s
WHERE NOT EXISTS (SELECT 1 FROM raw.merchants m WHERE m.merchant_id = s.merchant_id);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_UNKNOWN_DEVICE', 'transactions', _raw_id, transaction_id,
       'device ' || device_id
FROM dq.stg_transactions s
WHERE device_id IS NOT NULL
  AND NOT dq.is_sentinel_device(device_id)
  AND NOT EXISTS (SELECT 1 FROM raw.devices d WHERE d.device_id = s.device_id);

-- A card has exactly one owner. If the transaction names another customer we cannot tell
-- which of the two ids is wrong, so the row is rejected.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_CARD_CUSTOMER_MISMATCH', 'transactions', s._raw_id,
       s.transaction_id, 'card ' || s.card_id || ' belongs to ' || k.customer_id
                         || ', not ' || s.customer_id
FROM dq.stg_transactions s
JOIN dq.clean_cards k USING (card_id)
WHERE s.customer_id IS NOT NULL
  AND s.customer_id <> k.customer_id;

-- ...but if the customer id is simply missing, the card owner is a deterministic repair.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_MISSING_CUSTOMER', 'transactions', s._raw_id, s.transaction_id,
       'customer_id null -> ' || k.customer_id || ' (owner of card ' || s.card_id || ')'
FROM dq.stg_transactions s
JOIN dq.clean_cards k USING (card_id)
WHERE s.customer_id IS NULL;

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_PARENT_REJECTED', 'transactions', _raw_id, transaction_id, problems
FROM (
    SELECT s._raw_id, s.transaction_id,
           concat_ws('; ',
               CASE WHEN EXISTS (SELECT 1 FROM raw.customers c WHERE c.customer_id = s.customer_id)
                     AND NOT EXISTS (SELECT 1 FROM dq.clean_customers c
                                     WHERE c.customer_id = s.customer_id)
                    THEN 'customer ' || s.customer_id || ' rejected' END,
               CASE WHEN EXISTS (SELECT 1 FROM raw.cards k WHERE k.card_id = s.card_id)
                     AND NOT EXISTS (SELECT 1 FROM dq.clean_cards k WHERE k.card_id = s.card_id)
                    THEN 'card ' || s.card_id || ' rejected' END,
               CASE WHEN EXISTS (SELECT 1 FROM raw.merchants m WHERE m.merchant_id = s.merchant_id)
                     AND NOT EXISTS (SELECT 1 FROM dq.clean_merchants m
                                     WHERE m.merchant_id = s.merchant_id)
                    THEN 'merchant ' || s.merchant_id || ' rejected' END,
               CASE WHEN EXISTS (SELECT 1 FROM raw.devices d WHERE d.device_id = s.device_id)
                     AND NOT EXISTS (SELECT 1 FROM dq.clean_devices d
                                     WHERE d.device_id = s.device_id)
                    THEN 'device ' || s.device_id || ' rejected' END
           ) AS problems
    FROM dq.stg_transactions s
) checked
WHERE problems <> '';

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_NO_FX_RATE', 'transactions', s._raw_id, s.transaction_id,
       s.currency_code || ' on ' || (s.ts AT TIME ZONE 'UTC')::date
FROM dq.stg_transactions s
JOIN core.currencies c USING (currency_code)
WHERE s.ts IS NOT NULL
  AND s.ts <= (SELECT started_at FROM dq.runs WHERE run_id = dq.current_run())
  AND NOT EXISTS (
      SELECT 1 FROM core.fx_rates fx
      WHERE fx.currency_code = s.currency_code
        AND fx.rate_date = (s.ts AT TIME ZONE 'UTC')::date
  )
  -- a timestamp before the card was issued is reported by its own rule below
  AND NOT EXISTS (
      SELECT 1 FROM dq.clean_cards k WHERE k.card_id = s.card_id AND s.ts < k.issued_at
  );

-- ------------------------------------------------------------------------------------------
-- consistency between fields
-- ------------------------------------------------------------------------------------------
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_BEFORE_CARD_ISSUE', 'transactions', s._raw_id, s.transaction_id,
       'transaction ' || s.ts_raw || ', card issued ' || k.issued_at
FROM dq.stg_transactions s
JOIN dq.clean_cards k USING (card_id)
WHERE s.ts < k.issued_at;

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_SENTINEL_DEVICE', 'transactions', _raw_id, transaction_id,
       quote_literal(device_id) || ' -> NULL'
FROM dq.stg_transactions
WHERE dq.is_sentinel_device(device_id);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_DEVICE_ON_CARD_PRESENT', 'transactions', _raw_id, transaction_id,
       payment_method || ' with device ' || device_id
FROM dq.stg_transactions
WHERE payment_method IN ('chip_pin', 'contactless')
  AND device_id IS NOT NULL
  AND NOT dq.is_sentinel_device(device_id);

-- For card-present payments the terminal is at the merchant, so the merchant country is the
-- ground truth for the transaction country.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_COUNTRY_MISMATCH', 'transactions', s._raw_id, s.transaction_id,
       s.country_code || ' -> ' || m.country_code || ' (merchant country)'
FROM dq.stg_transactions s
JOIN dq.clean_merchants m USING (merchant_id)
WHERE s.payment_method IN ('chip_pin', 'contactless')
  AND s.country_code <> m.country_code;

-- ------------------------------------------------------------------------------------------
-- clean transactions (repairs applied)
-- ------------------------------------------------------------------------------------------
DROP TABLE IF EXISTS dq.clean_transactions;
CREATE UNLOGGED TABLE dq.clean_transactions AS
SELECT s._raw_id,
       s.transaction_id,
       coalesce(s.customer_id, k.customer_id)                     AS customer_id,
       s.card_id,
       s.merchant_id,
       CASE WHEN dq.is_sentinel_device(s.device_id) THEN NULL ELSE s.device_id END AS device_id,
       s.ts,
       s.amount,
       s.currency_code,
       greatest(round(s.amount * fx.eur_per_unit, 2), 0.01)      AS amount_eur,
       CASE WHEN s.payment_method IN ('chip_pin', 'contactless')
            THEN m.country_code ELSE s.country_code END           AS country_code,
       s.payment_method
FROM dq.stg_transactions s
JOIN dq.clean_cards k USING (card_id)
JOIN dq.clean_merchants m USING (merchant_id)
JOIN core.fx_rates fx
  ON fx.currency_code = s.currency_code
 AND fx.rate_date = (s.ts AT TIME ZONE 'UTC')::date
WHERE NOT dq.is_excluded('transactions', s._raw_id);
ALTER TABLE dq.clean_transactions ADD PRIMARY KEY (transaction_id);

-- ------------------------------------------------------------------------------------------
-- warnings: reported, never removed
-- ------------------------------------------------------------------------------------------
-- Outliers are not errors. In fraud the extreme amounts are exactly where the signal is, so
-- dropping them would remove fraud from the training data.
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_AMOUNT_OUTLIER', 'transactions', t._raw_id, t.transaction_id,
       t.amount_eur || ' EUR > p99.9 of ' || m.merchant_category || ' (' || round(p.p999, 2) || ')'
FROM dq.clean_transactions t
JOIN dq.clean_merchants m USING (merchant_id)
JOIN (
    SELECT m.merchant_category,
           percentile_cont(0.999) WITHIN GROUP (ORDER BY t.amount_eur)::numeric AS p999
    FROM dq.clean_transactions t
    JOIN dq.clean_merchants m USING (merchant_id)
    GROUP BY m.merchant_category
) p USING (merchant_category)
WHERE t.amount_eur > p.p999;

-- Computed on the raw ids on purpose: this is how the placeholder ids were discovered.
-- NB: uses the whole extract, which is fine for a data quality report but would be temporal
-- leakage as a model feature (features must only count devices seen BEFORE each transaction).
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'TXN_SHARED_DEVICE', 'transactions', s._raw_id, s.transaction_id,
       'device ' || s.device_id || ' used by ' || d.customers || ' customers'
FROM dq.stg_transactions s
JOIN (
    SELECT device_id, COUNT(DISTINCT customer_id) AS customers
    FROM dq.stg_transactions
    WHERE device_id IS NOT NULL
    GROUP BY device_id
    HAVING COUNT(DISTINCT customer_id) > 50
) d USING (device_id);
