-- Entity rules: customers -> cards (depend on customers), merchants, devices.
-- Each block flags violations, then builds dq.clean_<entity>: typed, deduplicated rows that
-- passed every reject rule. Transactions are validated against these clean tables.

-- ==========================================================================================
-- customers
-- ==========================================================================================
SELECT dq.flag_duplicates('customers', 'customer_id', 'CUS_DUP_EXACT', 'CUS_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CUS_MISSING_ID', 'customers', _raw_id, NULL, 'customer_id is null'
FROM raw.customers
WHERE customer_id IS NULL;

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CUS_MISSING_COUNTRY', 'customers', _raw_id, customer_id, 'country is null'
FROM raw.customers
WHERE country IS NULL;

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CUS_INVALID_VALUE', 'customers', _raw_id, customer_id, problems
FROM (
    SELECT _raw_id, customer_id,
           concat_ws('; ',
               CASE WHEN country IS NOT NULL
                     AND country NOT IN (SELECT country_code FROM core.countries)
                    THEN 'unknown country ' || country END,
               CASE WHEN customer_segment IS NULL
                     OR customer_segment NOT IN ('student', 'standard', 'premium', 'small_business')
                    THEN 'unknown segment ' || coalesce(customer_segment, 'null') END,
               CASE WHEN NOT pg_input_is_valid(coalesce(date_of_birth, ''), 'date')
                    THEN 'bad date_of_birth' END,
               CASE WHEN NOT pg_input_is_valid(coalesce(account_creation_date, ''), 'timestamptz')
                    THEN 'bad account_creation_date' END
           ) AS problems
    FROM raw.customers
) checked
WHERE problems <> '';

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CUS_UNDERAGE_ACCOUNT', 'customers', _raw_id, customer_id,
       'opened at age ' || date_part('year', age(account_creation_date::timestamptz,
                                                 date_of_birth::date))
FROM raw.customers
WHERE pg_input_is_valid(coalesce(date_of_birth, ''), 'date')
  AND pg_input_is_valid(coalesce(account_creation_date, ''), 'timestamptz')
  AND account_creation_date::timestamptz < date_of_birth::date + INTERVAL '18 years';

DROP TABLE IF EXISTS dq.clean_customers;
CREATE UNLOGGED TABLE dq.clean_customers AS
SELECT DISTINCT ON (customer_id)
       _raw_id,
       customer_id,
       date_of_birth::date                    AS date_of_birth,
       country                                AS country_code,
       account_creation_date::timestamptz     AS account_created_at,
       customer_segment
FROM raw.customers r
WHERE NOT dq.is_excluded('customers', r._raw_id)
ORDER BY customer_id, _raw_id;
ALTER TABLE dq.clean_customers ADD PRIMARY KEY (customer_id);

-- ==========================================================================================
-- cards
-- ==========================================================================================
SELECT dq.flag_duplicates('cards', 'card_id', 'CRD_DUP_EXACT', 'CRD_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CRD_INVALID_VALUE', 'cards', _raw_id, card_id, problems
FROM (
    SELECT _raw_id, card_id,
           concat_ws('; ',
               CASE WHEN card_id IS NULL THEN 'card_id is null' END,
               CASE WHEN card_type IS NULL
                     OR card_type NOT IN ('debit', 'credit', 'prepaid', 'virtual')
                    THEN 'unknown card_type ' || coalesce(card_type, 'null') END,
               CASE WHEN NOT pg_input_is_valid(coalesce(card_creation_date, ''), 'timestamptz')
                    THEN 'bad card_creation_date' END
           ) AS problems
    FROM raw.cards
) checked
WHERE problems <> '';

-- Unknown customer (orphan) vs. known customer that failed its own rules: two different
-- problems with two different owners (source system bug vs. our own rejection).
INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(),
       CASE WHEN EXISTS (SELECT 1 FROM raw.customers c WHERE c.customer_id = k.customer_id)
            THEN 'CRD_PARENT_REJECTED' ELSE 'CRD_ORPHAN_CUSTOMER' END,
       'cards', k._raw_id, k.card_id, 'customer ' || coalesce(k.customer_id, 'null')
FROM raw.cards k
WHERE NOT EXISTS (SELECT 1 FROM dq.clean_customers c WHERE c.customer_id = k.customer_id);

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'CRD_ISSUED_BEFORE_ACCOUNT', 'cards', k._raw_id, k.card_id,
       'issued ' || k.card_creation_date || ', account opened ' || c.account_created_at
FROM raw.cards k
JOIN dq.clean_customers c USING (customer_id)
WHERE pg_input_is_valid(coalesce(k.card_creation_date, ''), 'timestamptz')
  AND k.card_creation_date::timestamptz < c.account_created_at;

DROP TABLE IF EXISTS dq.clean_cards;
CREATE UNLOGGED TABLE dq.clean_cards AS
SELECT DISTINCT ON (card_id)
       _raw_id, card_id, customer_id, card_type,
       card_creation_date::timestamptz AS issued_at
FROM raw.cards k
WHERE NOT dq.is_excluded('cards', k._raw_id)
ORDER BY card_id, _raw_id;
ALTER TABLE dq.clean_cards ADD PRIMARY KEY (card_id);

-- ==========================================================================================
-- merchants
-- ==========================================================================================
SELECT dq.flag_duplicates('merchants', 'merchant_id', 'MER_DUP_EXACT', 'MER_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'MER_INVALID_VALUE', 'merchants', _raw_id, merchant_id, problems
FROM (
    SELECT _raw_id, merchant_id,
           concat_ws('; ',
               CASE WHEN merchant_id IS NULL THEN 'merchant_id is null' END,
               CASE WHEN merchant_category IS NULL OR merchant_category NOT IN (
                        'grocery', 'restaurants', 'fuel', 'transport', 'fashion', 'electronics',
                        'digital_goods', 'travel', 'health', 'utilities', 'entertainment',
                        'gift_cards', 'luxury', 'home')
                    THEN 'unknown category ' || coalesce(merchant_category, 'null') END,
               CASE WHEN channel IS NULL OR channel NOT IN ('online', 'physical')
                    THEN 'unknown channel' END,
               CASE WHEN risk_level IS NULL OR risk_level NOT IN ('low', 'medium', 'high')
                    THEN 'unknown risk_level' END,
               CASE WHEN country IS NULL
                     OR country NOT IN (SELECT country_code FROM core.countries)
                    THEN 'unknown country ' || coalesce(country, 'null') END,
               CASE WHEN NOT pg_input_is_valid(coalesce(onboarding_date, ''), 'timestamptz')
                    THEN 'bad onboarding_date' END
           ) AS problems
    FROM raw.merchants
) checked
WHERE problems <> '';

DROP TABLE IF EXISTS dq.clean_merchants;
CREATE UNLOGGED TABLE dq.clean_merchants AS
SELECT DISTINCT ON (merchant_id)
       _raw_id, merchant_id, merchant_category, channel, country AS country_code, risk_level,
       onboarding_date::timestamptz AS onboarded_at
FROM raw.merchants m
WHERE NOT dq.is_excluded('merchants', m._raw_id)
ORDER BY merchant_id, _raw_id;
ALTER TABLE dq.clean_merchants ADD PRIMARY KEY (merchant_id);

-- ==========================================================================================
-- devices
-- ==========================================================================================
SELECT dq.flag_duplicates('devices', 'device_id', 'DEV_DUP_EXACT', 'DEV_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'DEV_INVALID_VALUE', 'devices', _raw_id, device_id, problems
FROM (
    SELECT _raw_id, device_id,
           concat_ws('; ',
               CASE WHEN device_id IS NULL THEN 'device_id is null' END,
               CASE WHEN device_type IS NULL OR device_type NOT IN ('mobile', 'desktop', 'tablet')
                    THEN 'unknown device_type ' || coalesce(device_type, 'null') END,
               CASE WHEN country IS NULL
                     OR country NOT IN (SELECT country_code FROM core.countries)
                    THEN 'unknown country ' || coalesce(country, 'null') END,
               CASE WHEN NOT pg_input_is_valid(coalesce(first_seen_at, ''), 'timestamptz')
                    THEN 'bad first_seen_at' END
           ) AS problems
    FROM raw.devices
) checked
WHERE problems <> '';

DROP TABLE IF EXISTS dq.clean_devices;
CREATE UNLOGGED TABLE dq.clean_devices AS
SELECT DISTINCT ON (device_id)
       _raw_id, device_id, device_type, operating_system, country AS country_code,
       first_seen_at::timestamptz AS first_seen_at
FROM raw.devices d
WHERE NOT dq.is_excluded('devices', d._raw_id)
ORDER BY device_id, _raw_id;
ALTER TABLE dq.clean_devices ADD PRIMARY KEY (device_id);
