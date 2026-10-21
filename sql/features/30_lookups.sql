-- Step 3: history features that window functions cannot express, as LATERAL lookups.
-- Each lookup is an index range scan on (entity, ts) over [ts - w, ts): strictly before.
--
-- Label features read features.fraud_reports filtered on reported_at < ts: a fraud that
-- happened earlier but was reported later was NOT known at decision time (leakage risk L3).

DROP TABLE IF EXISTS features.fraud_reports;
CREATE UNLOGGED TABLE features.fraud_reports AS
SELECT t.card_id, t.device_id, t.merchant_id, t.ts AS txn_ts, l.reported_at
FROM core.fraud_labels l
JOIN core.transactions t USING (transaction_id)
WHERE l.is_fraud;
CREATE INDEX ON features.fraud_reports (card_id, reported_at);
CREATE INDEX ON features.fraud_reports (device_id, reported_at) WHERE device_id IS NOT NULL;
CREATE INDEX ON features.fraud_reports (merchant_id, reported_at);
ANALYZE features.fraud_reports;

DROP TABLE IF EXISTS features.f_lookups;
CREATE UNLOGGED TABLE features.f_lookups AS
SELECT b.transaction_id,
       card24.unique_merchants     AS card_unique_merchants_24h,
       card24.unique_countries     AS card_unique_countries_24h,
       cust30.unique_devices       AS cust_unique_devices_30d,
       cust90.median_amount        AS cust_amount_median_90d,
       prev.country_code           AS prev_country,
       dev24.unique_cards          AS device_unique_cards_24h,
       dev30.unique_customers      AS device_unique_customers_30d,
       (SELECT COUNT(*) FROM features.fraud_reports r
         WHERE r.card_id = b.card_id AND r.reported_at < b.ts)          AS card_prior_fraud_reports,
       CASE WHEN b.device_id IS NOT NULL THEN
           (SELECT COUNT(*) FROM features.fraud_reports r
             WHERE r.device_id = b.device_id AND r.reported_at < b.ts)
       END                                                              AS device_prior_fraud_reports,
       (SELECT COUNT(*) FROM features.fraud_reports r
         WHERE r.merchant_id = b.merchant_id
           AND r.reported_at >= b.ts - INTERVAL '90 days'
           AND r.reported_at < b.ts)                                    AS merchant_fraud_reports_90d
FROM features.f_base b
CROSS JOIN LATERAL (
    SELECT COUNT(DISTINCT h.merchant_id)  AS unique_merchants,
           COUNT(DISTINCT h.country_code) AS unique_countries
    FROM core.transactions h
    WHERE h.card_id = b.card_id
      AND h.ts >= b.ts - INTERVAL '24 hours' AND h.ts < b.ts
) card24
CROSS JOIN LATERAL (
    SELECT COUNT(DISTINCT h.device_id) AS unique_devices
    FROM core.transactions h
    WHERE h.customer_id = b.customer_id
      AND h.ts >= b.ts - INTERVAL '30 days' AND h.ts < b.ts
) cust30
CROSS JOIN LATERAL (
    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY h.amount_eur) AS median_amount
    FROM core.transactions h
    WHERE h.customer_id = b.customer_id
      AND h.ts >= b.ts - INTERVAL '90 days' AND h.ts < b.ts
) cust90
LEFT JOIN LATERAL (
    -- previous payment of the customer; ties on ts broken by id to stay deterministic
    SELECT h.country_code
    FROM core.transactions h
    WHERE h.customer_id = b.customer_id AND h.ts < b.ts
    ORDER BY h.ts DESC, h.transaction_id DESC
    LIMIT 1
) prev ON true
LEFT JOIN LATERAL (
    SELECT COUNT(DISTINCT h.card_id) AS unique_cards
    FROM core.transactions h
    WHERE h.device_id = b.device_id
      AND h.ts >= b.ts - INTERVAL '24 hours' AND h.ts < b.ts
) dev24 ON b.device_id IS NOT NULL
LEFT JOIN LATERAL (
    SELECT COUNT(DISTINCT h.customer_id) AS unique_customers
    FROM core.transactions h
    WHERE h.device_id = b.device_id
      AND h.ts >= b.ts - INTERVAL '30 days' AND h.ts < b.ts
) dev30 ON b.device_id IS NOT NULL;

ALTER TABLE features.f_lookups ADD PRIMARY KEY (transaction_id);
