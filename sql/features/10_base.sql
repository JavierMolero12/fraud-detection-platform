-- Step 1: one row per transaction with the attributes known at transaction time.
-- No history yet. Ages are computed at ts; the hour is local to the payment country.

DROP TABLE IF EXISTS features.f_base;
CREATE UNLOGGED TABLE features.f_base AS
SELECT t.transaction_id,
       t.ts,
       t.customer_id,
       t.card_id,
       t.merchant_id,
       t.device_id,
       t.amount_eur::double precision                                     AS amount_eur,
       ln(t.amount_eur)::double precision                                 AS log_amount,
       t.amount_eur < current_setting('features.micro_amount_eur')::numeric AS is_micro_amount,
       t.payment_method,
       t.payment_method IN ('chip_pin', 'contactless')                    AS is_card_present,
       m.merchant_category,
       m.risk_level                                                       AS merchant_risk_level,
       extract(hour FROM (t.ts AT TIME ZONE 'UTC')
                         + make_interval(hours => c.utc_offset_hours))::int AS local_hour,
       t.country_code                                                     AS txn_country,
       t.country_code <> cu.country_code                                  AS cross_border,
       m.country_code <> cu.country_code                                  AS merchant_foreign,
       k.card_type,
       cu.customer_segment,
       date_part('year', age(t.ts, cu.date_of_birth))::int                AS customer_age,
       extract(epoch FROM t.ts - cu.account_created_at) / 86400.0         AS account_age_days,
       CASE WHEN t.device_id IS NOT NULL THEN d.device_type
            WHEN t.payment_method IN ('chip_pin', 'contactless') THEN 'none'
            ELSE 'unknown' END                                            AS device_type,
       CASE WHEN t.device_id IS NOT NULL THEN d.operating_system
            WHEN t.payment_method IN ('chip_pin', 'contactless') THEN 'none'
            ELSE 'unknown' END                                            AS operating_system,
       -- first_seen_at is the device's first use, which is always <= ts: no future information.
       extract(epoch FROM t.ts - d.first_seen_at) / 86400.0               AS device_age_days
FROM core.transactions t
JOIN core.merchants m  USING (merchant_id)
JOIN core.customers cu USING (customer_id)
JOIN core.cards k      USING (card_id)
JOIN core.countries c  ON c.country_code = t.country_code
LEFT JOIN core.devices d USING (device_id);

ALTER TABLE features.f_base ADD PRIMARY KEY (transaction_id);
ANALYZE features.f_base;
