-- One row per transaction in [{start}, {end}) with the attributes known AT TRANSACTION TIME
-- (ages are computed at ts, local time uses the transaction country) plus the final label.
--
-- No historical aggregates here: those are features (Phase 5) and need point-in-time logic.
-- The label is the eventual, mature label: fine for a retrospective EDA of the train period.

SELECT t.transaction_id,
       t.ts,
       (t.ts AT TIME ZONE 'UTC') + make_interval(hours => c.utc_offset_hours)  AS local_ts,
       t.amount_eur,
       t.payment_method,
       t.country_code                                    AS txn_country,
       m.merchant_category,
       m.channel                                         AS merchant_channel,
       m.risk_level                                      AS merchant_risk_level,
       m.country_code                                    AS merchant_country,
       cu.customer_segment,
       cu.country_code                                   AS home_country,
       date_part('year', age(t.ts, cu.date_of_birth))::int        AS customer_age,
       extract(epoch FROM t.ts - cu.account_created_at) / 86400   AS account_age_days,
       extract(epoch FROM t.ts - k.issued_at) / 86400             AS card_age_days,
       k.card_type,
       coalesce(d.device_type, CASE WHEN t.payment_method IN ('chip_pin', 'contactless')
                                    THEN 'none (card present)' ELSE 'unknown' END) AS device_type,
       d.operating_system,
       coalesce(l.is_fraud, false)                       AS is_fraud,
       l.reported_at,
       l.label_source
FROM core.transactions t
JOIN core.merchants m  USING (merchant_id)
JOIN core.customers cu USING (customer_id)
JOIN core.cards k      USING (card_id)
JOIN core.countries c  ON c.country_code = t.country_code
LEFT JOIN core.devices d       USING (device_id)
LEFT JOIN core.fraud_labels l  USING (transaction_id)
WHERE t.ts >= {start} AND t.ts < {end}
