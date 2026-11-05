-- ONLINE feature computation for ONE incoming transaction (used by the API).
--
-- Same definitions as the offline build (sql/features/10..40), written as lookups instead of
-- window functions (decision 017). History = core.transactions with ts STRICTLY before the
-- incoming ts; windows are [ts - w, ts); labels count only if reported before ts. A parity
-- test checks that both implementations give the same 50 values.
--
-- Parameters (psycopg named): ts, amount_eur, payment_method, txn_country, customer_id,
-- card_id, merchant_id, device_id (nullable), device_type / operating_system (used only for a
-- device never seen before), micro_amount_eur, night_start, night_end, impossible_speed_kmh,
-- merchant_rate_prior, merchant_rate_prior_strength.
-- Every parameter is cast explicitly to the type the offline SQL uses: values are bound
-- client-side (COPY), where an untyped literal like 50.0 would be NUMERIC, not float8, and
-- the result would differ from the offline value in the last digit.

WITH tx AS (
    SELECT %(ts)s::timestamptz                     AS ts,
           %(amount_eur)s::double precision        AS amount_eur,
           %(amount_eur)s::numeric                 AS amount_num,  -- offline: ln(numeric)
           %(payment_method)s::text                AS payment_method,
           %(txn_country)s::text                   AS txn_country,
           %(customer_id)s::text                   AS customer_id,
           %(card_id)s::text                       AS card_id,
           %(merchant_id)s::text                   AS merchant_id,
           %(device_id)s::text                     AS device_id,
           %(payment_method)s::text IN ('chip_pin', 'contactless') AS is_card_present
),
base AS (
    SELECT tx.*,
           cu.country_code AS home_country, cu.customer_segment, cu.date_of_birth,
           cu.account_created_at,
           k.card_type,
           m.merchant_category, m.risk_level AS merchant_risk_level,
           m.country_code AS merchant_country,
           c.utc_offset_hours, c.centroid_lat, c.centroid_lon,
           d.device_id IS NOT NULL AS device_known,
           -- a device never seen before: its first use is this transaction
           coalesce(d.first_seen_at, tx.ts) AS device_first_seen,
           coalesce(d.device_type, %(device_type)s::text, 'unknown') AS new_device_type,
           coalesce(d.operating_system, %(operating_system)s::text, 'unknown') AS new_device_os
    FROM tx
    JOIN core.customers cu ON cu.customer_id = tx.customer_id
    JOIN core.cards k      ON k.card_id = tx.card_id AND k.customer_id = tx.customer_id
    JOIN core.merchants m  ON m.merchant_id = tx.merchant_id
    JOIN core.countries c  ON c.country_code = tx.txn_country
    LEFT JOIN core.devices d ON d.device_id = tx.device_id
),
prev AS (  -- previous payment of the customer (ties on ts broken by id, as offline)
    SELECT h.ts AS prev_ts, h.country_code AS prev_country
    FROM core.transactions h, base b
    WHERE h.customer_id = b.customer_id AND h.ts < b.ts
    ORDER BY h.ts DESC, h.transaction_id DESC
    LIMIT 1
),
card_hist AS (
    SELECT h.ts, h.amount_eur::double precision AS amount_eur, h.amount_eur AS amount_num,
           h.merchant_id, h.country_code
    FROM core.transactions h, base b
    WHERE h.card_id = b.card_id AND h.ts < b.ts AND h.ts >= b.ts - INTERVAL '7 days'
),
cust_hist AS (
    SELECT h.ts, h.amount_eur::double precision AS amount_eur,
           ln(h.amount_eur)::double precision AS log_amount, h.device_id, h.country_code
    FROM core.transactions h, base b
    WHERE h.customer_id = b.customer_id AND h.ts < b.ts AND h.ts >= b.ts - INTERVAL '180 days'
),
dev_hist AS (
    SELECT h.ts, h.card_id, h.customer_id
    FROM core.transactions h, base b
    WHERE b.device_id IS NOT NULL
      AND h.device_id = b.device_id AND h.ts < b.ts AND h.ts >= b.ts - INTERVAL '30 days'
)
SELECT
    -- transaction ---------------------------------------------------------------------------
    b.amount_eur,
    ln(b.amount_num)::double precision                                      AS log_amount,
    b.amount_num < %(micro_amount_eur)s::numeric                            AS is_micro_amount,
    b.payment_method,
    b.is_card_present,
    b.merchant_category,
    b.merchant_risk_level,
    extract(hour FROM (b.ts AT TIME ZONE 'UTC')
                      + make_interval(hours => b.utc_offset_hours))::int    AS local_hour,
    extract(hour FROM (b.ts AT TIME ZONE 'UTC')
                      + make_interval(hours => b.utc_offset_hours))::int
        BETWEEN %(night_start)s::int AND %(night_end)s::int                 AS is_night,
    b.txn_country <> b.home_country                                         AS cross_border,
    b.merchant_country <> b.home_country                                    AS merchant_foreign,
    -- entity attributes at ts ---------------------------------------------------------------
    b.card_type,
    b.customer_segment,
    date_part('year', age(b.ts, b.date_of_birth))::int                      AS customer_age,
    extract(epoch FROM b.ts - b.account_created_at) / 86400.0               AS account_age_days,
    CASE WHEN b.device_id IS NOT NULL THEN b.new_device_type
         WHEN b.is_card_present THEN 'none' ELSE 'unknown' END              AS device_type,
    CASE WHEN b.device_id IS NOT NULL THEN b.new_device_os
         WHEN b.is_card_present THEN 'none' ELSE 'unknown' END              AS operating_system,
    CASE WHEN b.device_id IS NOT NULL
         THEN extract(epoch FROM b.ts - b.device_first_seen) / 86400.0 END  AS device_age_days,
    -- card velocity -------------------------------------------------------------------------
    (SELECT COUNT(*) FROM card_hist WHERE ts >= b.ts - INTERVAL '5 minutes') AS card_txn_count_5m,
    (SELECT COUNT(*) FROM card_hist WHERE ts >= b.ts - INTERVAL '1 hour')   AS card_txn_count_1h,
    (SELECT COUNT(*) FROM card_hist WHERE ts >= b.ts - INTERVAL '24 hours') AS card_txn_count_24h,
    (SELECT COUNT(*) FROM card_hist)                                        AS card_txn_count_7d,
    (SELECT coalesce(SUM(amount_eur), 0) FROM card_hist
      WHERE ts >= b.ts - INTERVAL '1 hour')                                 AS card_amount_sum_1h,
    (SELECT coalesce(SUM(amount_eur), 0) FROM card_hist
      WHERE ts >= b.ts - INTERVAL '24 hours')                               AS card_amount_sum_24h,
    (SELECT COUNT(*) FROM card_hist WHERE ts >= b.ts - INTERVAL '1 hour'
       AND amount_num < %(micro_amount_eur)s::numeric)                      AS card_micro_count_1h,
    (SELECT extract(epoch FROM b.ts - MAX(h.ts)) FROM core.transactions h
      WHERE h.card_id = b.card_id AND h.ts < b.ts)                          AS card_seconds_since_prev,
    (SELECT COUNT(DISTINCT merchant_id) FROM card_hist
      WHERE ts >= b.ts - INTERVAL '24 hours')                               AS card_unique_merchants_24h,
    (SELECT COUNT(DISTINCT country_code) FROM card_hist
      WHERE ts >= b.ts - INTERVAL '24 hours')                               AS card_unique_countries_24h,
    -- customer behaviour --------------------------------------------------------------------
    (SELECT COUNT(*) FROM cust_hist WHERE ts >= b.ts - INTERVAL '24 hours') AS cust_txn_count_24h,
    extract(epoch FROM b.ts - p.prev_ts)                                    AS cust_seconds_since_prev,
    (SELECT COUNT(*) FROM cust_hist WHERE ts >= b.ts - INTERVAL '90 days')  AS cust_txn_count_90d,
    b.amount_eur / NULLIF((SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY amount_eur)
                           FROM cust_hist WHERE ts >= b.ts - INTERVAL '90 days'), 0)
                                                                            AS amount_to_cust_median_90d,
    (ln(b.amount_num)::double precision - (SELECT AVG(log_amount) FROM cust_hist
                                           WHERE ts >= b.ts - INTERVAL '90 days'))
      / NULLIF((SELECT STDDEV_SAMP(log_amount) FROM cust_hist
                WHERE ts >= b.ts - INTERVAL '90 days'), 0)                  AS amount_zscore_cust_90d,
    NOT EXISTS (SELECT 1 FROM core.transactions h
                WHERE h.customer_id = b.customer_id AND h.merchant_id = b.merchant_id
                  AND h.ts < b.ts)                                          AS merchant_new_for_customer,
    NOT EXISTS (SELECT 1 FROM core.transactions h
                JOIN core.merchants hm ON hm.merchant_id = h.merchant_id
                WHERE h.customer_id = b.customer_id
                  AND hm.merchant_category = b.merchant_category
                  AND h.ts < b.ts)                                          AS category_new_for_customer,
    (SELECT COUNT(DISTINCT device_id) FROM cust_hist
      WHERE ts >= b.ts - INTERVAL '30 days')                                AS cust_unique_devices_30d,
    -- device --------------------------------------------------------------------------------
    CASE WHEN b.device_id IS NOT NULL THEN
        NOT EXISTS (SELECT 1 FROM core.transactions h
                    WHERE h.customer_id = b.customer_id AND h.device_id = b.device_id
                      AND h.ts < b.ts)
    END                                                                     AS device_new_for_customer,
    CASE WHEN b.device_id IS NOT NULL THEN
        (SELECT COUNT(*) FROM dev_hist WHERE ts >= b.ts - INTERVAL '1 hour')
    END                                                                     AS device_txn_count_1h,
    CASE WHEN b.device_id IS NOT NULL THEN
        (SELECT COUNT(DISTINCT card_id) FROM dev_hist WHERE ts >= b.ts - INTERVAL '24 hours')
    END                                                                     AS device_unique_cards_24h,
    CASE WHEN b.device_id IS NOT NULL THEN
        (SELECT COUNT(DISTINCT customer_id) FROM dev_hist)
    END                                                                     AS device_unique_customers_30d,
    -- geography (country precision) ---------------------------------------------------------
    CASE WHEN p.prev_country IS NOT NULL THEN p.prev_country <> b.txn_country END
                                                                            AS country_changed,
    NOT EXISTS (SELECT 1 FROM cust_hist WHERE country_code = b.txn_country) AS new_country_for_customer_180d,
    geo.distance_km                                                         AS distance_from_prev_km,
    geo.distance_km / NULLIF(extract(epoch FROM b.ts - p.prev_ts) / 3600.0, 0)
                                                                            AS implied_speed_kmh,
    geo.distance_km / NULLIF(extract(epoch FROM b.ts - p.prev_ts) / 3600.0, 0)
        > %(impossible_speed_kmh)s::double precision                        AS impossible_travel,
    -- labels known at ts --------------------------------------------------------------------
    (SELECT COUNT(*) FROM core.fraud_labels l JOIN core.transactions h USING (transaction_id)
      WHERE h.card_id = b.card_id AND l.is_fraud AND l.reported_at < b.ts)  AS card_prior_fraud_reports,
    CASE WHEN b.device_id IS NOT NULL THEN
        (SELECT COUNT(*) FROM core.fraud_labels l JOIN core.transactions h USING (transaction_id)
          WHERE h.device_id = b.device_id AND l.is_fraud AND l.reported_at < b.ts)
    END                                                                     AS device_prior_fraud_reports,
    mr.reports                                                              AS merchant_fraud_reports_90d,
    mc.txns                                                                 AS merchant_txn_count_90d,
    (mr.reports + %(merchant_rate_prior_strength)s::double precision
                  * %(merchant_rate_prior)s::double precision)
      / (mc.txns + %(merchant_rate_prior_strength)s::double precision)      AS merchant_fraud_rate_90d_smoothed
FROM base b
LEFT JOIN prev p ON true
LEFT JOIN core.countries pc ON pc.country_code = p.prev_country
CROSS JOIN LATERAL (
    SELECT features.haversine_km(pc.centroid_lat::double precision, pc.centroid_lon::double precision,
                                 b.centroid_lat::double precision, b.centroid_lon::double precision)
           AS distance_km
) geo
CROSS JOIN LATERAL (
    SELECT COUNT(*) AS reports
    FROM core.fraud_labels l JOIN core.transactions h USING (transaction_id)
    WHERE h.merchant_id = b.merchant_id AND l.is_fraud
      AND l.reported_at >= b.ts - INTERVAL '90 days' AND l.reported_at < b.ts
) mr
CROSS JOIN LATERAL (
    SELECT COUNT(*) AS txns FROM core.transactions h
    WHERE h.merchant_id = b.merchant_id AND h.ts < b.ts AND h.ts >= b.ts - INTERVAL '90 days'
) mc
