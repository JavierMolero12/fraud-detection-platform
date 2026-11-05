-- Step 4: assemble the feature table and derive ratios, distances and flags.
-- Column order follows src/fraudplatform/features/catalog.py (checked by a test).

DROP TABLE IF EXISTS features.transaction_features;
CREATE TABLE features.transaction_features AS
SELECT b.transaction_id,
       b.ts,
       b.customer_id,
       b.card_id,
       b.merchant_id,
       b.device_id,
       -- transaction
       b.amount_eur,
       b.log_amount,
       b.is_micro_amount,
       b.payment_method,
       b.is_card_present,
       b.merchant_category,
       b.merchant_risk_level,
       b.local_hour,
       b.local_hour BETWEEN current_setting('features.night_start')::int
                        AND current_setting('features.night_end')::int        AS is_night,
       b.cross_border,
       b.merchant_foreign,
       -- entity attributes at ts
       b.card_type,
       b.customer_segment,
       b.customer_age,
       b.account_age_days,
       b.device_type,
       b.operating_system,
       b.device_age_days,
       -- card velocity
       w.card_txn_count_5m,
       w.card_txn_count_1h,
       w.card_txn_count_24h,
       w.card_txn_count_7d,
       w.card_amount_sum_1h,
       w.card_amount_sum_24h,
       w.card_micro_count_1h,
       w.card_seconds_since_prev,
       l.card_unique_merchants_24h,
       l.card_unique_countries_24h,
       -- customer behaviour
       w.cust_txn_count_24h,
       w.cust_seconds_since_prev,
       w.cust_txn_count_90d,
       b.amount_eur / NULLIF(l.cust_amount_median_90d, 0)                     AS amount_to_cust_median_90d,
       (b.log_amount - w.cust_log_amount_avg_90d)
           / NULLIF(w.cust_log_amount_std_90d, 0)                             AS amount_zscore_cust_90d,
       w.merchant_new_for_customer,
       w.category_new_for_customer,
       l.cust_unique_devices_30d,
       -- device
       w.device_new_for_customer,
       w.device_txn_count_1h,
       l.device_unique_cards_24h,
       l.device_unique_customers_30d,
       -- geography (country precision)
       CASE WHEN l.prev_country IS NOT NULL THEN l.prev_country <> b.txn_country END
                                                                              AS country_changed,
       w.new_country_for_customer_180d,
       geo.distance_km                                                        AS distance_from_prev_km,
       geo.distance_km / NULLIF(w.cust_seconds_since_prev / 3600.0, 0)        AS implied_speed_kmh,
       geo.distance_km / NULLIF(w.cust_seconds_since_prev / 3600.0, 0)
           > current_setting('features.impossible_speed_kmh')::double precision
                                                                              AS impossible_travel,
       -- labels known at ts
       l.card_prior_fraud_reports,
       l.device_prior_fraud_reports,
       l.merchant_fraud_reports_90d,
       w.merchant_txn_count_90d,
       (l.merchant_fraud_reports_90d
            + current_setting('features.merchant_rate_prior_strength')::double precision
              * current_setting('features.merchant_rate_prior')::double precision)
         / (w.merchant_txn_count_90d
            + current_setting('features.merchant_rate_prior_strength')::double precision)
                                                                              AS merchant_fraud_rate_90d_smoothed,
       current_setting('features.version')                                    AS feature_set_version
FROM features.f_base b
JOIN features.f_windows w USING (transaction_id)
JOIN features.f_lookups l USING (transaction_id)
LEFT JOIN core.countries cur  ON cur.country_code = b.txn_country
LEFT JOIN core.countries prv  ON prv.country_code = l.prev_country
CROSS JOIN LATERAL (
    SELECT features.haversine_km(prv.centroid_lat::double precision, prv.centroid_lon::double precision,
                                 cur.centroid_lat::double precision, cur.centroid_lon::double precision)
           AS distance_km
) geo
ORDER BY b.ts, b.transaction_id;

ALTER TABLE features.transaction_features ADD PRIMARY KEY (transaction_id);
CREATE INDEX ON features.transaction_features (ts);
ANALYZE features.transaction_features;

DROP TABLE features.f_base, features.f_windows, features.f_lookups;
