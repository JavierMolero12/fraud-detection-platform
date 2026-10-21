-- Step 2: history features expressible as window functions (fast: one sort per partition).
--
-- Every frame is
--     RANGE BETWEEN <w> PRECEDING AND CURRENT ROW EXCLUDE GROUP
-- which means [ts - w, ts]: the current row AND its peers (rows with the same ts) excluded,
-- i.e. only transactions strictly before this one. Same-second transactions are not "before":
-- their real order is unknown, and in production the current transaction is not yet stored.
--
-- Not possible here (PostgreSQL does not allow COUNT(DISTINCT) or percentiles in windows):
-- distinct counts and medians -> step 3 (LATERAL lookups).

DROP TABLE IF EXISTS features.f_windows;
CREATE UNLOGGED TABLE features.f_windows AS
SELECT transaction_id,

       -- card velocity -------------------------------------------------------------------------
       COUNT(*) OVER (PARTITION BY card_id ORDER BY ts
                      RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND CURRENT ROW EXCLUDE GROUP)
           AS card_txn_count_5m,
       COUNT(*) OVER card_1h                                          AS card_txn_count_1h,
       COUNT(*) OVER card_24h                                         AS card_txn_count_24h,
       COUNT(*) OVER (PARTITION BY card_id ORDER BY ts
                      RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW EXCLUDE GROUP)
           AS card_txn_count_7d,
       coalesce(SUM(amount_eur) OVER card_1h, 0)                      AS card_amount_sum_1h,
       coalesce(SUM(amount_eur) OVER card_24h, 0)                     AS card_amount_sum_24h,
       COUNT(*) FILTER (WHERE is_micro_amount) OVER card_1h           AS card_micro_count_1h,
       extract(epoch FROM ts - MAX(ts) OVER (
           PARTITION BY card_id ORDER BY ts
           RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP))
           AS card_seconds_since_prev,

       -- customer behaviour --------------------------------------------------------------------
       COUNT(*) OVER (PARTITION BY customer_id ORDER BY ts
                      RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW EXCLUDE GROUP)
           AS cust_txn_count_24h,
       extract(epoch FROM ts - MAX(ts) OVER (
           PARTITION BY customer_id ORDER BY ts
           RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP))
           AS cust_seconds_since_prev,
       COUNT(*) OVER cust_90d                                         AS cust_txn_count_90d,
       AVG(log_amount) OVER cust_90d                                  AS cust_log_amount_avg_90d,
       STDDEV_SAMP(log_amount) OVER cust_90d                          AS cust_log_amount_std_90d,
       COUNT(*) OVER (PARTITION BY customer_id, merchant_id ORDER BY ts
                      RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) = 0
           AS merchant_new_for_customer,
       COUNT(*) OVER (PARTITION BY customer_id, merchant_category ORDER BY ts
                      RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) = 0
           AS category_new_for_customer,

       -- geography -----------------------------------------------------------------------------
       COUNT(*) OVER (PARTITION BY customer_id, txn_country ORDER BY ts
                      RANGE BETWEEN INTERVAL '180 days' PRECEDING AND CURRENT ROW EXCLUDE GROUP) = 0
           AS new_country_for_customer_180d,

       -- device (NULL when the payment has no known device) ------------------------------------
       CASE WHEN device_id IS NOT NULL THEN
           COUNT(*) OVER (PARTITION BY customer_id, device_id ORDER BY ts
                          RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) = 0
       END AS device_new_for_customer,
       CASE WHEN device_id IS NOT NULL THEN
           COUNT(*) OVER (PARTITION BY device_id ORDER BY ts
                          RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW EXCLUDE GROUP)
       END AS device_txn_count_1h,

       -- merchant activity (denominator of the merchant fraud rate) ----------------------------
       COUNT(*) OVER (PARTITION BY merchant_id ORDER BY ts
                      RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW EXCLUDE GROUP)
           AS merchant_txn_count_90d

FROM features.f_base
WINDOW card_1h  AS (PARTITION BY card_id ORDER BY ts
                    RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW EXCLUDE GROUP),
       card_24h AS (PARTITION BY card_id ORDER BY ts
                    RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW EXCLUDE GROUP),
       cust_90d AS (PARTITION BY customer_id ORDER BY ts
                    RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW EXCLUDE GROUP);

ALTER TABLE features.f_windows ADD PRIMARY KEY (transaction_id);
