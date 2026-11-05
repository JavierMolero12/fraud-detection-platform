-- Question: how many times was the card used in the hour BEFORE each transaction?
-- Techniques: RANGE window with an INTERVAL offset, EXCLUDE GROUP, and a deliberately
-- leaky variant for comparison. Preview of the Phase 5 features (one month only, for speed).
--
-- Point-in-time correct:  RANGE BETWEEN '1 hour' PRECEDING AND CURRENT ROW EXCLUDE GROUP
--   * EXCLUDE GROUP removes the current row AND its peers (other rows with the same ts):
--     a transaction at the same second is not "before" this one.
-- Leaky:  RANGE BETWEEN '1 hour' PRECEDING AND '1 hour' FOLLOWING
--   * also counts the next hour, which does not exist yet at decision time.

WITH velocity AS (
    SELECT t.transaction_id,
           COUNT(*) OVER (
               PARTITION BY card_id ORDER BY ts
               RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW EXCLUDE GROUP
           ) AS txns_prev_1h,
           COUNT(*) OVER (
               PARTITION BY card_id ORDER BY ts
               RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 hour' FOLLOWING
               EXCLUDE GROUP
           ) AS txns_leaky_pm_1h
    FROM core.transactions t
    WHERE ts >= (SELECT date_trunc('month', MIN(ts)) + INTERVAL '1 month' FROM core.transactions)
      AND ts <  (SELECT date_trunc('month', MIN(ts)) + INTERVAL '2 month' FROM core.transactions)
)
SELECT least(txns_prev_1h, 5)                          AS txns_prev_1h_capped,
       COUNT(*)                                        AS transactions,
       round(100.0 * AVG((l.is_fraud IS TRUE)::int), 3) AS labelled_fraud_pct,
       round(AVG(txns_leaky_pm_1h - txns_prev_1h), 3)  AS avg_extra_rows_seen_by_leaky_version
FROM velocity v
LEFT JOIN core.fraud_labels l USING (transaction_id)
GROUP BY 1
ORDER BY 1;
