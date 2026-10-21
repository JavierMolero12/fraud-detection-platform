-- SIMULATION ONLY: are amount outliers errors or signal?
-- Compares the true fraud rate of transactions flagged as amount outliers with the rest.
-- Parameter %(run_id)s.

WITH outliers AS (
    SELECT DISTINCT record_key AS transaction_id
    FROM dq.violations
    WHERE run_id = %(run_id)s AND rule_id = 'TXN_AMOUNT_OUTLIER'
)
SELECT (o.transaction_id IS NOT NULL)          AS is_outlier,
       COUNT(*)                                AS transactions,
       round(100.0 * AVG(g.is_fraud_true::int), 3) AS true_fraud_pct
FROM core.transactions t
JOIN sim.ground_truth g USING (transaction_id)
LEFT JOIN outliers o USING (transaction_id)
GROUP BY 1
ORDER BY 1;
