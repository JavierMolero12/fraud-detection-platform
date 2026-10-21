-- Question: how much fraud do we KNOW about, depending on when we look?
-- Techniques: CTE, conditional aggregation with FILTER, interval arithmetic, subquery.
--
-- For each transaction month: fraud rate as known 7, 30 and 60 days after the month ended,
-- versus with all labels. If we trained on recent months with the labels known "today",
-- those months would look cleaner than they really are (temporal leakage risk L4).

WITH monthly AS (
    SELECT date_trunc('month', t.ts)                    AS month,
           t.transaction_id,
           l.is_fraud,
           l.reported_at
    FROM core.transactions t
    LEFT JOIN core.fraud_labels l USING (transaction_id)
)
SELECT to_char(month, 'YYYY-MM') AS month,
       COUNT(*)                  AS transactions,
       round(100.0 * COUNT(*) FILTER (
           WHERE is_fraud AND reported_at < month + INTERVAL '1 month' + INTERVAL '7 days')
           / COUNT(*), 3)        AS known_after_7d_pct,
       round(100.0 * COUNT(*) FILTER (
           WHERE is_fraud AND reported_at < month + INTERVAL '1 month' + INTERVAL '30 days')
           / COUNT(*), 3)        AS known_after_30d_pct,
       round(100.0 * COUNT(*) FILTER (
           WHERE is_fraud AND reported_at < month + INTERVAL '1 month' + INTERVAL '60 days')
           / COUNT(*), 3)        AS known_after_60d_pct,
       round(100.0 * COUNT(*) FILTER (WHERE is_fraud) / COUNT(*), 3) AS final_pct
FROM monthly
GROUP BY month
ORDER BY month;
