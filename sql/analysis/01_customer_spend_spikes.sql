-- Question: which customers suddenly spend far more than their own recent average?
-- Techniques: CTEs, GROUP BY, date_trunc, LAG, rolling AVG window, conditional filter.
--
-- A month is a "spike" when spend > 3x the customer's average of the 3 previous months.
-- Note the window ends at 1 PRECEDING: the current month is never part of its own baseline.

WITH monthly AS (
    SELECT customer_id,
           date_trunc('month', ts) AS month,
           COUNT(*)                AS transactions,
           SUM(amount_eur)         AS spend_eur
    FROM core.transactions
    GROUP BY customer_id, date_trunc('month', ts)
),
with_baseline AS (
    SELECT customer_id,
           month,
           transactions,
           spend_eur,
           LAG(spend_eur) OVER w AS previous_month_eur,
           AVG(spend_eur) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS baseline_3m_eur,
           COUNT(*) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS baseline_months
    FROM monthly
    WINDOW w AS (PARTITION BY customer_id ORDER BY month)
)
SELECT customer_id,
       to_char(month, 'YYYY-MM')               AS month,
       transactions,
       round(spend_eur, 2)                     AS spend_eur,
       round(baseline_3m_eur, 2)               AS baseline_3m_eur,
       round(spend_eur / baseline_3m_eur, 1)   AS ratio_to_baseline
FROM with_baseline
WHERE baseline_months = 3          -- need a full baseline
  AND spend_eur > 3 * baseline_3m_eur
ORDER BY ratio_to_baseline DESC
LIMIT 20;
