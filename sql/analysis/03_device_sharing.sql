-- Question: how many customers and cards share a device, and is sharing related to fraud?
-- Techniques: CTEs, COUNT(DISTINCT), CASE bucketing, HAVING, conditional aggregation.
--
-- RETROSPECTIVE view: it uses the whole history and the final labels. Fine for analysis,
-- but it is exactly the kind of aggregate that must NOT become a model feature as-is
-- (a feature may only count customers seen on the device BEFORE each transaction).

WITH device_usage AS (
    SELECT t.device_id,
           COUNT(*)                                       AS transactions,
           COUNT(DISTINCT t.customer_id)                  AS customers,
           COUNT(DISTINCT t.card_id)                      AS cards,
           COUNT(*) FILTER (WHERE l.is_fraud)             AS labelled_fraud
    FROM core.transactions t
    LEFT JOIN core.fraud_labels l USING (transaction_id)
    WHERE t.device_id IS NOT NULL
    GROUP BY t.device_id
),
bucketed AS (
    SELECT CASE WHEN cards = 1 THEN '1 card'
                WHEN cards <= 3 THEN '2-3 cards'
                WHEN cards <= 10 THEN '4-10 cards'
                ELSE '11+ cards' END AS cards_on_device,
           *
    FROM device_usage
)
SELECT cards_on_device,
       COUNT(*)                                                  AS devices,
       SUM(transactions)                                         AS transactions,
       round(AVG(customers), 2)                                  AS avg_customers,
       round(100.0 * SUM(labelled_fraud) / SUM(transactions), 3) AS labelled_fraud_pct
FROM bucketed
GROUP BY cards_on_device
HAVING COUNT(*) > 0
ORDER BY MIN(cards);
