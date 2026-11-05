-- Step 5: DELIBERATELY LEAKY features, for the leakage experiment only (hypothesis H8).
-- Each one is the "obvious" way to write a feature, and each uses information that does not
-- exist at decision time. They are never model inputs in the real pipeline; Phase 7 measures
-- how much they inflate offline metrics. Clean counterparts: see features/catalog.py.

DROP TABLE IF EXISTS features.transaction_features_leaky;
CREATE TABLE features.transaction_features_leaky AS
WITH labelled AS (
    SELECT t.transaction_id, t.ts, t.card_id, t.customer_id, t.merchant_id, t.device_id,
           t.amount_eur, coalesce(l.is_fraud, false) AS is_fraud
    FROM core.transactions t
    LEFT JOIN core.fraud_labels l USING (transaction_id)
),
-- final labels of ALL transactions, including this one and future ones
merchant_rate AS (
    SELECT merchant_id, AVG(is_fraud::int)::double precision AS rate
    FROM labelled GROUP BY merchant_id
),
-- global count over the whole extract, future included
device_customers AS (
    SELECT device_id, COUNT(DISTINCT customer_id) AS customers
    FROM labelled WHERE device_id IS NOT NULL GROUP BY device_id
),
-- median over the customer's whole history, future included
customer_median AS (
    SELECT customer_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY amount_eur) AS median
    FROM labelled GROUP BY customer_id
)
SELECT x.transaction_id,
       -- window centred on ts: counts the next 12 hours too
       COUNT(*) OVER (PARTITION BY x.card_id ORDER BY x.ts
                      RANGE BETWEEN INTERVAL '12 hours' PRECEDING
                                AND INTERVAL '12 hours' FOLLOWING EXCLUDE CURRENT ROW)
           AS leaky_card_txn_count_24h_centered,
       x.amount_eur / NULLIF(cm.median, 0)                    AS leaky_cust_amount_median_alltime,
       dc.customers                                           AS leaky_device_unique_customers_alltime,
       -- labels of earlier transactions, whether or not they had been reported yet
       COUNT(*) FILTER (WHERE x.is_fraud) OVER (PARTITION BY x.card_id ORDER BY x.ts, x.transaction_id
                                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
           AS leaky_card_prior_fraud_naive,
       mr.rate                                                AS leaky_merchant_fraud_rate_alltime
FROM labelled x
JOIN merchant_rate mr USING (merchant_id)
JOIN customer_median cm USING (customer_id)
LEFT JOIN device_customers dc USING (device_id);

ALTER TABLE features.transaction_features_leaky ADD PRIMARY KEY (transaction_id);
