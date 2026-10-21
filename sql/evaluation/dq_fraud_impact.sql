-- SIMULATION ONLY: how much fraud did data cleaning remove?
-- If the excluded rows were disproportionately fraudulent, cleaning would bias the model.
-- Parameter %(run_id)s.

WITH excluded AS (
    SELECT DISTINCT s.transaction_id
    FROM dq.violations v
    JOIN dq.rules r USING (rule_id)
    JOIN dq.stg_transactions s ON s._raw_id = v.raw_id
    WHERE v.run_id = %(run_id)s
      AND v.table_name = 'transactions'
      AND r.action IN ('reject', 'deduplicate')
      -- an id that was also loaded (the kept copy of an exact duplicate) was not lost
      AND NOT EXISTS (
          SELECT 1 FROM core.transactions t WHERE t.transaction_id = s.transaction_id
      )
),
truth AS (
    SELECT g.transaction_id,
           g.is_fraud_true,
           (e.transaction_id IS NOT NULL) AS lost
    FROM sim.ground_truth g
    LEFT JOIN excluded e USING (transaction_id)
)
SELECT COUNT(*) FILTER (WHERE lost)                                AS lost_transactions,
       COUNT(*) FILTER (WHERE lost AND is_fraud_true)              AS lost_true_fraud,
       round(100.0 * AVG(lost::int), 4)                             AS lost_pct_of_all,
       round(100.0 * COUNT(*) FILTER (WHERE lost AND is_fraud_true)
             / NULLIF(COUNT(*) FILTER (WHERE is_fraud_true), 0), 4) AS lost_pct_of_fraud,
       round(100.0 * AVG(is_fraud_true::int) FILTER (WHERE lost), 3)     AS fraud_rate_lost,
       round(100.0 * AVG(is_fraud_true::int) FILTER (WHERE NOT lost), 3) AS fraud_rate_kept
FROM truth;
