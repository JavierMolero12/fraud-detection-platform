-- Fraud label rules. A label is only as good as the transaction it points to.

SELECT dq.flag_duplicates('fraud_labels', 'transaction_id', 'LBL_DUP_EXACT', 'LBL_DUP_ID_CONFLICT');

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'LBL_INVALID_VALUE', 'fraud_labels', _raw_id, transaction_id, problems
FROM (
    SELECT _raw_id, transaction_id,
           concat_ws('; ',
               CASE WHEN transaction_id IS NULL THEN 'transaction_id is null' END,
               CASE WHEN is_fraud IS NULL OR is_fraud NOT IN ('true', 'false')
                    THEN 'is_fraud = ' || coalesce(is_fraud, 'null') END,
               CASE WHEN label_source IS NULL OR label_source NOT IN
                        ('chargeback', 'customer_report', 'investigation')
                    THEN 'unknown label_source ' || coalesce(label_source, 'null') END,
               CASE WHEN NOT pg_input_is_valid(coalesce(reported_at, ''), 'timestamptz')
                    THEN 'bad reported_at' END
           ) AS problems
    FROM raw.fraud_labels
) checked
WHERE problems <> '';

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(),
       CASE WHEN EXISTS (SELECT 1 FROM raw.transactions t WHERE t.transaction_id = l.transaction_id)
            THEN 'LBL_PARENT_REJECTED' ELSE 'LBL_UNKNOWN_TRANSACTION' END,
       'fraud_labels', l._raw_id, l.transaction_id,
       'is_fraud = ' || coalesce(l.is_fraud, 'null')
FROM raw.fraud_labels l
WHERE l.transaction_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM dq.clean_transactions t WHERE t.transaction_id = l.transaction_id
  );

INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
SELECT dq.current_run(), 'LBL_REPORTED_BEFORE_TXN', 'fraud_labels', l._raw_id, l.transaction_id,
       'reported ' || l.reported_at || ', transaction ' || t.ts
FROM raw.fraud_labels l
JOIN dq.clean_transactions t USING (transaction_id)
WHERE pg_input_is_valid(coalesce(l.reported_at, ''), 'timestamptz')
  AND l.reported_at::timestamptz <= t.ts;

DROP TABLE IF EXISTS dq.clean_fraud_labels;
CREATE UNLOGGED TABLE dq.clean_fraud_labels AS
SELECT DISTINCT ON (transaction_id)
       _raw_id, transaction_id, is_fraud::boolean AS is_fraud,
       reported_at::timestamptz AS reported_at, label_source
FROM raw.fraud_labels l
WHERE NOT dq.is_excluded('fraud_labels', l._raw_id)
ORDER BY transaction_id, _raw_id;
