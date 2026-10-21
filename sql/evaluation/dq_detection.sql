-- SIMULATION ONLY: compare the data quality rules with the defects the simulator injected.
-- In real life the injected-defect log does not exist; this is how we measure the rules.
--
-- detected     : injected defects caught by the rule(s) designed for them (recall)
-- Run with psycopg, parameter %(run_id)s.

WITH expected (table_name, defect_type, rule_id) AS (
    VALUES
        ('transactions', 'exact_duplicate',       'TXN_DUP_EXACT'),
        ('transactions', 'duplicate_id_conflict', 'TXN_DUP_ID_CONFLICT'),
        ('transactions', 'invalid_amount',        'TXN_INVALID_AMOUNT'),
        ('transactions', 'invalid_currency',      'TXN_INVALID_CURRENCY'),
        ('transactions', 'invalid_currency',      'TXN_CURRENCY_FORMAT'),
        ('transactions', 'invalid_timestamp',     'TXN_INVALID_TIMESTAMP'),
        ('transactions', 'invalid_timestamp',     'TXN_FUTURE_TIMESTAMP'),
        ('transactions', 'invalid_timestamp',     'TXN_BEFORE_CARD_ISSUE'),
        ('transactions', 'missing_customer_id',   'TXN_MISSING_CUSTOMER'),
        ('transactions', 'orphan_reference',      'TXN_UNKNOWN_MERCHANT'),
        ('transactions', 'orphan_reference',      'TXN_UNKNOWN_CUSTOMER'),
        ('transactions', 'orphan_reference',      'TXN_CARD_CUSTOMER_MISMATCH'),
        ('transactions', 'country_mismatch',      'TXN_COUNTRY_MISMATCH'),
        ('transactions', 'sentinel_device',       'TXN_SENTINEL_DEVICE'),
        ('customers',    'exact_duplicate',       'CUS_DUP_EXACT'),
        ('customers',    'missing_country',       'CUS_MISSING_COUNTRY'),
        ('cards',        'orphan_reference',      'CRD_ORPHAN_CUSTOMER')
),
injected AS (
    SELECT DISTINCT table_name, defect_type, detail, record_key
    FROM sim.dq_defects
),
flagged AS (
    SELECT DISTINCT e.table_name, e.defect_type, v.record_key, v.rule_id
    FROM dq.violations v
    JOIN expected e USING (rule_id)
    WHERE v.run_id = %(run_id)s
)
SELECT i.table_name,
       i.defect_type,
       COUNT(DISTINCT i.record_key)                                         AS injected,
       COUNT(DISTINCT i.record_key) FILTER (WHERE f.record_key IS NOT NULL) AS detected,
       string_agg(DISTINCT f.rule_id, ', ')                                 AS caught_by
FROM injected i
LEFT JOIN flagged f
       ON f.table_name = i.table_name
      AND f.defect_type = i.defect_type
      AND f.record_key = i.record_key
GROUP BY i.table_name, i.defect_type
ORDER BY i.table_name, i.defect_type;
