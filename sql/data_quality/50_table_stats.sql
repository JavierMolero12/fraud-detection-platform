-- Per-table summary of the run: what came in, what was excluded, repaired, flagged and loaded.

WITH row_status AS (
    SELECT v.table_name,
           v.raw_id,
           bool_or(r.action IN ('reject', 'deduplicate')) AS excluded,
           bool_or(r.action = 'repair')                   AS repaired,
           bool_or(r.action = 'warn')                     AS warned
    FROM dq.violations v
    JOIN dq.rules r USING (rule_id)
    WHERE v.run_id = dq.current_run()
    GROUP BY v.table_name, v.raw_id
),
per_table AS (
    SELECT table_name,
           COUNT(*) FILTER (WHERE excluded)                AS excluded_rows,
           COUNT(*) FILTER (WHERE repaired AND NOT excluded) AS repaired_rows,
           COUNT(*) FILTER (WHERE warned AND NOT excluded) AS warned_rows
    FROM row_status
    GROUP BY table_name
),
counts (table_name, raw_rows, loaded_rows) AS (
    VALUES ('customers',    (SELECT COUNT(*) FROM raw.customers),    (SELECT COUNT(*) FROM core.customers)),
           ('cards',        (SELECT COUNT(*) FROM raw.cards),        (SELECT COUNT(*) FROM core.cards)),
           ('merchants',    (SELECT COUNT(*) FROM raw.merchants),    (SELECT COUNT(*) FROM core.merchants)),
           ('devices',      (SELECT COUNT(*) FROM raw.devices),      (SELECT COUNT(*) FROM core.devices)),
           ('transactions', (SELECT COUNT(*) FROM raw.transactions), (SELECT COUNT(*) FROM core.transactions)),
           ('fraud_labels', (SELECT COUNT(*) FROM raw.fraud_labels), (SELECT COUNT(*) FROM core.fraud_labels))
)
INSERT INTO dq.table_stats (run_id, table_name, raw_rows, excluded_rows, repaired_rows,
                            warned_rows, loaded_rows)
SELECT dq.current_run(), c.table_name, c.raw_rows,
       coalesce(p.excluded_rows, 0), coalesce(p.repaired_rows, 0), coalesce(p.warned_rows, 0),
       c.loaded_rows
FROM counts c
LEFT JOIN per_table p USING (table_name);

-- Every raw row is either excluded or loaded: anything else means a silent loss.
DO $$
DECLARE
    leak RECORD;
BEGIN
    SELECT * INTO leak
    FROM dq.table_stats
    WHERE run_id = dq.current_run()
      AND raw_rows <> excluded_rows + loaded_rows
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'row accounting mismatch in %: raw % <> excluded % + loaded %',
            leak.table_name, leak.raw_rows, leak.excluded_rows, leak.loaded_rows;
    END IF;
END
$$;

ANALYZE core.customers, core.cards, core.merchants, core.devices, core.transactions,
        core.fraud_labels;
