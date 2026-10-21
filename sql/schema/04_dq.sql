-- Data quality: rule catalog, run log and violations.
--
-- Every rule has an ACTION:
--   reject      : the row is not loaded into core (kept in raw, listed in dq.violations)
--   deduplicate : exact copy of an earlier row; the first copy is loaded
--   repair      : the row is loaded after a deterministic correction (documented per rule)
--   warn        : the row is loaded unchanged; it is only reported
--
-- dq.violations is the quarantine: (run_id, table_name, raw_id) points at the raw row.

CREATE SCHEMA IF NOT EXISTS dq;

CREATE TABLE IF NOT EXISTS dq.rules (
    rule_id     TEXT PRIMARY KEY,
    table_name  TEXT NOT NULL,
    category    TEXT NOT NULL CHECK (category IN (
                    'duplicate', 'missing', 'invalid', 'broken_relationship',
                    'inconsistent', 'outlier')),
    action      TEXT NOT NULL CHECK (action IN ('reject', 'deduplicate', 'repair', 'warn')),
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dq.runs (
    run_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_hash TEXT,
    status       TEXT        NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dq.violations (
    run_id     BIGINT NOT NULL REFERENCES dq.runs ON DELETE CASCADE,
    rule_id    TEXT   NOT NULL REFERENCES dq.rules,
    table_name TEXT   NOT NULL,
    raw_id     BIGINT NOT NULL,
    record_key TEXT,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS ix_violations_row ON dq.violations (run_id, table_name, raw_id);
CREATE INDEX IF NOT EXISTS ix_violations_rule ON dq.violations (run_id, rule_id);

CREATE TABLE IF NOT EXISTS dq.table_stats (
    run_id        BIGINT NOT NULL REFERENCES dq.runs ON DELETE CASCADE,
    table_name    TEXT   NOT NULL,
    raw_rows      BIGINT NOT NULL,
    excluded_rows BIGINT NOT NULL,   -- rejected or duplicate copies
    repaired_rows BIGINT NOT NULL,
    warned_rows   BIGINT NOT NULL,
    loaded_rows   BIGINT NOT NULL,
    PRIMARY KEY (run_id, table_name)
);

-- Run id of the current pipeline execution (set by the runner with set_config).
CREATE OR REPLACE FUNCTION dq.current_run() RETURNS BIGINT
LANGUAGE sql STABLE AS $$ SELECT current_setting('dq.run_id')::BIGINT $$;

-- True when a raw row has a violation that keeps it out of core in the current run.
CREATE OR REPLACE FUNCTION dq.is_excluded(p_table TEXT, p_raw_id BIGINT) RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT EXISTS (
        SELECT 1
        FROM dq.violations v
        JOIN dq.rules r USING (rule_id)
        WHERE v.run_id = dq.current_run()
          AND v.table_name = p_table
          AND v.raw_id = p_raw_id
          AND r.action IN ('reject', 'deduplicate')
    )
$$;

-- Placeholder ids some source systems send when the real device is unknown. The list was
-- found with the TXN_SHARED_DEVICE warning (ids shared by thousands of customers).
CREATE OR REPLACE FUNCTION dq.is_sentinel_device(p_device_id TEXT) RETURNS BOOLEAN
LANGUAGE sql IMMUTABLE AS $$
    SELECT lower(trim(p_device_id)) IN ('unknown', '00000000', 'none', 'null', 'n/a', '0')
$$;

-- Generic duplicate detection for any raw table:
--   * exact duplicate : same content as an earlier row          -> p_exact_rule (deduplicate)
--   * id conflict     : same key, two or more different contents -> p_conflict_rule (reject all;
--                       we cannot know which version is true)
CREATE OR REPLACE FUNCTION dq.flag_duplicates(
    p_table TEXT, p_key TEXT, p_exact_rule TEXT, p_conflict_rule TEXT
) RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    content_cols TEXT;
BEGIN
    SELECT string_agg(quote_ident(column_name), ', ' ORDER BY ordinal_position)
      INTO content_cols
      FROM information_schema.columns
     WHERE table_schema = 'raw'
       AND table_name = p_table
       AND column_name NOT IN ('_raw_id', '_loaded_at');

    EXECUTE format($q$
        INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
        SELECT dq.current_run(), %1$L, %2$L, _raw_id, %3$I, 'copy of raw row ' || first_raw_id
        FROM (
            SELECT _raw_id, %3$I,
                   ROW_NUMBER() OVER w  AS copy_number,
                   FIRST_VALUE(_raw_id) OVER w AS first_raw_id
            FROM raw.%2$I
            WINDOW w AS (PARTITION BY %4$s ORDER BY _raw_id)
        ) d
        WHERE copy_number > 1
    $q$, p_exact_rule, p_table, p_key, content_cols);

    EXECUTE format($q$
        INSERT INTO dq.violations (run_id, rule_id, table_name, raw_id, record_key, detail)
        SELECT dq.current_run(), %1$L, %2$L, t._raw_id, t.%3$I,
               c.versions || ' different versions of this id'
        FROM raw.%2$I t
        JOIN (
            SELECT %3$I, COUNT(*) AS versions
            FROM (SELECT DISTINCT %4$s FROM raw.%2$I) distinct_rows
            WHERE %3$I IS NOT NULL
            GROUP BY %3$I
            HAVING COUNT(*) > 1
        ) c USING (%3$I)
    $q$, p_conflict_rule, p_table, p_key, content_cols);
END
$$;
