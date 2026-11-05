-- Question: how much time passes between two uses of the same card, by payment method?
-- Techniques: LAG over PARTITION BY, percentile_cont (ordered-set aggregate), FILTER.
--
-- Very short gaps (bursts) are typical of automated card testing.

WITH gaps AS (
    SELECT payment_method,
           extract(epoch FROM ts - LAG(ts) OVER (PARTITION BY card_id ORDER BY ts, transaction_id))
               AS seconds_since_previous
    FROM core.transactions
)
SELECT payment_method,
       COUNT(seconds_since_previous)                                           AS gaps,
       round((percentile_cont(0.05) WITHIN GROUP (ORDER BY seconds_since_previous) / 60)::numeric, 1)
           AS p05_minutes,
       round((percentile_cont(0.50) WITHIN GROUP (ORDER BY seconds_since_previous) / 3600)::numeric, 1)
           AS median_hours,
       COUNT(*) FILTER (WHERE seconds_since_previous < 60)                     AS under_1_minute,
       round(100.0 * COUNT(*) FILTER (WHERE seconds_since_previous < 60)
             / NULLIF(COUNT(seconds_since_previous), 0), 3)                    AS under_1_minute_pct
FROM gaps
WHERE seconds_since_previous IS NOT NULL
GROUP BY payment_method
ORDER BY under_1_minute_pct DESC;
