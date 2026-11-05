-- SIMULATION ONLY: violations raised on records with NO injected defect in that table.
-- Warnings and cascades (a clean row whose parent was rejected) are expected and reported
-- separately; any other rule appearing here is a false alarm.
-- Parameter %(run_id)s.

SELECT v.rule_id,
       r.action,
       COUNT(*) AS rows_without_injected_defect
FROM dq.violations v
JOIN dq.rules r USING (rule_id)
WHERE v.run_id = %(run_id)s
  AND NOT EXISTS (
      SELECT 1 FROM sim.dq_defects d
      WHERE d.table_name = v.table_name AND d.record_key = v.record_key
  )
GROUP BY v.rule_id, r.action
ORDER BY rows_without_injected_defect DESC;
