# PR 12 — Monitoring: drift, delayed performance, alerts and retraining rules

Branch: `feature/monitoring` → `main` (stacked on `feature/docker-ci`)

## Objective

Monitor the deployed model the way it would be monitored in production: each week, with only
the data and labels known that day. Replay six months of simulated production with an injected
drift, and answer when each alert fires compared with when the drift started (H9).

## Changes

- **`configs/monitoring.yaml`**: reference period, weekly replay, and every alert threshold.
  All fixed before the replay was run (decision 029).
- **`monitoring/drift.py`**: PSI with bins fitted on the reference (0 and missing values in
  their own bins, unseen categories in "other"), and KS.
- **`monitoring/labels.py`**: reporting-delay distribution estimated at deployment;
  delay-adjusted expected reports; Poisson and binomial tests.
- **`monitoring/run.py`**, one run as of a date:
  - input and score PSI, decision rates;
  - observed vs expected fraud reports among recent payments;
  - weekly cohorts with maturity and recall;
  - alerts, and retraining when label alerts persist;
  - the replay of all runs, and the `monitoring` schema tables (`sql/schema/07_monitoring.sql`).
- **`monitoring/evaluation.py`** (the only monitoring code that reads the simulator's truth),
  run after the replay:
  - the alert timeline against the drift date and the true weekly picture;
  - post-hoc rules, checked on drift-free weeks of 2025.
- **`monitoring/report.py`**: self-contained HTML report. Inline SVG, tooltips, data tables,
  light and dark themes.
- **CLI**: `fraud monitor run --as-of`, `fraud monitor replay`, `fraud monitor report`.
- **Docs**: [`reports/monitoring/phase12_findings.md`](../../reports/monitoring/phase12_findings.md),
  [`monitoring_report.html`](../../reports/monitoring/monitoring_report.html), decisions
  029–030, model card monitoring section, README.

## Validation

| Check | Result |
|---|---|
| Unit tests (15 new: PSI properties and bins, delay distribution, statistical checks, a run never sees the future, drift and label alerts on a synthetic world, maturity, retraining rule, rolling reference, report, simulator truth read only by evaluation code, unique test file names) | 131 passed, 1 skipped on Windows |
| Integration tests (2 new: monitoring refuses to run without batch scores; replay on the small dataset, written and read back, re-run replaces) | 55 passed |
| Replay of the full production period | 42 runs in 2 minutes; 94 alerts in 33 runs |
| `ruff check`, `ruff format --check` | clean |

## Findings

| Rule (fixed in advance) | Runs alerting before the drift | First alert after the drift |
|---|---|---|
| Input PSI ≥ 0.25 | 8 (all `device_age_days`) | never caused by the drift |
| Block rate ×1.5 | 1 (a real January fraud peak) | +46 days |
| Review rate ×1.5 | 0 | +60 days |
| **Labels: observed / expected ≥ 1.25, p < 0.01** | **0** | **+67 days** |
| **Retraining recommended** | 0 | **+74 days** |
| Recall on mature cohorts −5 points | 0 | +123 days |

- **Delay-adjusted calibration was the reliable signal.** It stayed at 0.98–1.10 for 8 runs
  before the drift, then climbed to 1.77, two months before mature-cohort recall could show
  the drop.
- **The 0.10 PSI convention missed the drift.**
  - The payment-mix inputs stepped from about 0.000 to 0.06–0.075 in the first week of March.
  - The noise inside the reference is ≤ 0.009.
  - `device_age_days` alerted every week: it drifts by construction.
- **Post hoc** (a hypothesis, not a result): a rolling reference (each week vs the previous 4,
  PSI ≥ 0.03) would have alerted after 11 days, with 0 false alarms in 22 drift-free weeks of
  2025. But it detects the change in legitimate behaviour, not the new fraud.
- **H9 is not supported as stated.** The input drift that is easy to see is not the one that
  hurts the model: the model still catches 76% of the new fraud, and the harm shows as
  under-predicted fraud in the labels.

## Problems found during the phase

- `device_age_days` and other history-accumulating inputs drift by construction against a
  fixed reference.
- The guard test for the simulator's truth found the Phase 3 data quality evaluation reading it.
  Legitimate, so it is now on the allow-list.
- The first rendered report had odd ticks, an overlapping label, a duplicated legend entry and
  one wrong summary sentence. Fixed after viewing it. Dark mode could not be checked visually.
- The same test file name in `unit/` and `integration/` broke collection again; a new test
  guards against it.

## Risks

- **One drift scenario and one seed.** The post-hoc rules are not adopted until a second
  scenario validates them.
- **No retraining was run.** Whether retraining on 14 May recovers recall is untested.
- **No scheduler runs the weekly monitoring**: `fraud monitor run` exists, nothing calls it.
