# PR 03 — Data quality stage and analytical SQL

Branch: `feature/data-quality-sql` → `main` (stacked on `feature/simulator-and-db`)

## Objective

Validate the raw extract with explicit rules, quarantine what cannot be trusted, repair what
can be repaired deterministically, load `core`, and **measure** the rules against the defects
the simulator injected. Add analytical SQL that later phases build on.

## Changes

**Rule catalog and quarantine** (`sql/schema/04_dq.sql`, `sql/data_quality/00_rules.sql`)
- 47 rules in `dq.rules`, each with a category (duplicate, missing, invalid, broken
  relationship, inconsistent, outlier) and an action: 35 `reject`, 6 `deduplicate`,
  4 `repair`, 2 `warn`.
- `dq.violations` is the quarantine: every flagged raw row with rule, run and detail.
- `dq.flag_duplicates()`: generic PL/pgSQL (dynamic SQL) that finds exact copies with
  `ROW_NUMBER()` and id conflicts with `GROUP BY ... HAVING COUNT(*) > 1`.
- `pg_input_is_valid()` (PostgreSQL 16) parses dirty text without raising errors.

**Pipeline** (`sql/data_quality/10..50`, `fraud dq run`)
- Entities → transactions → labels, each building a typed `dq.clean_*` table; then `core` is
  replaced in one transaction. A final check fails the run if any raw row is neither loaded
  nor excluded (no silent losses).
- Repairs only when deterministic: missing customer → card owner; `eur` → `EUR`;
  card-present country → merchant country; placeholder device → `NULL`.
- `core.transactions` now allows an unknown device on online payments
  (`ck_txn_no_device_when_card_present`).

**Report** (`reports/data_quality/dq_report.md`): headline counts in the agreed format, per
table and per rule counts, shared-device warnings, and simulation-only evaluation (recall per
defect type, violations on clean records, fraud removed by cleaning, fraud rate of outliers).

**Analytical SQL** (`sql/analysis/`): spend spikes (LAG + rolling AVG), time between card
uses (LAG + percentile_cont), device sharing (COUNT DISTINCT + HAVING), point-in-time
velocity (RANGE INTERVAL + EXCLUDE GROUP vs. a leaky window), label maturity (FILTER).

## Validation

| Check | Result |
|---|---|
| Unit tests | 24 passed |
| Integration tests (incl. DQ on the `small` dataset and the 5 analysis queries) | 26 passed |
| Recall per injected defect type, full dataset | 12/12 types at 100% |
| Rules firing on records without a defect | only cascades and warnings |
| Raw rows = excluded + loaded, every table | enforced by the pipeline |
| Full run (2.87M raw transactions) | 567 s |

Numbers from the full run (`reports/data_quality/dq_report.md`):

- 2,866,727 raw transactions → 2,852,459 in `core`; 14,268 excluded (0.50%), 10,853 repaired.
- Cleaning removed 7,975 distinct transactions (0.28%) including 76 true frauds. Fraud rate
  of the removed rows 0.953% vs. 0.995% kept: no visible bias.
- Amount outliers (p99.9 per category, kept): 7.49% true fraud vs 0.99% for the rest.
- The shared-device warning surfaced the two placeholder ids (`unknown`: 5,032 customers,
  `00000000`: 2,408) and four real devices with 51–92 customers.

## Findings worth knowing

- **The data quality stage found a bug in the simulator.** `CUS_UNDERAGE_ACCOUNT` flagged 448
  customers opening their account up to 12 h before turning 18: the generator used
  18 × 365.25 days instead of calendar years. Fixed in simulator **v1.0.1**, with a unit
  test. Dataset hash changed `a16b1bf8…` → `350d74bd…` (decision 013).
- **100% recall is not impressive by itself**: I wrote the rules knowing which defects exist.
  The useful results are (a) the zero false alarms on clean records, (b) the bug above, which
  no injected defect predicted, and (c) a measurement framework that will catch regressions.
- **Rejecting 30 customers without country removed 3,094 valid transactions** in cascade
  (decision 012). Documented trade-off.
- **Previews from the analysis queries** (retrospective, final labels; not features yet):
  - E-commerce has ~10× more sub-minute gaps between uses of the same card (0.23% vs 0.02%).
  - Devices used with 11+ cards: 91.5% labelled fraud vs ~1% for single-card devices.
  - One previous use of the card in the last hour: 16.4% fraud vs 0.43% with none.
  - Seven days after a month ends only ~60% of its eventual fraud labels are known; after 30
    days ~85%. Training on recent months with today's labels would understate fraud.

## Risks / not done

- **Performance:** 228 s for transaction rules and 320 s for the `core` load (FK checks and
  six indexes on 2.85M rows). Acceptable for a batch run; dropping and rebuilding indexes
  around the load is the obvious optimisation if it becomes a bottleneck.
- The merchant category list is duplicated between the simulator and
  `10_entities.sql` (no category reference table yet).
- Rules use the whole extract for the outlier and shared-device warnings. That is fine for
  data quality, but the same aggregates would be temporal leakage as model features (Phase 5).
