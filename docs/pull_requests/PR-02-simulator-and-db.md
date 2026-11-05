# PR 02 — Synthetic data simulator and PostgreSQL

Branch: `feature/simulator-and-db` → `main`

## Objective

Produce a reproducible, realistic dataset with the relational structure required by the design
(customers, cards, merchants, devices, transactions, delayed fraud labels), load it into
PostgreSQL, and put a minimal CI in place from the start.

## Changes

**Simulator** (`src/fraudplatform/simulation/`, parameters in `configs/simulation.yaml`)
- Legit behaviour: per-customer Poisson process with day seasonality and local hour-of-day
  curves, Zipf-distributed usual merchants, exploration, trips abroad, phone upgrades, shared
  family devices, occasional large purchases. These produce realistic false-positive pressure.
- Six base fraud scenarios generated as incidents (card testing, account takeover, card
  cloning, fraud rings, merchant compromise, stealth) plus one drift scenario
  (`wallet_provisioning`) that ramps up from 2026-03-01.
- Delayed and noisy labels: per-incident reporting probability, lognormal delays capped at
  120 days, partially reported card-testing micro-charges, friendly fraud and cleared disputes.
- Injected data quality defects in the raw extract, each one logged in `sim/dq_defects`.
- Named random streams per component (`RngFactory`) and random hex IDs (IDs never encode time).

**Database** (`sql/schema/`, `docker-compose.yml`, `src/fraudplatform/db.py`)
- Schemas `raw`, `core`, `features`, `scoring`, `monitoring`, `sim`.
- `raw`: text-only copies of the source files with a physical row id; `core`: constraints that
  encode business rules (composite FK card↔customer, device iff card-not-present, amount > 0,
  adult account holders) and point-in-time indexes on `(entity, ts)` plus a BRIN on `ts`.
- Loader uses `COPY`, runs in one transaction, checks row counts against the manifest and
  records the dataset hash in `raw.load_log`.

**Tooling**: `fraud simulate | db init | db load` CLI, ruff, pytest, GitHub Actions
(lint → unit tests → integration tests against a PostgreSQL service container).

## Validation

| Check | Result |
|---|---|
| Unit tests (simulation invariants) | 23 passed |
| Integration tests (schema, loader, constraint rejections) | 10 passed |
| `ruff check` / `ruff format --check` | clean |
| Same seed → same dataset hash (full scale, two runs, same machine) | identical `a16b1bf8…` |
| Full-scale generation time | 32.5 s |
| Full-scale load into PostgreSQL | 47.6 s |
| Duplicate transaction IDs found by SQL vs. injected | 6,293 vs 6,293 |

Full dataset report: [`reports/simulation/v1.0.0-full.md`](../../reports/simulation/v1.0.0-full.md).
Key numbers (from that run, not targets):

- 2,860,444 clean transactions, 30,000 customers, 24 months.
- True fraud rate 0.995%; eventually-labelled fraud rate 0.879%.
- 15.7% of true fraud is never labelled; 1,144 legit transactions are labelled as fraud.
- Median label delay 15.5 days; 78.8% of labels arrive within 30 days.
- Fraud rate is 0.74%–1.26% per month until 2026-03 and 1.52%–1.63% in 2026-04..06 (drift).

## Deviations from the design and things worth knowing

- **Scenario mix differs from the configured mix** (e.g. card testing 22.3% vs 25%, card
  cloning 13.2% vs 15%). Incident counts are derived from *average* transactions per incident,
  and the drift scenario adds 10.6% of all fraud on top. Accepted: the configuration is an
  input, the report is what the data actually contains.
- **Monthly fraud rate is noisy** (e.g. 1.16% in 2024-11) because fraud rings and merchant
  breaches are clustered in time. This is realistic and matters for temporal validation.
- Added `merchants.channel` (online/physical), not in the original ER diagram: needed to define
  what a card-present payment at an online merchant means.
- **Reproducibility scope:** the hash was verified twice on Windows with numpy 2.5 / pandas 3.0.
  Bit-identical output across operating systems is *not* guaranteed (floating-point math
  libraries can differ in the last bit, which can flip a rounded amount). CI only checks
  determinism within one platform.
- Reloading a dataset truncates `core` too (cascade from reference tables). Acceptable because
  `core` is rebuilt from `raw` by the data quality stage.

## Risks

- The generator is the source of truth for every pattern the models will find. It must be
  frozen (v1.0.0) before Phase 6; any later change bumps the version and is documented.
- Window-function features over 2.9M rows have not been benchmarked yet (Phase 5).
- Defect injection is verified for internal consistency only; whether SQL rules catch the
  defects is the subject of Phase 3.

## Not in this PR

Data quality rules and the raw → core move (Phase 3), a committed `data/sample/`, LICENSE,
translation of the Phase 1 design documents to English.
