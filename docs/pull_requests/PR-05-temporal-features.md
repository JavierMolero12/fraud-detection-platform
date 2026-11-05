# PR 05 — Point-in-time feature engineering

Branch: `feature/temporal-features` → `main` (stacked on `feature/eda`)

## Objective

Build the model features so that every value only uses information available at the moment
of the transaction, prove it with tests, and quantify how much the "obvious", leaky versions
of the same features would flatter the model.

## Changes

- **50 features** in 7 groups (transaction, entity at ts, card velocity, customer behaviour,
  device, geography, labels known at ts). Catalog with definition and rationale per feature:
  `src/fraudplatform/features/catalog.py` → `docs/feature_catalog.md`.
- **SQL** (`sql/features/`):
  - `20_windows.sql`: window functions with `RANGE BETWEEN <w> PRECEDING AND CURRENT ROW
    EXCLUDE GROUP`: strictly earlier transactions, same-second ties excluded.
  - `30_lookups.sql`: LATERAL lookups for what windows cannot do in PostgreSQL
    (`COUNT(DISTINCT)`, medians, previous country, label counts with `reported_at < ts`).
  - `40_assemble.sql`: ratios, z-scores, haversine distance between country centroids
    (country precision, documented), implied speed, smoothed merchant fraud rate.
  - `50_leaky.sql`: five deliberately leaky variants for the leakage experiment (H8).
- `configs/features.yaml` (thresholds, fixed merchant prior from the train period, warm-up),
  `fraud features build` and `fraud features catalog`.
- `notebooks/03_feature_analysis` (train only) and `reports/features/feature_findings.md`.
- `binned_auc()` screening metric (captures non-monotonic effects and NULL as information).
- Decisions 016 (90-day warm-up excluded from training) and 017 (offline windows vs online
  lookups, guarded by tests).

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. catalog checks and 3 new `binned_auc` tests) | 45 passed |
| Integration tests | 37 passed |
| – of which: hand-built scenario with exact expected values (ties, micro-charges, late label, impossible travel, shared device) | 9 |
| – of which: **time-travel test** (delete the future, rebuild, compare the past) | 2: 0 / 50 clean features change; 5 / 5 leaky ones do |
| Full build, 2,852,459 rows | 760 s |

## Findings (train period)

- Strongest features on their own (binned AUC): device age 0.832, merchant fraud rate as
  known at ts 0.820, time since the card's previous use 0.808, merchant category 0.796,
  cross-border 0.789.
- One previous use of the card in the last hour: 16.3% fraud (vs 0.47%); two: 73.2%. Devices
  with 3+ cards in 24 h: 92.3%.
- **`card_prior_fraud_reports` is useless point-in-time** (reports arrive ~15 days later),
  while its naive version scores 0.789: +0.282 of pure leakage. Every leaky variant looks
  better than its honest counterpart (+0.025 to +0.282).
- **H2 not supported univariately:** amount relative to the customer's median (0.725) is not
  better than the raw amount (0.774).
- **Warm-up bias:** 76.8% of July 2024 payments look like "new merchant" vs 24.5% a year later.
  Training will start on 2024-10-01.

## Found and fixed during the phase

- `binned_auc` collapsed zero-inflated features (98% zeros) into one bin and reported 0.500
  for `card_amount_sum_1h` while its plain AUC was 0.720. Fixed (point masses get their own
  bin) with a regression test.

## Risks / not done

- The online (single transaction) implementation and its parity test are Phase 10 work.
- Build time (12.7 min) is fine for batch; no incremental build yet.
- Feature importance of `operating_system` will be inflated by how the simulator was built.
