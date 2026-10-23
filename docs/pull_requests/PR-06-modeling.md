# PR 06 — Baselines and first models

Branch: `feature/modeling` → `main` (stacked on `feature/temporal-features`)

## Objective

Find out whether machine learning is worth it compared with simple rules, compare three model
families with default hyperparameters, and settle how to handle class imbalance. Every run is
tracked in MLflow with its data, features and code version.

## Changes

- **Baselines** (`models/baselines.py`): B0 approve-all, B1 rank by amount, B2 seven expert
  rules written from the train-period EDA (thresholds in `configs/model.yaml`).
- **Pipelines** (`models/pipelines.py`): logistic regression (median imputation with missing
  indicators, signed log1p, scaling, one-hot including the local hour), random forest and
  histogram gradient boosting (ordinal-encoded categoricals, native NaN). All preprocessing is
  inside the pipeline, fitted on training rows only.
- **Metrics** (`models/metrics.py`): PR-AUC with a day-block bootstrap interval, a *paired*
  day-block bootstrap for differences, ROC-AUC, Brier, log loss, and precision / recall /
  fraud EUR caught at a fixed alert budget (top 0.5%).
- **Tracking** (`models/tracking.py`): MLflow (local SQLite) with lineage tags: dataset hash,
  simulator version, DQ run, feature build, feature set version, git commit and dirty flag.
  Models are saved with **skops** and an explicit trusted-type list (decision 019).
- `fraud models compare`: trains on 2024-10-01 → 2025-06-30 (warm-up excluded, now exactly
  92 days), evaluates on validation, writes `reports/models/phase6_comparison.md`.
- `notebooks/04_modeling` and `reports/models/phase6_findings.md`; decisions 018 and 019.

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. 15 new: rules, pipelines learn, unseen categories, skops round-trip, metrics, paired bootstrap) | 60 passed |
| Integration tests | 37 passed |
| Comparison re-run from a clean tree | identical numbers; report lineage `f032bbb` (dirty: False) |
| Test split loaded | never |

## Results (validation, 359,220 transactions, base rate 0.726%)

| | PR-AUC | Precision @0.5% | Recall @0.5% |
|---|---|---|---|
| Amount | 0.026 | 5.3% | 3.6% |
| Expert rules | 0.256 | 46.3% | 31.9% |
| Logistic regression | 0.766 | 88.4% | 60.9% |
| Random forest | 0.765 | 89.0% | 61.3% |
| **Hist gradient boosting** | **0.790** | **91.8%** | **63.2%** (ceiling 68.9%) |

- Gradient boosting beats logistic regression by **+0.024** (paired CI +0.015 to +0.033):
  real, modest. Logistic regression ≈ random forest.
- `class_weight="balanced"`: no ranking gain (−0.002 for boosting, −0.046 for logistic
  regression) and probabilities inflated 9–17×. Decision 018: no weights, no resampling.
- Masking the artificially strong device OS/type costs only 0.007 PR-AUC.

## Problems found during the phase

- MLflow 3 refused to save the models: skops does not load unknown types. Fixed by declaring
  the trusted types instead of switching to pickle; a test verifies the round-trip.
- The warm-up was 90 days, so training started on 2024-09-29 instead of the documented
  2024-10-01. Changed to 92 days.
- A notebook explanation claimed the 0.5 threshold flags *fewer* transactions than the budget;
  the data shows it flags more (2,268 vs 1,797). Text corrected.

## Risks / caveats

- Absolute numbers are high because the data is synthetic; the comparisons are what transfers.
- All models hit a recall cliff around 85% (likely stealth fraud and friendly-fraud labels):
  to be confirmed by scenario in Phase 9.
- `mlflow` is now a core dependency; the API image (Phase 11) should not need it.
