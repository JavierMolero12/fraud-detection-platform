# Phase 7 findings: temporal validation, tuning and leakage

Training window **2024-10-01 → 2025-06-30** (1,024,160 transactions), four expanding temporal
folds with a 14-day gap. Validation (Jul–Sep 2025) only confirms choices made by
cross-validation. The test period was not loaded. Sources: `fraud models tune`
(`data/processed/phase7/results.json`), [`notebooks/05_temporal_validation.ipynb`](../../notebooks/05_temporal_validation.ipynb),
MLflow experiment `phase7-temporal-cv-tuning`. Total run time: 33 minutes.

| Fold | Train until | Train rows | Test month | Test rows | Test frauds |
|---|---|---|---|---|---|
| 1 | 2025-02-15 | 511,153 | 2025-03 | 115,608 | 954 |
| 2 | 2025-03-18 | 625,182 | 2025-04 | 113,139 | 1,035 |
| 3 | 2025-04-17 | 737,986 | 2025-05 | 118,711 | 1,038 |
| 4 | 2025-05-18 | 856,274 | 2025-06 | 114,314 | 1,197 |

## 1. H7: shuffled cross-validation is optimistic (modestly, here)

Same model (gradient boosting, Phase 6 defaults), same rows:

| | PR-AUC |
|---|---|
| shuffled stratified 4-fold, mean | **0.816** (folds 0.793–0.829) |
| temporal 4-fold, mean | **0.797** (folds 0.759–0.825) |
| validation, out of time | **0.790** |

Shuffled CV overstates by +0.019 against temporal CV and +0.025 against the out-of-time result;
temporal CV lands within 0.007 of it. **H7 is supported**, but the gap is smaller than the
textbook warning suggests. The likely reason: the biggest source of optimism, features that
see the future, was already removed in Phase 5. What remains comes from training on later
months and from fraud campaigns that end up on both sides of a random split. (Shuffled folds
also train on more rows, 768k vs 511k–856k, which adds a little to their score.)

The more practical lesson is the **month-to-month spread**: the same configuration scores
between 0.759 (April) and 0.825 (June). A single validation month can be off by ±0.03.

## 2. Tuning

**Logistic regression**, grid over C ∈ {0.01, 0.1, 1, 10, 100}: CV 0.783 to 0.786. Regularisation
strength barely matters with a million rows and ~60 inputs; C = 10 is "best" by 0.0004.
Validation 0.766, identical to the Phase 6 default.

**Gradient boosting**, randomized search, 16 candidates × 4 folds (24 min):

- Best CV **0.803**: learning rate 0.021, 400 iterations, 31 leaves, ≥ 50 samples per leaf, L2
  0.68, 80% of features per split. The Phase 6 default scores 0.797 through the same folds.
- **13 of 16 candidates fall between 0.795 and 0.803**: a flat landscape. The two with learning
  rates of 0.18–0.19 collapse to 0.61–0.64 (with no early stopping, a high learning rate
  overfits).
- Fit time is not a reason to choose either way (best: 16 s per fold; default: similar).

**On validation the tuned model is not significantly better:**

| | CV (temporal) | Validation PR-AUC | Precision @0.5% | Recall @0.5% | Fraud EUR caught |
|---|---|---|---|---|---|
| gradient boosting, default | 0.797 | 0.790 | 91.7% | 63.2% | 84.3% |
| **gradient boosting, tuned** | **0.803** | **0.793** | **92.0%** | **63.4%** | **85.9%** |
| logistic regression, tuned | 0.786 | 0.766 | 88.3% | 60.9% | 79.7% |

Tuned minus default, paired day-block bootstrap: **+0.002 (CI −0.004 to +0.008)**, better in 80% of
resamples. The month-to-month noise (±0.03) is several times larger than the tuning gain
(+0.006 in CV).

**Decision 020:** keep the tuned gradient boosting as the selected model. It was chosen by
temporal CV, not by validation; validation does not contradict it, and it costs nothing extra.
But the honest summary is that the Phase 6 defaults were already near the optimum for this data.
Logistic regression stays as the interpretable challenger (−0.024 on validation, paired CI
−0.034 to −0.015).

## 3. H8: leaky features look better offline and fail in production

The same gradient boosting trained with the five deliberately leaky features in place of
their clean counterparts:

| | PR-AUC |
|---|---|
| clean features, temporal CV | 0.797 |
| leaky features, temporal CV | **0.868** (+0.071) |
| clean features, validation | 0.790 |
| leaky features, validation with leaky values (what an offline report shows) | **0.847** (+0.057) |
| leaky features, validation with the values available at decision time (production) | **0.399** (−0.391, paired CI −0.417 to −0.366) |

**H8 is supported, and the size of the effect is the point.** On paper the leaky model is the
best one built so far. In production the scoring service can only compute point-in-time
values (the future does not exist yet), the model receives inputs it never saw in training,
and it performs at **half** the level of the honest model. No offline metric computed on the
same leaky extract would reveal this; only the time-travel test from Phase 5, or deployment
itself, would.

## 4. What changed in the code

- `build_hist_gradient_boosting` ignored `min_samples_leaf` and `max_features`. The search was
  not affected (it sets parameters directly on the estimator), but building the selected model
  from `configs/model.yaml` would have silently produced a different model. Fixed; defaults
  equal scikit-learn's, so Phase 6 is unchanged; a test now checks every selected parameter
  reaches the estimator.
- `configs/model.yaml` records the selected model (`selected_model`), the CV folds and the
  search spaces.
