# Phase 7 findings: temporal validation, tuning and leakage

Training window **2024-10-01 → 2025-06-30** (1,024,160 transactions), four expanding temporal
folds with a 14-day gap. Validation (Jul–Sep 2025) only confirms choices made by
cross-validation. The test period was not loaded. Sources: `fraud models tune`
(`data/processed/phase7/results.json`), [`notebooks/05_temporal_validation.ipynb`](../../notebooks/05_temporal_validation.ipynb),
MLflow experiment `phase7-temporal-cv-tuning`. Total run time: 36 minutes.

> **Re-run in Phase 13 with a fixed row order.** This phase loaded its frames through a query
> without `ORDER BY` that PostgreSQL ran in parallel, so the rows came in a different order on
> every load. Row order changes gradient boosting (bin edges from a 200k-row subsample) and the
> shuffled folds. The from-scratch reproduction exposed it
> ([reproducibility, section 6](../../docs/reproducibility.md#6-what-the-reproduction-found)).
> Every number below comes from the re-run with the fix, which gives the same numbers every
> time. The original values are in [section 5](#5-the-original-run-before-the-row-order-fix).
> No conclusion changed, and the search selected the same configuration.

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
| shuffled stratified 4-fold, mean | **0.817** (folds 0.810–0.825) |
| temporal 4-fold, mean | **0.799** (folds 0.768–0.822) |
| validation, out of time | **0.790** |

Shuffled CV overstates by +0.018 against temporal CV and +0.027 against the out-of-time result;
temporal CV lands within 0.009 of it. **H7 is supported**, but the gap is smaller than the
textbook warning suggests. The likely reason: the biggest source of optimism, features that
see the future, was already removed in Phase 5. What remains comes from training on later
months and from fraud campaigns that end up on both sides of a random split. (Shuffled folds
also train on more rows, 768k vs 511k–856k, which adds a little to their score.)

The more practical lesson is the **month-to-month spread**: the same configuration scores
between 0.768 (April) and 0.822 (June). A single validation month can be off by ±0.03.

The out-of-time model is Phase 6's M3: same parameters, same rows, same order. It gives the
same probabilities bit for bit on all 359,220 validation payments, so the same 0.7901.

## 2. Tuning

**Logistic regression**, grid over C ∈ {0.01, 0.1, 1, 10, 100}: CV 0.783 to 0.786. Regularisation
strength barely matters with a million rows and ~60 inputs; C = 10 is "best" by 0.0004.
Validation 0.766, identical to the Phase 6 default.

**Gradient boosting**, randomized search, 16 candidates × 4 folds (22 min):

- Best CV **0.803**: learning rate 0.021, 400 iterations, 31 leaves, ≥ 50 samples per leaf, L2
  0.68, 80% of features per split. The Phase 6 default scores 0.799 through the same folds.
- **13 of 16 candidates fall between 0.795 and 0.803**: a flat landscape. The two with learning
  rates of 0.18–0.19 collapse to 0.60–0.64 (with no early stopping, a high learning rate
  overfits).
- Fit time is not a reason to choose either way (best: 15 s per fold; default: similar).

**On validation the tuned model is not significantly better:**

| | CV (temporal) | Validation PR-AUC | Precision @0.5% | Recall @0.5% | Fraud EUR caught |
|---|---|---|---|---|---|
| gradient boosting, default | 0.799 | 0.790 | 91.8% | 63.2% | 83.8% |
| **gradient boosting, tuned** | **0.803** | **0.794** | **91.7%** | **63.2%** | **85.5%** |
| logistic regression, tuned | 0.786 | 0.766 | 88.3% | 60.9% | 79.7% |

Tuned minus default, paired day-block bootstrap: **+0.004 (CI −0.001 to +0.010)**, better in 93% of
resamples. A small real gain is plausible but not established. The month-to-month noise
(±0.03) is several times larger than the tuning gain (+0.004 in CV).

**Decision 020:** keep the tuned gradient boosting as the selected model. It was chosen by
temporal CV, not by validation; validation does not contradict it, and it costs nothing extra.
But the honest summary is that the Phase 6 defaults were already near the optimum for this data.
Logistic regression stays as the interpretable challenger (−0.024 on validation, paired CI
−0.033 to −0.015).

## 3. H8: leaky features look better offline and fail in production

The same gradient boosting trained with the five deliberately leaky features in place of
their clean counterparts:

| | PR-AUC |
|---|---|
| clean features, temporal CV | 0.799 |
| leaky features, temporal CV | **0.869** (+0.070) |
| clean features, validation | 0.790 |
| leaky features, validation with leaky values (what an offline report shows) | **0.852** (+0.062) |
| leaky features, validation with the values available at decision time (production) | **0.401** (−0.389, paired CI −0.414 to −0.365) |

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

## 5. The original run, before the row-order fix

The numbers this report showed until Phase 13, from a run whose rows came in an order nobody
chose:

| | Original run | Re-run, fixed order |
|---|---|---|
| H7 shuffled CV, mean (folds) | 0.816 (0.793–0.829) | 0.817 (0.810–0.825) |
| H7 temporal CV, mean (folds) | 0.797 (0.759–0.825) | 0.799 (0.768–0.822) |
| H7 validation, out of time | 0.790 | 0.790 |
| Logistic regression, best CV / validation | 0.786 / 0.766 | 0.786 / 0.766 |
| Gradient boosting, best CV / validation | 0.803 / 0.793 | 0.803 / 0.794 |
| Tuned minus default on validation | +0.002 (CI −0.004 to +0.008), 80% | +0.004 (CI −0.001 to +0.010), 93% |
| H8 leaky offline / served | 0.847 / 0.399 | 0.852 / 0.401 |

- **Row order alone is worth up to 0.003** in the means, and up to 0.03 in a single shuffled
  fold (fold 3: 0.793 → 0.825). That is the size of the tuning gain on validation, which the
  report already called noise.
- **Logistic regression did not move** in the first four decimals: its fit does not depend on
  row order beyond rounding.
- **The model in the bundle is not this exact model.** Phase 8 trains the selected
  configuration as written in `configs/model.yaml`, where the learning rate and the L2 penalty
  are rounded (0.021 and 0.68 for the 0.020970… and 0.679657… the search drew). It scores
  0.792 on validation against 0.794 here, a difference of the same size as row order.
