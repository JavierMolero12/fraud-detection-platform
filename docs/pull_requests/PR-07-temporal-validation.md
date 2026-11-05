# PR 07 — Temporal validation, tuning and leakage experiments

Branch: `feature/temporal-validation` → `main` (stacked on `feature/modeling`)

## Objective

Choose hyperparameters without looking at validation, measure how much a shuffled
cross-validation would flatter the model (H7), and measure what leaky features would do to the
offline numbers and to production (H8).

## Changes

- `models/validation.py`: expanding-window temporal folds on calendar months with a gap in
  days (sklearn's `TimeSeriesSplit` splits by row count), plus shuffled stratified folds for
  the comparison.
- `models/tuning.py` + `fraud models tune`: GridSearchCV over C for logistic regression,
  RandomizedSearchCV (16 candidates) for gradient boosting, both on the temporal folds with
  PR-AUC; H7 and H8 experiments; everything logged to MLflow (`phase7-temporal-cv-tuning`).
- `configs/model.yaml`: folds, search spaces and the **selected model**.
- **Bug fix**: `build_hist_gradient_boosting` ignored `min_samples_leaf` and `max_features`.
- `notebooks/05_temporal_validation`, `reports/models/phase7_findings.md`, CV tables in
  `reports/models/phase7/`, decisions 010 (resolved) and 020.

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. 9 new: folds, gap, month bounds, search-space parsing, leaky substitution, selected model valid and fully applied) | 68 passed |
| Integration tests | 37 passed |
| Phase 7 job | 33 min, exit 0, run from a clean tree |
| Test period loaded | never |

## Results

| | PR-AUC |
|---|---|
| H7: shuffled CV / temporal CV / out-of-time validation | 0.816 / 0.797 / 0.790 |
| Gradient boosting: default CV → tuned CV | 0.797 → 0.803 |
| Gradient boosting on validation: default → tuned | 0.790 → 0.793 (+0.002, CI −0.004 to +0.008) |
| Logistic regression tuned (C = 10) on validation | 0.766 |
| H8: leaky features offline / in production / clean model | 0.847 / **0.399** / 0.790 |

- **H7 supported, modestly** (+0.019). Month-to-month spread (0.759–0.825) matters more.
- **Tuning gain is within noise**; the tuned model is kept because CV chose it (decision 020).
- **H8 supported dramatically**: the leaky model looks best offline and performs at half the
  honest model's level with the inputs production can actually compute.

## Risks

- One fixed set of folds; month-to-month variance means single-month numbers are ±0.03.
- The randomized search explored 16 points; a finer search is unlikely to matter given the
  flat landscape, but it was not tried.
