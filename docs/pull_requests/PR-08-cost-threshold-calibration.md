# PR 08 — Calibration, cost model, decision policy and the test evaluation

Branch: `feature/cost-threshold-calibration` → `main` (stacked on `feature/temporal-validation`)

## Objective

Turn fraud probabilities into approve / review / block decisions that minimise cost under
explicit assumptions and a daily review capacity, check that the probabilities mean what they
say, test how fragile the result is to the assumptions, and evaluate the frozen result once on
the test period.

## Changes

- `configs/costs.yaml`: the cost assumptions of the Phase 1 design, documented as assumptions.
- `decision/costs.py`: cost model (amount-dependent), break-even probabilities, global-threshold
  policy (P1) and expected-cost policy (P2), both with a daily review capacity; threshold grid.
- `decision/calibration.py`: Platt and isotonic, reliability tables, ECE, and a plain-JSON
  calibrator the API can apply with numpy.
- `decision/optimise.py` + `fraud policy optimise`: calibration chosen forward in time and
  cross-fitted by month, policy chosen on validation, sensitivity analysis, frozen bundle.
- `decision/test_evaluation.py` + `fraud policy evaluate-test`: runs **once**; the committed
  result file blocks a second run.
- `notebooks/06_cost_calibration`, `reports/policy/` (policy, validation and test results,
  findings), decisions 021 and 022.

## Commit order that matters

1. `32ce00c` freezes the policy, the calibrator and the hashes of the model files.
2. `3b95790` records the test evaluation.

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. 20 new: design hand calculation, costs, capacity, P2, calibration, JSON round-trip, sensitivity regret, run-once guard) | 89 passed |
| Integration tests | 37 passed |
| Second `fraud policy evaluate-test` | refused (`TestAlreadyEvaluated`) |

## Results

| | Validation | Test |
|---|---|---|
| PR-AUC | 0.792 | **0.752** (0.718–0.777) |
| Deployed policy cost | €62,925 | **€64,098** |
| Approve-all cost | €946,616 | €946,956 |
| Rules cost | €520,946 | €482,886 |
| Fraud EUR stopped | 95.3% | 95.5% |
| False decline rate | 0.13% | 0.14% |

- Block-if-p ≥ 0.5 costs 49% more than the optimum on validation.
- The model was already calibrated; Platt ≈ identity, isotonic overfits. Underestimation in the
  10–20% band (21.7% observed), on 207 transactions.
- Most sensitive assumption: how much fraud a review stops (+7.3% regret at 45%).
- Test PR-AUC dropped 0.04 (month variation or model ageing); the business result held.

## Bugs found during the phase

- The sensitivity analysis re-decided the deployed policy with the *changed* costs, which
  re-optimises it: every regret was exactly 0. Fixed, with a regression test.
- The validation column of the findings table first mixed numbers from the Phase 7 run of the
  model; replaced with the Phase 8 model's own numbers before committing.

## Risks

- Every cost number is an assumption; the reviewer-effectiveness one matters most.
- Review capacity never binds with this (synthetic, very separable) data; with real data it
  probably would.
