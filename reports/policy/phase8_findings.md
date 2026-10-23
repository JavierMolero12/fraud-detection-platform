# Phase 8 findings: calibration, cost model and decision policy

Model: the selected gradient boosting (Phase 7), trained 2024-10-01 → 2025-06-30. Calibration
and policy chosen on **validation** (Jul–Sep 2025, 359,220 transactions). The bundle was
frozen and committed (`32ce00c`) and then evaluated **once** on **test** (Oct–Dec 2025,
387,978 transactions, `3b95790`). Sources: [`notebooks/06_cost_calibration.ipynb`](../../notebooks/06_cost_calibration.ipynb),
[`policy_v1.json`](policy_v1.json), [`validation_results.json`](validation_results.json),
[`test_evaluation.json`](test_evaluation.json), MLflow experiment `phase8-cost-calibration`.

Cost assumptions ([`configs/costs.yaml`](../../configs/costs.yaml), all assumptions): chargeback
fee €15, review €4, review stops 90% of the fraud it sees, €1 friction for reviewing a legit
customer, a wrongful block costs 2% of the amount + €10, 60 reviews per day.

## 1. The best decision depends on the amount

| Amount | approve below | review between | block above |
|---|---|---|---|
| €5 | 26.3% | 26.3% – 46.0% | 46.0% |
| €20 | 15.4% | 15.4% – 41.9% | 41.9% |
| €100 | 4.8% | 4.8% – 31.1% | 31.1% |
| €200 | 2.6% | 2.6% – 26.1% | 26.1% |
| €1,000 | 0.55% | 0.55% – 19.2% | 19.2% |

(The €20 and €200 rows reproduce the hand calculation in the Phase 1 design; a unit test checks
it.) With a 0.7% base rate, a €1,000 payment is worth a review at a 0.55% fraud probability.

## 2. Calibration: the model was already calibrated

Held-out month (calibrators fitted on Jul–Aug, scored on Sep):

| | Brier | Log loss | ECE | Mean predicted (observed 0.735%) |
|---|---|---|---|---|
| none | 0.002331 | 0.012259 | 0.00035 | 0.751% |
| Platt | 0.002335 | 0.012257 | 0.00048 | 0.748% |
| isotonic | 0.002344 | 0.012649 | 0.00052 | 0.752% |

The pre-registered rule (lowest held-out log loss) picks Platt by 0.000002, a tie. Its fitted
transform is almost the identity (slope 1.03, intercept 0.07). Isotonic is *worse*: it
overfits and creates ties that lower PR-AUC (0.790 → 0.785). Without class weights (decision
018), gradient boosting's log-loss objective already produces calibrated probabilities.

**One weak spot in the decision region** (validation, cross-fitted, fixed bins):

| Predicted | Transactions | Mean predicted | Observed (95% CI) |
|---|---|---|---|
| 0.5–2% | 6,293 | 0.85% | 0.62% (0.45–0.85%) |
| 2–5% | 879 | 3.07% | 2.96% (2.03–4.30%) |
| 5–10% | 390 | 7.00% | 6.67% (4.59–9.59%) |
| **10–20%** | **207** | **14.3%** | **21.7% (16.7–27.9%)** |
| 20–40% | 237 | 29.5% | 35.0% (29.2–41.3%) |
| 40–70% | 568 | 55.8% | 56.9% (52.8–60.9%) |
| 70–100% | 1,909 | 91.2% | 90.5% (89.1–91.7%) |

Between 10% and 20% the model underestimates by ~1.5×, on few transactions. For mid-sized
amounts this band is where review turns into block. Not fixed: the evidence is 207 rows, and
isotonic (the method that could bend this) overfits elsewhere.

## 3. Policies on validation

| Policy | Total cost | Per 1,000 txn | Blocked (legit) | Reviewed | Max reviews/day | Fraud EUR stopped |
|---|---|---|---|---|---|---|
| Approve all | €946,616 | €2,635 | 0 | 0 | 0 | 0% |
| Rules (1 rule: review, 2+: block) | €520,946 | €1,450 | 1,113 (318) | 5,520 | 60 | 48.4% |
| Block if p ≥ 0.5, never review | €95,162 | €265 | – | 0 | 0 | – |
| P1 global thresholds (review ≥ 4.4%, block ≥ 25%) | €63,742 | €177 | 2,651 (527) | 755 | 16 | 94.9% |
| **P2 expected cost per transaction (deployed)** | **€62,925** | **€175** | 2,513 (455) | 1,279 | 24 | **95.3%** |

- The "default" 0.5 threshold costs **49% more** than the optimum.
- P2 beats P1 by only 1.3%: it reviews more and blocks fewer legit customers (455 vs 527)
  because it lets the amount decide.
- **Capacity never binds** for the model policies (24 of 60 reviews on the busiest day): the
  model is confident enough that most decisions are approve or block. The rules hit the cap
  every day.

## 4. Sensitivity: which assumption matters?

Each assumption moved to 50% and 150%; regret = extra cost of keeping the deployed policy
(which still believes the original numbers) vs re-optimising with the true value.

| Assumption | Regret ×0.5 | Regret ×1.5 |
|---|---|---|
| **review stops 90% of fraud** | **+7.3%** (0.45) | +2.1% (1.00) |
| review cost €4 | +1.2% | +0.9% |
| block friction €10 | +1.0% | +0.9% |
| chargeback fee €15 | −0.1% | +0.4% |
| review friction €1 | 0.0% | +0.7% |
| lost margin 2% | 0.0% | +0.3% |
| review capacity 60/day | 0.0% | 0.0% |

The policy is robust to most assumptions (≤ 1.2%). **The one to verify with the fraud team is
how much fraud a review actually stops.** Small negative regrets are luck: the policy
minimises expected, not realised, cost.

## 5. The single test evaluation (Oct–Dec 2025)

| | Validation | **Test** |
|---|---|---|
| PR-AUC (95% CI) | 0.792 | **0.752 (0.718–0.777)** |
| ROC-AUC | 0.955 | 0.953 |
| Precision / recall at top 0.5% | 91.8% / 63.3% | 88.7% / 60.2% |
| Fraud EUR caught at top 0.5% | 85.2% | 83.8% |
| Mean predicted vs observed | 0.73% / 0.73% | 0.73% / 0.74% |
| Deployed policy cost | €62,925 | **€64,098** |
| Approve-all cost | €946,616 | €946,956 |
| Rules cost | €520,946 | €482,886 |
| Cost reduction vs approve-all | 93.4% | **93.2%** |
| Fraud EUR stopped | 95.3% | 95.5% |
| False decline rate | 0.13% | 0.14% |

**The ranking metric dropped** (0.792 → 0.752), outside the validation interval. Two
explanations, not separable with this data: month-to-month variation (Phase 7 folds ranged
0.759–0.825), and model ageing (validation is 0–3 months after training, test 3–6 months).
**The business result did not drop**: cost, fraud EUR stopped and false declines are within a
few percent of validation, and calibration held (mean predicted 0.73% vs 0.74% observed). The
policy is driven by confident, high-value decisions, which degraded less than the overall
ranking.

Phase 12 must watch the PR-AUC trend over time: if ageing is the cause, it will keep falling.
