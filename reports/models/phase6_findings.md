# Phase 6 findings: baselines vs first models

Trained on **2024-10-01 → 2025-06-30** (train period minus the 92-day warm-up, 1,024,160
transactions), evaluated on **validation 2025-07-01 → 2025-09-30** (359,220 transactions,
2,608 labelled fraud, base rate 0.726%). Default hyperparameters (`configs/model.yaml`); the
test period was not loaded. Sources: [`phase6_comparison.md`](phase6_comparison.md),
[`notebooks/04_modeling.ipynb`](../../notebooks/04_modeling.ipynb), MLflow experiment
`phase6-baselines-and-models`.

Alert budget for the operational metrics: top **0.5%** of transactions = 1,797 alerts in 92
days (~19.5 per day). With 2,608 frauds, the best possible recall at that budget is 68.9%.

## 1. Does ML beat simple rules?

| Candidate | PR-AUC (95% CI) | ROC-AUC | Precision @0.5% | Recall @0.5% | Fraud EUR caught @0.5% |
|---|---|---|---|---|---|
| B0 approve all | 0.007 (0.007–0.008) | 0.500 | 0.3% | 0.2% | 0.1% |
| B1 amount | 0.026 (0.022–0.030) | 0.715 | 5.3% | 3.6% | 30.4% |
| B2 seven expert rules | 0.256 (0.235–0.283) | 0.790 | 46.3% | 31.9% | 31.8% |
| M1 logistic regression | 0.766 (0.745–0.792) | 0.941 | 88.4% | 60.9% | 79.8% |
| M2 random forest | 0.765 (0.736–0.794) | 0.944 | 89.0% | 61.3% | 75.5% |
| **M3 hist gradient boosting** | **0.790 (0.768–0.816)** | **0.955** | **91.8%** | **63.2%** | **83.8%** |

Yes, by a wide margin: at the same review budget the best model catches twice the fraud of
the rules (63.2% vs 31.9%) with twice the precision (91.8% vs 46.3%), and 83.8% of the fraud
EUR. Amount alone is almost useless as a ranking (PR-AUC 0.026), even though it flags 30% of
fraud EUR: it finds big frauds and misses everything else.

ROC-AUC is misleading here: the rules reach 0.79, which sounds decent, while their PR-AUC is
0.26. With 137 legit transactions per fraud, ranking a fraud above *a random* legit payment is
easy; making the *alerts* mostly fraud is not.

## 2. Are the differences between models real?

Paired day-block bootstrap (300 resamples, both models scored on the same days):

| A vs B | PR-AUC difference | 95% CI | A better in |
|---|---|---|---|
| gradient boosting vs logistic regression | +0.024 | +0.015 to +0.033 | 100% |
| gradient boosting vs random forest | +0.026 | +0.015 to +0.035 | 100% |
| logistic regression vs random forest | +0.002 | −0.007 to +0.012 | 59% |
| logistic regression vs expert rules | +0.511 | +0.489 to +0.528 | 100% |

The separate intervals of gradient boosting and logistic regression overlap, yet the paired
comparison shows the gain is real. It is also **modest**: +0.024. Hypothesis H4 (gradient
boosting wins through interactions) is supported, but the linear model with log-transformed
features and a one-hot hour gets most of the way. Logistic regression and random forest are
indistinguishable.

## 3. Class weights: no gain in ranking, broken probabilities

| | PR-AUC | Mean predicted probability (observed 0.0073) | Brier |
|---|---|---|---|
| gradient boosting | 0.790 | 0.0072 | 0.0023 |
| gradient boosting, balanced | 0.788 (−0.002, CI −0.007 to +0.010) | 0.0684 (9×) | 0.0121 |
| logistic regression | 0.766 | 0.0072 | 0.0026 |
| logistic regression, balanced | 0.721 (−0.046, CI −0.058 to −0.035) | 0.1233 (17×) | 0.0320 |

`class_weight="balanced"` does not improve ranking (and hurts the linear model). It inflates
every probability: in the top decile the balanced logistic regression predicts 48% fraud
where 6.4% is observed. Without weights both models are already well calibrated on average
(top decile: predicted 6.5% vs observed 6.5% for gradient boosting, 6.6% vs 6.4% for
logistic regression). Calibration across the whole range is examined in Phase 8.

**Decision 018: no class weights and no resampling.** Imbalance is handled where it belongs,
in the decision threshold (Phase 8). SMOTE is not tried: it would distort probabilities the same
way (it changes the training class ratio) and interpolates synthetic transactions between
unrelated cards and customers.

## 4. The rules, one by one (validation)

| Rule | Fires | Precision | Recall |
|---|---|---|---|
| card used ≥ 2 times in the previous hour | 703 | 64.7% | 17.4% |
| ≥ 2 distinct cards on the device in 24 h | 536 | 62.3% | 12.8% |
| micro-charge in the previous hour, card not present | 626 | 43.5% | 10.4% |
| new device and amount > 200 EUR | 561 | 39.6% | 8.5% |
| impossible travel | 45 | 33.3% | 0.6% |
| night + gift cards / luxury / electronics | 1,293 | 19.9% | 9.9% |
| card not present from another country | 10,346 | 10.1% | 40.0% |
| **any rule** | 12,658 | 12.4% | 60.2% |

Several rules are precise but narrow; the broad cross-border rule produces most of the alerts
at 10% precision. A rules engine has to trade one for the other; the model ranks within each
rule's population.

## 5. How much comes from the simulator's artificially strong signals?

Masking device operating system and device type (both constant) and retraining the gradient
boosting: PR-AUC **0.790 → 0.783** (−0.007, paired CI −0.013 to −0.001), recall at budget 63.2%
→ 62.8%. The concern raised in the EDA does not materialise: the model gets the same signal from
device novelty, device age and device sharing.

## 6. Honest caveats

- **These numbers are high because the data is synthetic.** Fraud scenarios in the simulator
  are cleaner and more separable than real fraud. Real-world PR-AUC for card fraud is usually
  far lower. What transfers is the method and the *relative* conclusions.
- **Recall ceiling.** All models fall off a cliff at ~85% recall. The likely explanation, to be
  checked by scenario in Phase 9: stealth fraud (9% of fraud) and legit transactions labelled
  as fraud (friendly fraud, ~4.6% of positive labels) are close to undetectable.
- **The validation base rate (0.73%) is lower than in train (0.85%).** Expected month-to-month
  variation; it matters for calibration (Phase 8).
- **Model selection uses validation.** Comparing families here is a model-selection step. Phase 7
  tunes hyperparameters with temporal cross-validation inside the train period, so validation
  is not used again for tuning; the test period stays untouched until the end.
