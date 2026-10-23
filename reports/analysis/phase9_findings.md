# Phase 9 findings: error analysis and interpretability

The frozen bundle of Phase 8 (gradient boosting + Platt calibration + expected-cost policy) on
**validation** (Jul–Sep 2025, 359,220 transactions). The test period is not analysed: it has
been used once, and studying its errors to change the model would turn it into another
validation set. Sources: `fraud analyse validation` (`data/processed/phase9/`, 16.5 min),
[`notebooks/07_error_analysis.ipynb`](../../notebooks/07_error_analysis.ipynb).

Errors are defined at the decision level: a **false positive** is a payment labelled legit and
**blocked**; a **false negative** is a payment labelled fraud and **approved**. The simulator's
ground truth explains them; it is never a model input.

| Outcome (by label) | Transactions |
|---|---|
| legit approved | 355,002 |
| fraud blocked | 2,058 |
| legit reviewed | 1,155 |
| **legit blocked (false positive)** | **455** |
| **fraud approved (false negative)** | **429** |
| fraud reviewed | 121 |

## 1. Most measured errors are label noise

| | Measured | Truly an error | Label noise |
|---|---|---|---|
| False positives | 455 | **128** | 327 are fraud that was never reported |
| False negatives | 429 | **276** (221 of them stealth) | 153 are legit payments disputed as fraud |

- **All 198 blocked micro-charges (< €5) are real fraud**: unreported card testing (customers
  rarely dispute a €2 charge). The model was right; the label was not.
- The **real false decline rate is 0.036%** (128 of 356,311 truly legit payments), not the
  measured 0.128%: 3.6× lower.
- 153 of the "missed frauds" are legit purchases the customer later disputed (friendly fraud):
  approving them was correct.

In real life the ground truth is not available. The practical lesson: **a sample of blocked
"legit" payments should be investigated before trusting the false positive rate**, and
friendly-fraud disputes should be tracked separately from fraud labels.

The two "error" examples explained with SHAP below both turn out to be correct decisions: the
most confident "false positive" is unreported card testing, and the largest "false negative"
(€767) is a disputed legit purchase.

## 2. What the model misses: stealth fraud

What happened to each true scenario (blocked or reviewed = stopped):

| True scenario | Transactions | Stopped | Approved | Median p |
|---|---|---|---|---|
| account takeover | 799 | 99.7% | 0.3% | 0.93 |
| merchant compromise | 511 | 99.8% | 0.2% | 0.94 |
| fraud ring | 149 | 100% | 0% | 0.93 |
| card cloning | 470 | 95.1% | 4.9% | 0.87 |
| card testing | 694 | 91.5% | 8.5% | 0.61 |
| **stealth** | **286** | **2.8%** | **97.2%** | **0.0016** |

(Median p of the labelled transactions of each scenario.)

Stealth fraud (usual merchant, usual device, usual amount) is invisible to the model, by
design of the simulator: it is the irreducible floor. Together with friendly fraud it explains
**87% of the false negatives** and the recall cliff at ~85% seen in Phase 6. The remaining real
misses are 31 card-testing and 22 cloning transactions.

## 3. What drives the model

**Permutation importance** (drop in PR-AUC on validation, 5 repetitions), by feature family
(all columns of a family shuffled together) and by feature:

| Family | Drop | | Feature | Drop |
|---|---|---|---|---|
| entity (device age, card type, …) | 0.169 | | device_age_days | 0.140 |
| transaction (border, category, amount, …) | 0.164 | | cross_border | 0.068 |
| device (sharing, novelty) | 0.102 | | merchant_category | 0.067 |
| customer behaviour | 0.056 | | device_unique_customers_30d | 0.029 |
| card velocity | 0.045 | | merchant_new_for_customer | 0.028 |
| labels known at ts | 0.034 | | device_prior_fraud_reports | 0.023 |
| geography | 0.011 | | device_unique_cards_24h | 0.021 |

- **22 of 49 features add nothing measurable** (importance ≤ 0.0005), including the operating
  system (rank 44), impossible travel, card_prior_fraud_reports and customer age. They are
  candidates for removal in a future version; the frozen model is not changed.
- **Correlated features hide each other.** `card_txn_count_1h` ranks 48th by permutation (the
  model falls back on `card_seconds_since_prev`) but 15th by SHAP. Shuffled together, the card
  velocity family costs 0.045, about twice the sum of its members shuffled one by one.
- **The operating system**, artificially strong in this data (EDA), is not relied upon: rank 44
  by permutation, 21 by SHAP.

**SHAP values** (log-odds; 900 validation payments covering every outcome) agree on the top
three: device age (mean |SHAP| 0.84), cross-border (0.45), merchant category (0.42).

### A warning about SHAP on this model

`shap.TreeExplainer`, the usual fast choice for tree models, **returns wrong values for
HistGradientBoosting with native categorical splits**: on this model the attributions miss the
model output by up to 4.6 log-odds, on every row, with no warning. It treats categorical splits
as numeric thresholds. Found by checking additivity (base value + attributions = model output);
a toy model without categorical splits gives errors of 1e-14. The explanations here use
`shap.PermutationExplainer` (model-agnostic, exact) and every run checks additivity (max error
4.1e-14).

What SHAP values are *not*: causal effects ("changing the device would lower the risk by…"),
facts about the fraud itself, or independent of the background sample. Correlated features
split credit arbitrarily.

## 4. Where errors concentrate (by label)

- **False positives are over-represented** in cross-border payments (15×), desktop devices
  (10×), amounts under €5 (8.6×, all of them real fraud), luxury (5.7×), digital goods (3.8×)
  and night hours (4.4×). After removing label noise, the **real false positives are mostly
  cross-border purchases of €100–500** (102 of 128): travellers buying electronics, fuel,
  groceries abroad.
- **False negatives are over-represented** where stealth fraud lives: home goods, utilities,
  transport, restaurants, health (fraud approved 66–100% of the time), contactless payments
  (42.8% approved) and **€5–20 payments (86.6% approved)**. The last one is partly a business
  decision: the expected-cost policy does not spend a €4 review on a €10 payment.

## 5. Segment analysis

| Segment | PR-AUC | Fraud EUR stopped | False decline rate |
|---|---|---|---|
| card not present | 0.825 | 97.0% | 0.169% |
| card present | **0.667** | 79.4% | 0.071% |
| contactless | **0.544** | 67.5% | 0.051% |
| mobile wallet | 0.888 | 99.3% | 0.052% |
| cross-border | 0.925 | 99.2% | 1.941% |
| domestic | **0.602** | 90.6% | 0.047% |
| €5–20 | **0.140** | 14.0% | 0.000% |
| ≥ €500 | 0.889 | 98.0% | 0.558% |

- **Card-present and domestic fraud is harder**: no device signal, no country mismatch.
  Contactless is the weakest channel.
- **Customer characteristics**: PR-AUC between 0.71 and 0.84 across age bands and segments;
  false decline rates between 0.085% and 0.143%. No segment stands out.
- **Home country**: false decline rates from 0% to 0.31% (Romania, 6 blocks out of 1,931
  payments) on small samples; PR-AUC is only measurable for the six largest countries
  (0.76–0.82, except the Netherlands at 0.44 on 31 frauds). No consistent pattern, but the
  small samples mean it should be monitored rather than declared fair.

## 6. Implications

1. Investigate a sample of blocked "legit" payments: most of them are unreported fraud.
2. Stealth fraud needs a different signal (customer reports, merchant feedback, behavioural
   biometrics), not a better model on the same features.
3. Feature set v2 could drop the ~20 features with no measurable importance; it would need
   its own validation (and a new test period).
4. Monitor card-present / contactless performance separately: it is the weakest segment and
   would be the first to hide a drift.
