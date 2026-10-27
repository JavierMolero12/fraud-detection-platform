# Model card: transaction fraud risk score v1

Version 1, final (Phase 13). Every number below comes from a recorded run; sources are linked.
The whole project is summarised in the [final report](final_report.md).

## Model details

| | |
|---|---|
| Model | scikit-learn `HistGradientBoostingClassifier` in a pipeline with ordinal encoding; native categorical splits and missing values |
| Hyperparameters | learning rate 0.021, 400 iterations, 31 leaves, ≥ 50 samples per leaf, L2 0.68, 80% of features per split (chosen by temporal CV, decision 020) |
| Calibration | Platt scaling, fitted on validation (practically the identity: slope 1.03, intercept 0.07) |
| Decision policy | per transaction, the action (approve / review / block) with the lowest expected cost, max 60 reviews per day (decision 021) |
| Artefacts | `data/processed/phase8/model.skops` (skops, explicit trusted types), [`reports/policy/policy_v1.json`](policy/policy_v1.json); hashes in [`frozen_bundle.sha256`](policy/frozen_bundle.sha256) |
| Lineage | dataset v1.0.1 `350d74bd…`, feature set v1.0.0, data quality run and feature build recorded in MLflow tags |
| Serving | real-time `POST /predict` and batch `fraud serve batch`, both from the same frozen bundle: model version `fcf88c67457e`, policy version `65a2111fa8c2` (first 12 hex digits of their SHA-256; they identify the files, so the same model saved again gets another version, see [reproducibility](../docs/reproducibility.md#5-checking-your-result)). Online and offline model inputs are identical bit for bit (decision 024); see [serving findings](serving/phase10_findings.md) |
| Environment | scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6 (trained on Windows, Python 3.13.1). The Docker images pin the same versions; on Linux the tree outputs are identical and 0.4% of probabilities differ by 1–2 ulps, with no decision changed (decision 027). `fraud serve verify-bundle` checks any new environment |
| Author | portfolio project; synthetic data only |

## Purpose

Score each card payment at authorisation time with a calibrated probability of being
fraudulent, and turn it into approve / review / block with the lowest expected cost under
explicit business assumptions ([`configs/costs.yaml`](../configs/costs.yaml)).

## Training data

- **Synthetic**, from the project's simulator (v1.0.1): 29,970 customers and 2.85M transactions
  after data quality, over 24 months, six fraud scenarios plus a drift scenario from March 2026,
  delayed and noisy labels. Nothing in it describes real people or real fraud rates.
- Trained on 2024-10-01 → 2025-06-30 (1,024,160 transactions, base rate 0.85%); the first 92
  days of data are excluded as feature warm-up (decision 016).
- Labels: fraud reported (chargeback, customer report, investigation) within 120 days.

## Features

50 point-in-time features (catalog: [`docs/feature_catalog.md`](../docs/feature_catalog.md)) in
seven families: transaction, entity attributes at the time of payment, card velocity, customer
behaviour, device, geography (country precision), and fraud labels already reported at that
time. 49 are inputs of this model (`log_amount` is only used by the linear challenger). History-based
features only see transactions strictly before the current one; label-based ones only see
labels reported before it. A time-travel test checks this.

No raw country is used as a feature (decision based on the EDA: the country effect is a
confounder of cross-border use).

## Metrics

| | Validation (Jul–Sep 2025) | Test (Oct–Dec 2025, evaluated once) |
|---|---|---|
| Base rate | 0.73% | 0.74% |
| PR-AUC (95% CI) | 0.792 | 0.752 (0.718–0.777) |
| ROC-AUC | 0.955 | 0.953 |
| Precision / recall at top 0.5% | 91.8% / 63.3% | 88.7% / 60.2% |
| Mean predicted vs observed | 0.73% / 0.73% | 0.73% / 0.74% |
| Policy cost (vs approve-all) | €62,925 (€946,616) | €64,098 (€946,956) |
| Fraud EUR stopped | 95.3% | 95.5% |
| False decline rate (by label) | 0.13% | 0.14% |

Baselines on validation: seven expert rules PR-AUC 0.256 (policy cost €520,946); amount alone
0.026. Sources: [`phase6_findings.md`](models/phase6_findings.md),
[`phase8_findings.md`](policy/phase8_findings.md).

**In simulated production** (January–June 2026, judged afterwards with every label known), with a
drift from March:
- the model still stops 76% of the new wallet fraud pattern;
- but it under-predicts fraud (observed 1.51% vs predicted 1.10% in the week of 9 April);
- weekly recall on reported fraud slides from 0.857 (January–February) to 0.778 (May–June).

Source: [`phase12_findings.md`](monitoring/phase12_findings.md).

## Intended use

- Real-time risk score at authorisation, combined with the expected-cost policy.
- Prioritising a manual review queue.
- As a reproducible reference implementation for the methods: point-in-time features,
  temporal validation, cost-based decisions, calibration, monitoring.

## Not intended use

- **Any decision about real customers.** The model was trained on simulated data.
- Credit, AML or identity decisions: different targets, different regulation.
- Blocking without a review or appeal path for the customer.
- Ranking customers or merchants (it scores transactions, not people).
- Use after the training data no longer resembles current traffic (see monitoring).

## Limitations

- **Synthetic data, high scores.** Fraud patterns in the simulator are cleaner than real ones;
  real-world PR-AUC would be far lower. The comparisons and methods transfer, the levels don't.
- **Stealth fraud is invisible** (2.8% stopped): fraud that copies the customer's usual
  merchants, device and amounts.
- **Card-present fraud is harder** (PR-AUC 0.667 vs 0.825 online; contactless 0.544).
- **Test PR-AUC is 0.04 below validation**, consistent with month-to-month variation or with
  model ageing; not yet separable.
- **Calibration weak spot**: in the 10–20% band the model predicts 14.3% where 21.7% is
  observed (207 transactions).
- **Label noise inflates measured errors**: on validation 72% of blocked "legit" payments were
  unreported fraud and 36% of approved "fraud" were legit disputes.
- Geography has country precision only (country centroids).
- 22 of 49 inputs have no measurable importance; the model is larger than it needs to be.

## Risks

- **Wrongful declines** hurt legitimate customers, concentrated on cross-border purchases of
  €100–500 (travellers): 102 of the 128 truly legit blocked payments on validation.
- **Segment fairness**: false decline rates by home country range 0–0.31% on small samples; no
  consistent pattern, but samples are too small to declare it fair. Age bands: 0.085–0.143%.
- **Cost assumptions**: if a review stops 45% of the fraud instead of 90%, the policy costs
  7.3% more than necessary. This assumption must be verified with the fraud team.
- **Feedback loops**: blocked payments never generate chargebacks, so future labels will be
  biased towards what the model lets through.
- **Leakage regressions**: a future feature that uses unreported labels or a future window
  would look excellent offline (Phase 7: 0.852) and fail in production (0.401).

## Monitoring (Phase 12)

Implemented in `fraud monitor run | replay | report` (decision 029). Each weekly run uses only
the data known on its date. Results:
[monitoring findings](monitoring/phase12_findings.md), [report](monitoring/monitoring_report.html).

**What each run computes:**
- **Input and prediction drift:** weekly PSI of the 49 inputs and of the score against the
  validation period; review and block rates.
- **Delay-adjusted calibration:** frauds reported so far vs the model's expectation for that
  delay. On the simulated drift it was the first reliable signal: an alert 67 days after the
  drift, with no false alarm before it.
- **Recall on mature cohorts** (about 48 days old): alerted 123 days after the drift.

**Retraining** is recommended when a label alert repeats in two consecutive runs; in the replay,
74 days after the drift. Input drift alone opens an investigation, not a retraining.

**Known weaknesses:**
- **PSI ≥ 0.10 against a fixed reference missed the drift**, whose step peaked at 0.075.
- **Age-like inputs drift by construction:** `device_age_days` alerted every week, and
  `merchant_new_for_customer` and `account_age_days` drift more slowly.
- **A rolling reference** (each week against the previous four, PSI ≥ 0.03) is the proposed
  fix. It was designed post hoc, with 0 false alarms in 22 drift-free weeks, but is not
  validated on a second drift yet.
