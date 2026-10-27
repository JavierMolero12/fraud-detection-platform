# Final report: transaction fraud detection and risk scoring

An end-to-end fraud scoring system on 2.85M simulated card payments: data quality, point-in-time
features, models validated in time, a cost-based decision policy, a real-time API, Docker and CI,
and monitoring with delayed labels. Every number below comes from a recorded run; each section
links to the report it comes from.

**The data is synthetic.** The project demonstrates method (leakage control, temporal
validation, cost-based decisions, calibration, delayed labels, drift), not discoveries about real
fraud. Where the simulator makes a result true by construction, it is said.

## 1. Results at a glance

| | Result |
|---|---|
| Ranking (PR-AUC), validation → the single test evaluation | 0.792 → **0.752** (base rate 0.73%; expert rules 0.256) |
| Cost of the deployed policy on test | **€64,098** vs €946,956 approving everything (−93.2%) and €482,886 for expert rules |
| Fraud EUR stopped on test | **95.5%**, with a 0.14% measured false decline rate |
| Real false decline rate (validation, label noise removed with the simulator's truth) | **0.036%**, 3.6× lower than measured |
| Leakage, offline vs production | a leaky model scores 0.852 offline and **0.401** in production |
| API latency, one at a time (Docker, 4 workers) | **44.6 ms** p50 on the client; 74 requests/s with 8 clients |
| Online vs offline model input | identical bit for bit (0 differences in 76,950 values) |
| Drift detection (injected on 1 March 2026) | labels alerted after **67 days**, retraining recommended after 74; input PSI with the conventional threshold never alerted |

## 2. Setup

- **Data** ([simulation report](simulation/v1.0.1-full.md)):
  - 24 months (July 2024 – June 2026), 30,030 customers, 4,000 merchants, 2.87M raw transactions;
  - fraud from seven scenarios, 0.995% true fraud;
  - labels with delay and noise: 0.879% labelled, 15.7% of true fraud never reported, 1,157
    legitimate payments disputed as fraud;
  - a drift from 1 March 2026.
- **Splits** (decision 005):
  - train July 2024 – June 2025, without a 92-day warm-up (decision 016);
  - validation July – September 2025, for calibration and policy;
  - test October – December 2025, **evaluated once** (decision 022);
  - simulated production January – June 2026, for monitoring.
- **Cost** ([`configs/costs.yaml`](../configs/costs.yaml), all assumptions):
  - an approved fraud costs its amount plus a €15 chargeback fee;
  - a review costs €4 and stops 90% of the fraud it sees;
  - a wrongful block costs 2% of the amount plus €10;
  - at most 60 reviews per day.

## 3. Data quality ([report](data_quality/dq_report.md))

- **47 SQL rules**, each with an explicit action: reject, deduplicate, repair or warn
  (decision 011).
- **Every defect type caught.** The simulator logged every injected defect, and each type was
  caught with **100% recall**: 5,721 exact duplicates, 572 conflicting ids, 1,430 invalid
  amounts, 8,581 sentinel devices, and so on.
- **The rows not loaded** (14,268 of 2,866,727 transactions, 0.50%) are defects plus cascades:
  3,094 valid transactions of the 30 customers without a country, whose fraud rate (0.953%)
  matches the rest (0.995%) (decision 012).
- **Outliers were kept**: they are 7.49% real fraud.
- **The rules found a simulator bug**: 448 customers opening an account a few hours before turning
  18. It was fixed in v1.0.1 (decision 013).

## 4. What the data looks like ([EDA](eda/eda_findings.md), train period only)

- **Base rate 0.848%** (Wilson CI 0.833–0.864%). Approving everything is 99.15% accurate, so
  accuracy is never used.
- **Labels arrive late:** 76.3% within 30 days, 94.4% within 60, all within 120. Every
  label-based feature and the monitoring must respect `reported_at`.
- **Strong signals:** local hour (02:00–05:00: 5.7–9.4% fraud), cross-border (11.4% vs 0.34%),
  merchant category.
- **Weak ones:** customer age, account age (H11, cold start, not supported).
- **What the simulator makes artificially strong** (operating system, exact €1.00 charges) is
  listed so it is not trusted.

## 5. Features and leakage ([findings](features/feature_findings.md), [catalogue](../docs/feature_catalog.md))

- **50 point-in-time features in SQL:** history strictly before `ts`, ties excluded, labels only
  if `reported_at < ts`. Building them takes 760 s.
- **Can they see the future? No.** A time-travel test deletes everything after a date and
  rebuilds; **0 of 50** clean features change, while **5 of 5** deliberately leaky features do.
- **Leakage measured, not just described (H8, [Phase 7](models/phase7_findings.md)).** The same
  model with five leaky features scores 0.852 PR-AUC on validation. With the values available at
  decision time it scores **0.401**, half the honest model. No offline metric on the leaky
  extract would reveal it.
- **The label-based card feature has no signal** (binned AUC 0.507), but its naive version reaches
  0.789 by reading the labels of the same, still running, fraud incident.

## 6. Models ([Phase 6](models/phase6_findings.md), [Phase 7](models/phase7_findings.md))

Validation (359,220 payments), alert budget = top 0.5% (~20 a day):

| | PR-AUC | Precision | Recall | Fraud EUR caught |
|---|---|---|---|---|
| Expert rules (7) | 0.256 | 46.3% | 31.9% | 31.8% |
| Logistic regression | 0.766 | 88.4% | 60.9% | 79.8% |
| Random forest | 0.765 | 89.0% | 61.3% | 75.5% |
| **Gradient boosting (tuned)** | **0.794** | **91.7%** | **63.2%** | **85.5%** |

- **Gradient boosting wins, but modestly.** The paired day-block bootstrap gives +0.024 over
  logistic regression (CI +0.015 to +0.033); logistic regression stays as the interpretable
  alternative (decision 010).
- **Shuffled CV overstates, but not by much (H7).** It overstates by +0.018 against temporal CV,
  smaller than the textbook warning, because feature leakage was removed first.
- **Tuning changed little.** It found +0.004 in CV, not significant on validation (+0.004, CI
  −0.001 to +0.010). The month-to-month spread (0.768–0.822) is larger than any tuning gain
  (decision 020).
- **Row order mattered more than it should.** Phase 7 was re-run in Phase 13 with the rows in a
  fixed order (section 14). Its unordered first run differed by up to 0.003, as much as the
  tuning gain. The bundle trains the same configuration with the learning rate and L2 rounded,
  as written in the config: 0.792 on validation, not 0.794.
- **Class weights hurt.** No gain in PR-AUC, and mean probabilities went from 0.0072 to
  0.068–0.123, so the models are trained without weights or resampling (decision 018).

## 7. Decisions, calibration and the one test evaluation ([Phase 8](policy/phase8_findings.md))

- **The model was already calibrated.** On held-out data, mean predicted 0.75% vs observed
  0.735%. Platt scaling ties with no calibration (slope 1.03), isotonic is worse; H5 is not
  supported for this model.
- **The decision depends on the amount (H6).** A €1,000 payment deserves a review at a 0.55% fraud
  probability, a €5 one not until 26%. The usual 0.5 threshold costs **49% more** than the
  optimum.
- **Per-transaction expected-cost policy (P2), deployed.** It costs €62,925 on validation vs
  €63,742 for the best global thresholds (P1). It reviews more and blocks fewer legitimate
  customers (455 vs 527). The capacity was never reached (24 of 60 reviews on the busiest day).
- **Sensitivity.** The policy is robust (≤ 1.2% regret) to every assumption but one: how much
  fraud a review really stops (+7.3% regret if it is half). That is the number to verify with a
  fraud team.
- **Test, evaluated once after freezing the policy** (git commits `32ce00c` then `3b95790`).
  - PR-AUC dropped from 0.792 to **0.752** (CI 0.718–0.777), beyond the validation interval:
    month-to-month variation or model ageing, not separable with this data.
  - The business result held: cost reduction 93.2% (93.4% on validation), fraud EUR stopped 95.5%,
    calibration 0.73% predicted vs 0.74% observed.

## 8. Where the model is wrong, and why ([Phase 9](analysis/phase9_findings.md))

- **Most measured errors are label noise.**
  - Of 455 blocked "legit" payments, 327 are fraud that was never reported. All 198 blocked
    micro-charges are unreported card testing.
  - Of 429 approved "fraud" payments, 153 are legitimate purchases disputed later.
  - The real false decline rate is 0.036%, not 0.128%. In a real deployment, a sample of blocked
    payments should be investigated before trusting the false positive rate.
- **Stealth fraud is invisible (H10, by construction).** Only 2.8% is stopped, against 91–100% of
  every other scenario. With friendly fraud, it explains 87% of the false negatives.
- **What drives the model:** device age, cross-border, merchant category, device sharing.
  - 22 of 49 inputs add nothing measurable; candidates for a future feature set.
  - Explanations use SHAP's permutation explainer: the tree explainer is silently wrong for this
    model's categorical splits, by up to 4.6 log-odds (decision 023).
- **Weak segments:** card present (PR-AUC 0.667), contactless (0.544), domestic (0.602), €5–20
  payments (0.140, partly by design: a €4 review is not spent on a €10 payment).

## 9. Serving ([Phase 10](serving/phase10_findings.md))

- **The same inputs online as in training.** A FastAPI service scores one payment at
  authorisation. The online feature SQL mirrors the offline build and is read through the same
  parser.
- **Exact parity was necessary, not pedantry.** Parity "within 1e-9" had passed while the API
  disagreed with batch scoring on 1 in 150 probabilities: tree thresholds sit exactly on repeated
  training values, so one ulp changes the branch (decision 024).
- **Production details:**
  - idempotent retries;
  - a review capacity that holds under concurrency (10 simultaneous requests, 3 slots → exactly 3
    reviews);
  - input validation (422);
  - every decision logged with its inputs.
- **Batch scoring:** 760,511 production payments in 47 s.
- **Latency:** 41 ms per prediction, 63% of it the feature query. A single sub-query (a merchant's
  fraud reports in 90 days) takes 13 of the query's 17 ms.

## 10. Packaging and CI ([Phase 11](deployment/phase11_findings.md))

- **Two images:**
  - `app` (682 MB) for the API;
  - `train` (944 MB) for every command.

  Both pin the libraries the model was trained with. The model is mounted, never baked in.
- **Linux reproduces Windows almost bit for bit.** The tree outputs are identical on 359,220
  payments; 1,446 probabilities differ by 1–2 ulps, from the operating system's `exp()`; no
  decision changes.
- **`uvicorn --workers` added ~40 ms to every request.** Its listening socket lost `TCP_NODELAY`.
  Fixed with a launcher, and guarded in CI (decision 027).
- **CI tests the images end to end.** It runs the pipeline in containers on a tiny dataset and
  checks the API container against batch scoring (decision 028).
- **Coverage:** 74% overall (Linux CI); 92–100% for the logic modules.

## 11. Monitoring with delayed labels ([Phase 12](monitoring/phase12_findings.md))

42 weekly runs, each with only the data known that day, against a drift injected on 1 March 2026:
legitimate payments move online and to wallets, and a new wallet fraud pattern ramps up.

| Rule (fixed before the replay) | False alarms before the drift | First alert after it |
|---|---|---|
| Input PSI ≥ 0.25 | 8 (one input that drifts by construction) | never caused by the drift |
| Block rate ×1.5 | 1 (a real January fraud peak) | +46 days |
| **Delay-adjusted calibration** (frauds reported vs expected so far) | **0** | **+67 days** |
| **Retraining recommended** | 0 | **+74 days** |
| Recall on mature cohorts | 0 | +123 days |

- **What happened in reality:** the model still catches 76% of the new fraud, but under-predicts
  it, and recall slides from 0.86 to 0.78.
- **Input drift was visible but below the threshold.** A step from about 0 to 0.06–0.075 PSI in
  the first week of March, under the conventional 0.10 threshold (10× the noise here).
- **A post-hoc alternative** (a rolling reference) would have alerted after 11 days with 0 false
  alarms in 22 drift-free weeks. It is a hypothesis, not a result, and what it detects is the
  change in legitimate behaviour, not the new fraud.
- **H9 is not supported as stated** (decision 030).

## 12. Hypotheses

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Card velocity adds signal | Supported | one earlier use in the hour: 0.47% → 16.3% fraud; the family costs 0.045 PR-AUC when shuffled |
| H2 | Relative amount beats raw amount | Not supported (univariate) | binned AUC 0.725 vs 0.774; no ablation was run |
| H3 | Device sharing catches rings, legit sharing creates false positives | Partly | rings 100% stopped; 2 customers per device: 13.4% fraud, mostly families; real FPs were mostly travellers |
| H4 | Gradient boosting beats logistic regression | Supported, modestly | +0.024 PR-AUC, paired CI +0.015 to +0.033 |
| H5 | Trees are badly calibrated; Platt/isotonic fix it | Not supported | without class weights the model was already calibrated; isotonic was worse |
| H6 | The optimal threshold is far below 0.5 and depends on the amount | Supported | review from 0.55% at €1,000, 26% at €5; 0.5 costs 49% more |
| H7 | Shuffled CV overstates | Supported, modestly | +0.018 PR-AUC vs temporal CV |
| H8 | Leaky features inflate offline metrics | Supported, strongly | 0.852 offline, 0.401 in production |
| H9 | PSI detects drift before the labels | Not supported with the rules fixed in advance | labels +67 days; PSI never crossed 0.10 |
| H10 | Stealth fraud goes undetected | Supported (by construction) | 2.8% stopped |
| H11 | New accounts perform worse (cold start) | Not supported | 1.10% vs 0.86% fraud for new vs old accounts; no segment stands out |

## 13. Limitations

- **Synthetic data.** The model may have learned the simulator's rules; results on real payments
  would differ. The optional external check on a public dataset (Sparkov) was not done.
- **One seed, one drift scenario.** The monitoring conclusions and the post-hoc rules need a
  second scenario.
- **No retraining was run.** The monitoring says *when* (14 May 2026); whether a retrained model
  recovers recall is untested.
- **Laptop measurements.** Latency was measured on a shared laptop and Docker Desktop; tails moved
  by a factor of 3.6 between runs.
- **Production gaps:** no authentication or rate limiting on the API; no scheduler for the weekly
  monitoring; images not published or scanned.
- **The test period was used once.** Any future model change needs a new, untouched test
  period.

## 14. Reproducing it

[`docs/reproducibility.md`](../docs/reproducibility.md) gives the commands and timings. It also
records the result of a from-scratch reproduction in a clean clone, in Docker, with an empty
database: 77 minutes of computing.

- **It reproduced the results.** Every table is identical row for row. The model is the same 400
  trees bit for bit, and the batch decisions on 760,511 payments are identical. The monitoring
  alerts match, and the API gives the README's probability exactly. The only differences are a
  few probabilities 1–4 ulps apart, from Linux instead of Windows.
- **It found four problems, all fixed.**
  - A feature build that stalled for 13 hours for lack of memory.
  - Runs in containers without lineage.
  - Tuning killed for lack of memory.
  - Phase 7 loading its rows in an order that changed on every run (section 6).
