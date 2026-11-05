# Feature engineering findings (Phase 5)

Feature set **v1.0.0**: 51 point-in-time features (catalog:
[`docs/feature_catalog.md`](../../docs/feature_catalog.md)), built by `fraud features build`
into `features.transaction_features`. Analysis: [`notebooks/03_feature_analysis.ipynb`](../../notebooks/03_feature_analysis.ipynb),
**train period only**, eventual labels. All numbers come from that run.

## 1. Correctness: can the features see the future?

| Check | Result |
|---|---|
| Hand-built scenario (ties in the same second, micro-charges, a fraud reported after a later payment, impossible travel, shared device) | 9 / 9 tests with exact expected values |
| **Time-travel test**: delete all transactions after 2025-04-01 and all labels reported after it, rebuild, compare earlier rows (`small` dataset) | **0 of 51** clean features change |
| Same test on the deliberately leaky features | **5 of 5** change, so the test can catch leakage |
| One feature row per `core` transaction; no negative counts, gaps or ages | pass |
| Feature SQL never reads the simulator's `sim` schema | pass (unit test) |

How "before" is defined: windows are `[ts − w, ts)` with `EXCLUDE GROUP`, so transactions in
the same second do not count; label features only use labels with `reported_at < ts`.

## 2. Build cost (full dataset, 2,852,459 rows)

| Step | Seconds |
|---|---|
| Base attributes | 32 |
| Window functions | 199 |
| LATERAL lookups (distinct counts, medians, label counts) | 244 |
| Assemble + indexes | 175 |
| Leaky variants | 107 |
| **Total** | **760** |

Acceptable for a nightly batch. The API (Phase 10) will compute the same definitions for one
transaction at a time, which needs a second implementation and a parity test.

## 3. Univariate screening (train)

Binned AUC replaces values by their bin's fraud rate (NULL is a bin), so it captures U-shapes
and "no history". Plain AUC only measures monotonic ranking and ignores NULLs.

| Feature | binned AUC | plain AUC | note |
|---|---|---|---|
| device_age_days | **0.832** | 0.135 | strongly *inverse*: new devices are risky |
| merchant_fraud_rate_90d_smoothed | 0.820 | 0.824 | labels known at ts only |
| card_seconds_since_prev | 0.808 | 0.221 | short gaps are risky |
| merchant_category | 0.796 | – | |
| cross_border | 0.789 | 0.789 | |
| card_txn_count_24h | 0.777 | 0.777 | |
| amount_eur | 0.774 | 0.703 | U-shape (micro-charges and large tickets) |
| card_unique_merchants_24h | 0.772 | 0.772 | |
| device_unique_cards_24h | 0.765 | 0.814 | plain AUC only on payments with a device |
| amount_to_cust_median_90d | 0.725 | 0.690 | |

Weakest: card_prior_fraud_reports (0.507), impossible_travel (0.509), account_age_days
(0.528), customer_age (0.534), is_micro_amount (0.533).

### Fraud rate by value

| Feature | values → fraud rate |
|---|---|
| card_txn_count_1h | 0 → 0.47% · 1 → 16.3% · 2 → **73.2%** · 3+ → 61.0% |
| card_micro_count_1h | 0 → 0.76% · 1 → 18.2% · 2+ → 56.2% |
| device_unique_cards_24h | 0 → 0.36% · 1 → 6.0% · 2 → 50.3% · 3+ → **92.3%** (n = 532) |
| device_unique_customers_30d | 1 → 0.74% · 2 → 13.4% · 3–5 → 90.6% · 6+ → 91.6% |
| device_age_days | < 1 h → 6.4% · 1 h–1 d → **41.5%** · 1–30 d → 3.4% · > 30 d → 0.28% |
| amount_to_cust_median_90d | < 2× → 0.36–0.57% · 2–5× → 1.22% · 5–10× → 2.29% · ≥ 10× → 4.80% |
| device_new_for_customer | true 5.68% (lift 6.7) vs false 0.89% |
| impossible_travel | true **32.5%** (n = 265, CI 27.1–38.3%) vs 0.85% |

## 4. Hypotheses (univariate evidence only; ablation comes in Phases 6–7)

- **H1, card velocity adds signal: supported.** One previous use in the hour already takes the
  rate from 0.47% to 16.3%.
- **H2, relative amount beats raw amount: NOT supported univariately.** Amount relative to the
  customer's median scores 0.725 vs 0.774 for the raw amount. It may still add information on
  top of the raw amount in a model; that is for the ablation to say.
- **H3, device-sharing features catch rings but legit sharing creates false positives:
  supported.** Two customers on a device: 13.4% fraud (mostly legit families); three or more:
  ~91%.

## 5. The label-based card feature is useless, and its leaky version looks great

`card_prior_fraud_reports` has **no signal** (0.88% vs 0.85%; binned AUC 0.507). A fraud
report arrives a median of ~15 days after the fraud, by which time the incident on that card
is over. (In this simulator, cards are not blocked after a report, so later payments are the
legitimate owner's.)

The naive version, which uses the labels of earlier transactions as soon as they exist in the
table, reaches **0.789**: the largest inflation of all the leaky features. It "works" because
it reads the labels of the other transactions in the same, still running, fraud incident. It is
the textbook example of label leakage (risk L3).

| Clean feature | binned AUC | Leaky variant | binned AUC | Inflation |
|---|---|---|---|---|
| card_prior_fraud_reports | 0.507 | leaky_card_prior_fraud_naive | 0.789 | **+0.282** |
| card_txn_count_24h | 0.777 | leaky_card_txn_count_24h_centered (±12 h) | 0.899 | +0.122 |
| merchant_fraud_rate_90d_smoothed | 0.820 | leaky_merchant_fraud_rate_alltime | 0.883 | +0.063 |
| amount_to_cust_median_90d | 0.725 | leaky_cust_amount_median_alltime | 0.774 | +0.049 |
| device_unique_customers_30d | 0.691 | leaky_device_unique_customers_alltime | 0.715 | +0.025 |

Every leaky variant looks better than its honest counterpart. How much this inflates a full
model's PR-AUC is measured in Phase 7 (hypothesis H8).

## 6. Warm-up bias: history starts on 2024-07-01 for everybody

| Month | merchant new | category new | new country (180 d) | no 90-day history |
|---|---|---|---|---|
| 2024-07 | 76.8% | 63.0% | 22.7% | 21.1% |
| 2024-08 | 55.1% | 33.8% | 3.7% | 1.5% |
| 2024-09 | 45.5% | 23.1% | 2.5% | 0.4% |
| 2025-01 | 30.4% | 10.0% | 2.2% | 0.3% |
| 2025-06 | 24.5% | 6.2% | 2.2% | 0.3% |

In the first months everything looks "new" because earlier history is not in the extract,
not because behaviour is new. Country and history features stabilise after ~1–2 months;
**merchant_new_for_customer is still drifting after 12 months**. Decision 016: train models
from 2024-10-01 (exclude a 90-day warm-up) and watch this feature's distribution in
monitoring.

## 7. Redundancy

- Real duplicates: `amount_eur` / `log_amount` (same ranking: keep the raw value for trees,
  the log for linear models); `distance_from_prev_km` / `implied_speed_kmh` /
  `country_changed` (distance > 0 exactly when the country changed).
- Card vs customer counts in 24 h (Spearman 0.947): most customers use one card.
- Several velocity pairs show Spearman ≈ 1.0 only because both are 0 for ~98% of rows; they
  differ in the tail. Not dropped on this basis.

## 8. Carried over from the EDA

The operating-system signal (binned AUC 0.719) is strong because of how the simulator picks
attacker devices. Treat its importance in later phases with suspicion.
