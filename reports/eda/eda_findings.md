# EDA findings (Phase 4)

Source: [`notebooks/02_eda.ipynb`](../../notebooks/02_eda.ipynb), run on dataset v1.0.1
(`350d74bd…`) after the data quality stage. **Train period only** (2024-07-01 → 2025-06-30,
1,344,750 transactions); validation, test and production periods were not analysed, so that
no modelling decision is informed by data used to evaluate it. Labels are the observed labels,
not the simulator's ground truth. Every number below comes from that run.

Figures: [`reports/figures/eda/`](../figures/eda/).

## 1. Base rate and imbalance

| | |
|---|---|
| Labelled fraud | 11,410 of 1,344,750 |
| Base rate | **0.848%** (95% Wilson CI 0.833% – 0.864%) |
| Volume | 3,684 transactions and 31.3 frauds per day; 1 fraud per 117 legit |

- "Approve everything" has **99.15% accuracy** and catches nothing. Accuracy will not be used.
- A random score has PR-AUC ≈ 0.848% (the base rate) and ROC-AUC = 0.5: these are the floors
  any model is compared against.
- **Base-rate effect:** a detector with 90% recall and only 1% false positive rate would have
  **43.5% precision** and raise ~65 alerts per day for ~28 real frauds. At 0.1% FPR precision
  is 88.5%. Review capacity, not just model quality, will shape the decision policy (Phase 8).

## 2. Labels arrive late

| Reported within | 7 d | 14 d | 30 d | 60 d | 90 d | 120 d |
|---|---|---|---|---|---|---|
| Share of fraud labels | 15.4% | 42.9% | 76.3% | 94.4% | 98.4% | 100% |

Median delay is similar across sources (chargeback 16.5 d, customer report 16.1 d,
investigation 14.6 d). 420 disputes were investigated and cleared as legit.

Implications: features built on past labels must only use labels with `reported_at < ts`
(risk L3); production performance can only be measured ~60–90 days late; any retraining
needs a label-maturity gap (risk L4).

## 3. Amount: skewed, non-monotonic, informative

- Skewness **59.8** on the raw scale, **0.10** on log10: amounts must be modelled on a log
  scale (or by trees, which are scale-invariant).
- Tukey's rule on the raw scale flags **10.6%** of transactions as outliers (fence €191):
  meaningless on a skewed variable. On log10 it flags **1.5%** (fence €891), and those
  transactions are **5.0% fraud** vs 0.78% for the rest. Same conclusion as the data quality
  report: extreme amounts are signal, not errors.
- The relationship is **U-shaped**: the lowest decile (≤ €8.35) is 1.04% fraud, the second
  decile 0.11%, then the rate rises to **3.0%** in the top decile (> €202). Fraud amounts have
  three modes: micro-charges (with a spike at exactly **€1.00**), and a main mode around €100–
  €300 vs ~€40 for legit.
- Univariate AUC of the raw amount is 0.703, which *understates* its value: AUC only measures
  a monotonic ranking.

## 4. Time

- **Local hour is the strongest time signal:** 02:00–05:00 local time reaches 5.7%–9.4% fraud
  (lift up to 11×) vs 0.5%–0.8% during the day. Its univariate AUC is 0.399, a second example
  of AUC hiding a non-monotonic (here U-shaped) effect.
- **Weekday carries no signal:** 0.83%–0.89% every day, overlapping intervals.
- **Monthly rate moves between 0.65% and 1.05%** inside the train year, highest in 2025-06.
  The process is not stationary even before the designed drift, which is why validation must
  be temporal.

## 5. Categorical attributes

Association with fraud (Cramér's V, bias-corrected; all p < 1e-7 after Benjamini-Hochberg):

| Attribute | V | Highlights |
|---|---|---|
| cross_border | **0.253** | 11.4% vs 0.34% domestic (lift 13.4) |
| operating_system | 0.179 | linux 26.9%, windows 6.0%, iOS 0.44% |
| txn_country | 0.141 | confounded, see below |
| merchant_category | 0.133 | gift cards 7.2%, luxury 4.3%, electronics 3.3%; home 0.11% |
| device_type | 0.123 | desktop 6.2% vs mobile 0.75% |
| merchant_risk_level | 0.079 | high 2.5%, medium 1.15%, low 0.32% |
| payment_method | 0.056 | e-commerce 1.43%, contactless 0.25% |
| merchant_channel | 0.049 | |
| card_type | 0.036 | virtual 2.4%, prepaid 1.35%, debit 0.66% |
| customer_segment | 0.017 | student 1.06% vs premium 0.64%: weak |
| home_country | 0.006 | no signal |

**The country effect is a confounder.** Transaction countries such as BR (10.2%) or TR (7.8%)
look risky, but split by domestic vs cross-border, **domestic fraud is 0.14%–0.48% in all 14
countries**. The differences come from how many payments in each country are cross-border for
the customer. Decision for feature design: encode the *mismatch* (cross-border, country change,
distance), not the country itself. Besides being the actual mechanism, it avoids a model that
penalises legitimate residents of particular countries.

Card presence and cross-border combined:

| | fraud rate | lift |
|---|---|---|
| card not present · cross-border | 13.4% | 15.8 |
| card present · cross-border | 7.7% | 9.1 |
| card not present · domestic | 0.51% | 0.6 |
| card present · domestic | 0.11% | 0.13 |

**Interactions exist.** Payments under €5 are 2.86% fraud in e-commerce but ≤ 0.04% in any
card-present channel; mobile-wallet payments ≥ €500 are 5.75% fraud vs 0.31% for contactless
cards at the same amount. A linear model without interaction terms cannot express this
(hypothesis H4).

## 6. Numeric attributes

| | univariate AUC | median legit | median fraud |
|---|---|---|---|
| amount_eur | 0.703 | €40.78 | €122.80 |
| customer_age | 0.491 | 43 | 43 |
| account_age_days | 0.507 | 1,426 | 1,462 |
| card_age_days | 0.473 | 1,328 | 1,203 |

Customer age, account age and card age carry almost no signal on their own. New accounts are
only slightly riskier (8–30 days: 1.10%, CI 0.92%–1.32%, vs 0.86% for accounts older than 3
years). **This is weaker than expected:** hypothesis H11 (cold start) is not supported by static
attributes; whether behaviour-based features do better is a Phase 5 question.

## 7. Redundancy between attributes

- payment_method ↔ merchant_channel: V = 0.93; merchant_channel ↔ device_type: V = 0.91;
  txn_country ↔ home_country: V = 0.89.
- account_age ↔ card_age: Spearman 0.944.

These pairs carry overlapping information: keep one of each, or expect unstable coefficients
in logistic regression.

## 8. What is artificial (read before trusting the numbers)

These signals are mostly a direct consequence of simulator parameters I chose, so their
*strength* says little about real fraud:

- **Operating system** (linux 26.9%): follows from the attacker device mix in
  `reference.FRAUD_DEVICES`. Real user agents are spoofable and the signal is weaker. Models
  will lean on it; feature importances must be read with this in mind.
- **Night hours**: remote fraud is spread evenly over the day in the simulator while legit
  activity follows a daily curve, so the night rate is high by construction.
- **Cross-border rates by destination** (e.g. BR 38% vs PT 2.9% of cross-border payments)
  follow the synthetic `attacker_weight` per country.
- **Exactly €1.00 charges** come from the card-testing scenario.
- **Data artefact:** 93 transactions (0.003%) sit on the first and last second of the
  simulated period because the generator clips timestamps to the period. Negligible here;
  documented in decision 014.

## 9. Implications for Phase 5 (feature engineering)

1. Log-transform amounts; add amount relative to the customer's own history (H2), a
   micro-amount flag, and keep extremes (no winsorising of the target signal).
2. Encode geography as mismatch features (cross-border, country change, distance, new
   country), not raw country.
3. Local hour as a cyclical or bucketed feature; drop weekday unless behavioural features
   change the picture.
4. Device features: new-device-for-customer and device sharing (point-in-time), not only the
   static OS.
5. Keep one of each redundant pair (payment method vs channel, account vs card age).
6. Label-based features must respect `reported_at < ts`; with 42.9% of labels known after two
   weeks, such features will be weaker online than a naive offline computation suggests.
