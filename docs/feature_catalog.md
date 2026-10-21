# Feature catalog

Generated from `src/fraudplatform/features/catalog.py` (`fraud features catalog`).

Rules for every history-based feature: only transactions with `ts` strictly before
the current one (same-second transactions are not "before"), only labels with
`reported_at < ts`, windows are `[ts - w, ts)`.

## transaction

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `amount_eur` | numeric | Amount in EUR | Fraud targets value; top amount decile is 3.0% fraud (EDA) |
| `log_amount` | numeric | ln(amount_eur) | Amount is extremely skewed (skewness 59.8); log scale for linear models |
| `is_micro_amount` | boolean | amount_eur < micro_amount_eur | Card testing uses micro-charges; < 5 EUR online is 2.86% fraud |
| `payment_method` | categorical | chip_pin, contactless, ecommerce, mobile_wallet | Card-not-present payments carry most fraud |
| `is_card_present` | boolean | chip_pin or contactless | Card-present fraud needs a physical clone |
| `merchant_category` | categorical | Merchant category | Resellable goods (gift cards, luxury, electronics) attract fraud |
| `merchant_risk_level` | categorical | External rating fixed at onboarding | Noisy view of merchant exposure (high 2.5% vs low 0.32%) |
| `local_hour` | numeric | Hour in the payment country's local time | Fraud does not follow the customer's daily rhythm |
| `is_night` | boolean | local_hour within night_hours | 02-05h local reach 5.7-9.4% fraud (EDA) |
| `cross_border` | boolean | Payment country differs from home country | Strongest EDA attribute (11.4% vs 0.34%); used instead of raw country |
| `merchant_foreign` | boolean | Merchant country differs from home country | Distinguishes foreign online merchants from foreign presence |

## entity

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `card_type` | categorical | debit, credit, prepaid, virtual | Prepaid and virtual cards are targeted more |
| `customer_segment` | categorical | student, standard, premium, small_business | Weak on its own (V = 0.017); kept for interactions |
| `customer_age` | numeric | Age in years at ts | Weak on its own (AUC 0.49); kept for interactions |
| `account_age_days` | numeric | Days since account opening at ts | Cold-start accounts; card age dropped as redundant (Spearman 0.94) |
| `device_type` | categorical | mobile, desktop, tablet, none (card present), unknown | Fraud is far more frequent from desktops |
| `operating_system` | categorical | Device OS; none / unknown when no device | Strong in this data BY CONSTRUCTION of the simulator; interpret with care |
| `device_age_days` | numeric | Days since the device was first seen (NULL without device) | Attackers use devices never seen before |

## card velocity

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `card_txn_count_5m` | numeric | Card transactions in [ts-5m, ts) | Card-testing bursts happen within minutes |
| `card_txn_count_1h` | numeric | Card transactions in [ts-1h, ts) | One previous use in the hour: 16.4% fraud vs 0.43% (SQL preview) |
| `card_txn_count_24h` | numeric | Card transactions in [ts-24h, ts) | Account takeover cashes out within hours |
| `card_txn_count_7d` | numeric | Card transactions in [ts-7d, ts) | Baseline activity level of the card |
| `card_amount_sum_1h` | numeric | EUR spent with the card in [ts-1h, ts) | Rapid draining of the card |
| `card_amount_sum_24h` | numeric | EUR spent with the card in [ts-24h, ts) | Rapid draining of the card |
| `card_micro_count_1h` | numeric | Micro-charges on the card in [ts-1h, ts) | Card testing precedes the large purchase |
| `card_seconds_since_prev` | numeric | Seconds since the card's previous use (NULL if none) | Automated attacks are faster than people |
| `card_unique_merchants_24h` | numeric | Distinct merchants for the card in [ts-24h, ts) | Fraudsters spread purchases across merchants |
| `card_unique_countries_24h` | numeric | Distinct payment countries for the card in [ts-24h, ts) | A card cannot be in several countries the same day |

## customer behaviour

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `cust_txn_count_24h` | numeric | Customer transactions (all cards) in [ts-24h, ts) | Activity across all of the customer's cards |
| `cust_seconds_since_prev` | numeric | Seconds since the customer's previous transaction | Unusually short gaps |
| `cust_txn_count_90d` | numeric | Customer transactions in [ts-90d, ts) | History depth; profile features are unreliable without history |
| `amount_to_cust_median_90d` | numeric | amount / median amount of the customer in [ts-90d, ts) | 'High' is relative to each customer (hypothesis H2) |
| `amount_zscore_cust_90d` | numeric | z-score of log amount vs the customer's last 90 days | Same idea on the log scale, accounts for spread |
| `merchant_new_for_customer` | boolean | No previous transaction of the customer at this merchant | Fraud happens at merchants the victim never uses |
| `category_new_for_customer` | boolean | No previous transaction of the customer in this category | Same at category level |
| `cust_unique_devices_30d` | numeric | Distinct devices of the customer in [ts-30d, ts) | Account takeover adds devices |

## device

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `device_new_for_customer` | boolean | First time this customer uses this device (NULL without device) | Account takeover starts on a new device |
| `device_txn_count_1h` | numeric | Device transactions in [ts-1h, ts) | Bots and fraud farms are busy |
| `device_unique_cards_24h` | numeric | Distinct cards used on the device in [ts-24h, ts) | One device cashing out many stolen cards (fraud rings) |
| `device_unique_customers_30d` | numeric | Distinct customers on the device in [ts-30d, ts) | Legit sharing is 1-2 people; rings are dozens |

## geo

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `country_changed` | boolean | Payment country differs from the customer's previous payment | Location jumps |
| `new_country_for_customer_180d` | boolean | No payment of the customer in this country in [ts-180d, ts) | Unfamiliar country |
| `distance_from_prev_km` | numeric | Distance between country centroids of this and the previous payment | COUNTRY precision only: same country = 0 km |
| `implied_speed_kmh` | numeric | distance_from_prev_km / hours since the previous payment | Physical impossibility points to a cloned card |
| `impossible_travel` | boolean | implied_speed_kmh > impossible_speed_kmh | Card-present use abroad right after a use at home |

## labels as of

| feature | kind | definition | why it can predict fraud |
|---|---|---|---|
| `card_prior_fraud_reports` | numeric | Fraud reports on earlier transactions of the card, reported before ts | A compromised card keeps being abused |
| `device_prior_fraud_reports` | numeric | Fraud reports on earlier transactions from the device, reported before ts | Devices used for fraud before |
| `merchant_fraud_reports_90d` | numeric | Fraud reports for the merchant with reported_at in [ts-90d, ts) | Merchants under attack or compromised |
| `merchant_txn_count_90d` | numeric | Merchant transactions in [ts-90d, ts) | Denominator of the merchant rate |
| `merchant_fraud_rate_90d_smoothed` | numeric | Bayesian-smoothed rate of the two previous columns | Stable even for small merchants |

## Deliberately leaky variants (experiment only)

| leaky feature | clean counterpart | what leaks |
|---|---|---|
| `leaky_card_txn_count_24h_centered` | `card_txn_count_24h` | window ±12h includes the future |
| `leaky_cust_amount_median_alltime` | `amount_to_cust_median_90d` | median over the whole history, future included |
| `leaky_device_unique_customers_alltime` | `device_unique_customers_30d` | global count over the whole extract |
| `leaky_card_prior_fraud_naive` | `card_prior_fraud_reports` | earlier labels used before they were reported |
| `leaky_merchant_fraud_rate_alltime` | `merchant_fraud_rate_90d_smoothed` | all labels, including the transaction's own |
