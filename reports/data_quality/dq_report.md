# Data Quality Report

- Run: `1` (succeeded, 567.4s, started 2026-10-21 09:46 UTC)
- Dataset: v1.0.1 `full` seed 42, hash `350d74bdc9880beb…`

## Transactions at a glance

```
Rows: 2,866,727
Duplicates: 6,865
Missing values: 572
Invalid values: 11,741
Broken relationships: 3,976
Inconsistencies: 2,030
Outliers (warnings, kept): 12,159
```

Counts are raw rows with at least one violation of that category.

## Rows per table

| table_name | raw_rows | excluded_rows | repaired_rows | warned_rows | loaded_rows | excluded_pct |
|---|---|---|---|---|---|---|
| customers | 30030 | 60 | 0 | 0 | 29970 | 0.200 |
| cards | 40164 | 58 | 0 | 0 | 40106 | 0.144 |
| merchants | 4000 | 0 | 0 | 0 | 4000 | 0.000 |
| devices | 55425 | 0 | 0 | 0 | 55425 | 0.000 |
| transactions | 2866727 | 14268 | 10853 | 12146 | 2852459 | 0.498 |
| fraud_labels | 26071 | 67 | 0 | 0 | 26004 | 0.257 |

## Rules

Actions: `reject` (not loaded), `deduplicate` (first copy loaded), `repair` (loaded after a deterministic fix), `warn` (loaded, only reported).

| rule_id | table_name | category | action | rows_flagged | description |
|---|---|---|---|---|---|
| CRD_PARENT_REJECTED | cards | broken_relationship | reject | 38 | Customer exists but was rejected by another rule |
| CRD_ORPHAN_CUSTOMER | cards | broken_relationship | reject | 20 | customer_id does not exist in the customer extract |
| CRD_DUP_EXACT | cards | duplicate | deduplicate | 0 | Row identical to an earlier row |
| CRD_DUP_ID_CONFLICT | cards | duplicate | reject | 0 | card_id appears with different contents |
| CRD_INVALID_VALUE | cards | invalid | reject | 0 | Missing id, unknown card type or unparseable date |
| CRD_ISSUED_BEFORE_ACCOUNT | cards | inconsistent | reject | 0 | Card issued before the account was opened |
| CUS_DUP_EXACT | customers | duplicate | deduplicate | 30 | Row identical to an earlier row |
| CUS_MISSING_COUNTRY | customers | missing | reject | 30 | Home country is null (needed for every geo feature) |
| CUS_DUP_ID_CONFLICT | customers | duplicate | reject | 0 | customer_id appears with different contents |
| CUS_INVALID_VALUE | customers | invalid | reject | 0 | Unknown country or segment, or unparseable date |
| CUS_MISSING_ID | customers | missing | reject | 0 | customer_id is null |
| CUS_UNDERAGE_ACCOUNT | customers | inconsistent | reject | 0 | Account opened before the customer turned 18 |
| DEV_DUP_EXACT | devices | duplicate | deduplicate | 0 | Row identical to an earlier row |
| DEV_DUP_ID_CONFLICT | devices | duplicate | reject | 0 | device_id appears with different contents |
| DEV_INVALID_VALUE | devices | invalid | reject | 0 | Missing id, unknown type or country, or unparseable date |
| LBL_PARENT_REJECTED | fraud_labels | broken_relationship | reject | 67 | Labelled transaction was rejected by another rule |
| LBL_DUP_EXACT | fraud_labels | duplicate | deduplicate | 0 | Row identical to an earlier row |
| LBL_DUP_ID_CONFLICT | fraud_labels | duplicate | reject | 0 | Two different labels for the same transaction |
| LBL_INVALID_VALUE | fraud_labels | invalid | reject | 0 | is_fraud not boolean, unknown source or unparseable report date |
| LBL_REPORTED_BEFORE_TXN | fraud_labels | inconsistent | reject | 0 | Label reported before the transaction happened |
| LBL_UNKNOWN_TRANSACTION | fraud_labels | broken_relationship | reject | 0 | Label for a transaction that is not in the extract |
| MER_DUP_EXACT | merchants | duplicate | deduplicate | 0 | Row identical to an earlier row |
| MER_DUP_ID_CONFLICT | merchants | duplicate | reject | 0 | merchant_id appears with different contents |
| MER_INVALID_VALUE | merchants | invalid | reject | 0 | Missing id, unknown category/channel/risk/country or bad date |
| TXN_SHARED_DEVICE | transactions | outlier | warn | 9329 | Device used by more than 50 distinct customers in the extract |
| TXN_SENTINEL_DEVICE | transactions | invalid | repair | 8581 | Placeholder device id ("unknown", "00000000"): device set to NULL |
| TXN_DUP_EXACT | transactions | duplicate | deduplicate | 5721 | Row identical to an earlier row (retried message) |
| TXN_PARENT_REJECTED | transactions | broken_relationship | reject | 3120 | Customer, card, merchant or device was rejected by another rule |
| TXN_AMOUNT_OUTLIER | transactions | outlier | warn | 2857 | Amount above the 99.9th percentile of its merchant category (kept: fraud lives in the tail) |
| TXN_COUNTRY_MISMATCH | transactions | inconsistent | repair | 1430 | Card-present payment whose country differs from the merchant country: terminal country = merchant country |
| TXN_INVALID_AMOUNT | transactions | invalid | reject | 1430 | Amount null, not numeric or <= 0 |
| TXN_INVALID_CURRENCY | transactions | invalid | reject | 1146 | Currency is not a known ISO code |
| TXN_DUP_ID_CONFLICT | transactions | duplicate | reject | 1144 | transaction_id appears with different amount/time: no version can be trusted |
| TXN_MISSING_CUSTOMER | transactions | missing | repair | 572 | customer_id is null: recovered from the card owner (a card has exactly one owner) |
| TXN_CARD_CUSTOMER_MISMATCH | transactions | broken_relationship | reject | 567 | Card belongs to a different customer |
| TXN_BEFORE_CARD_ISSUE | transactions | inconsistent | reject | 322 | Timestamp before the card was issued |
| TXN_INVALID_TIMESTAMP | transactions | invalid | reject | 300 | Timestamp null or unparseable |
| TXN_UNKNOWN_CUSTOMER | transactions | broken_relationship | reject | 291 | customer_id does not exist in the customer extract |
| TXN_UNKNOWN_MERCHANT | transactions | broken_relationship | reject | 290 | merchant_id does not exist in the merchant extract |
| TXN_CURRENCY_FORMAT | transactions | invalid | repair | 284 | Valid ISO currency in wrong case/whitespace: normalised |
| TXN_FUTURE_TIMESTAMP | transactions | inconsistent | reject | 278 | Timestamp after the extraction time |
| TXN_DEVICE_ON_CARD_PRESENT | transactions | inconsistent | reject | 0 | Card-present payment carrying a customer device |
| TXN_INVALID_VALUE | transactions | invalid | reject | 0 | Unknown payment method or country |
| TXN_MISSING_ID | transactions | missing | reject | 0 | transaction_id is null |
| TXN_NO_FX_RATE | transactions | broken_relationship | reject | 0 | No exchange rate for the currency on that day |
| TXN_UNKNOWN_CARD | transactions | broken_relationship | reject | 0 | card_id does not exist in the card extract |
| TXN_UNKNOWN_DEVICE | transactions | broken_relationship | reject | 0 | device_id does not exist in the device extract |

## Devices shared by many customers (warning)

This warning is how placeholder device ids are discovered: no real device is used by thousands of customers.

| device_id | distinct_customers | transactions |
|---|---|---|
| unknown | 5032 | 5995 |
| 00000000 | 2408 | 2586 |
| D4fabf497721b | 92 | 230 |
| Db6068a18a130 | 82 | 201 |
| D2d02285db0b5 | 76 | 196 |
| D434613f52f8a | 51 | 121 |

## Evaluation against injected defects (simulation only)

The simulator logged every defect it injected, so each rule can be measured. This is not possible with real data.

| table_name | defect_type | injected | detected | caught_by | recall |
|---|---|---|---|---|---|
| cards | orphan_reference | 20 | 20 | CRD_ORPHAN_CUSTOMER | 100.0% |
| customers | exact_duplicate | 30 | 30 | CUS_DUP_EXACT | 100.0% |
| customers | missing_country | 30 | 30 | CUS_MISSING_COUNTRY | 100.0% |
| transactions | country_mismatch | 1430 | 1430 | TXN_COUNTRY_MISMATCH | 100.0% |
| transactions | duplicate_id_conflict | 572 | 572 | TXN_DUP_ID_CONFLICT | 100.0% |
| transactions | exact_duplicate | 5721 | 5721 | TXN_DUP_EXACT | 100.0% |
| transactions | invalid_amount | 1430 | 1430 | TXN_INVALID_AMOUNT | 100.0% |
| transactions | invalid_currency | 1430 | 1430 | TXN_CURRENCY_FORMAT, TXN_INVALID_CURRENCY | 100.0% |
| transactions | invalid_timestamp | 858 | 858 | TXN_BEFORE_CARD_ISSUE, TXN_FUTURE_TIMESTAMP, TXN_INVALID_TIMESTAMP | 100.0% |
| transactions | missing_customer_id | 572 | 572 | TXN_MISSING_CUSTOMER | 100.0% |
| transactions | orphan_reference | 858 | 858 | TXN_CARD_CUSTOMER_MISMATCH, TXN_UNKNOWN_CUSTOMER, TXN_UNKNOWN_MERCHANT | 100.0% |
| transactions | sentinel_device | 8581 | 8581 | TXN_SENTINEL_DEVICE | 100.0% |

### Violations on records without an injected defect

Expected: warnings, and cascades (clean rows whose customer, card or transaction was rejected). Anything else would be a false alarm.

| rule_id | action | rows_without_injected_defect |
|---|---|---|
| TXN_PARENT_REJECTED | reject | 3094 |
| TXN_AMOUNT_OUTLIER | warn | 2836 |
| TXN_SHARED_DEVICE | warn | 747 |
| LBL_PARENT_REJECTED | reject | 67 |
| CRD_PARENT_REJECTED | reject | 38 |

### Fraud removed by cleaning

If excluded rows were much more (or less) fraudulent than kept rows, cleaning would bias the training data.

| lost_transactions | lost_true_fraud | lost_pct_of_all | lost_pct_of_fraud | fraud_rate_lost | fraud_rate_kept |
|---|---|---|---|---|---|
| 7975 | 76 | 0.2788 | 0.2670 | 0.953 | 0.995 |

### Are amount outliers errors or signal?

| is_outlier | transactions | true_fraud_pct |
|---|---|---|
| False | 2849602 | 0.989 |
| True | 2857 | 7.490 |

