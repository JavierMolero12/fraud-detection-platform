"""Feature catalog: the single list of model features, their meaning and their leakage control.

Used by the tests (expected columns), by the modelling pipeline (numeric vs categorical) and
to render docs/feature_catalog.md. Every history-based feature only uses transactions with
ts STRICTLY before the current one (ties excluded) and labels with reported_at < ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["numeric", "boolean", "categorical"]


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    kind: Kind
    definition: str
    why: str


# fmt: off
FEATURES: tuple[Feature, ...] = (
    # --- transaction --------------------------------------------------------------------------
    Feature("amount_eur", "transaction", "numeric", "Amount in EUR",
            "Fraud targets value; top amount decile is 3.0% fraud (EDA)"),
    Feature("log_amount", "transaction", "numeric", "ln(amount_eur)",
            "Amount is extremely skewed (skewness 59.8); log scale for linear models"),
    Feature("is_micro_amount", "transaction", "boolean", "amount_eur < micro_amount_eur",
            "Card testing uses micro-charges; < 5 EUR online is 2.86% fraud"),
    Feature("payment_method", "transaction", "categorical", "chip_pin, contactless, ecommerce, mobile_wallet",
            "Card-not-present payments carry most fraud"),
    Feature("is_card_present", "transaction", "boolean", "chip_pin or contactless",
            "Card-present fraud needs a physical clone"),
    Feature("merchant_category", "transaction", "categorical", "Merchant category",
            "Resellable goods (gift cards, luxury, electronics) attract fraud"),
    Feature("merchant_risk_level", "transaction", "categorical", "External rating fixed at onboarding",
            "Noisy view of merchant exposure (high 2.5% vs low 0.32%)"),
    Feature("local_hour", "transaction", "numeric", "Hour in the payment country's local time",
            "Fraud does not follow the customer's daily rhythm"),
    Feature("is_night", "transaction", "boolean", "local_hour within night_hours",
            "02-05h local reach 5.7-9.4% fraud (EDA)"),
    Feature("cross_border", "transaction", "boolean", "Payment country differs from home country",
            "Strongest EDA attribute (11.4% vs 0.34%); used instead of raw country"),
    Feature("merchant_foreign", "transaction", "boolean", "Merchant country differs from home country",
            "Distinguishes foreign online merchants from foreign presence"),
    # --- entity attributes at ts --------------------------------------------------------------
    Feature("card_type", "entity", "categorical", "debit, credit, prepaid, virtual",
            "Prepaid and virtual cards are targeted more"),
    Feature("customer_segment", "entity", "categorical", "student, standard, premium, small_business",
            "Weak on its own (V = 0.017); kept for interactions"),
    Feature("customer_age", "entity", "numeric", "Age in years at ts",
            "Weak on its own (AUC 0.49); kept for interactions"),
    Feature("account_age_days", "entity", "numeric", "Days since account opening at ts",
            "Cold-start accounts; card age dropped as redundant (Spearman 0.94)"),
    Feature("device_type", "entity", "categorical", "mobile, desktop, tablet, none (card present), unknown",
            "Fraud is far more frequent from desktops"),
    Feature("operating_system", "entity", "categorical", "Device OS; none / unknown when no device",
            "Strong in this data BY CONSTRUCTION of the simulator; interpret with care"),
    Feature("device_age_days", "entity", "numeric", "Days since the device was first seen (NULL without device)",
            "Attackers use devices never seen before"),
    # --- card velocity ------------------------------------------------------------------------
    Feature("card_txn_count_5m", "card_velocity", "numeric", "Card transactions in [ts-5m, ts)",
            "Card-testing bursts happen within minutes"),
    Feature("card_txn_count_1h", "card_velocity", "numeric", "Card transactions in [ts-1h, ts)",
            "One previous use in the hour: 16.4% fraud vs 0.43% (SQL preview)"),
    Feature("card_txn_count_24h", "card_velocity", "numeric", "Card transactions in [ts-24h, ts)",
            "Account takeover cashes out within hours"),
    Feature("card_txn_count_7d", "card_velocity", "numeric", "Card transactions in [ts-7d, ts)",
            "Baseline activity level of the card"),
    Feature("card_amount_sum_1h", "card_velocity", "numeric", "EUR spent with the card in [ts-1h, ts)",
            "Rapid draining of the card"),
    Feature("card_amount_sum_24h", "card_velocity", "numeric", "EUR spent with the card in [ts-24h, ts)",
            "Rapid draining of the card"),
    Feature("card_micro_count_1h", "card_velocity", "numeric", "Micro-charges on the card in [ts-1h, ts)",
            "Card testing precedes the large purchase"),
    Feature("card_seconds_since_prev", "card_velocity", "numeric", "Seconds since the card's previous use (NULL if none)",
            "Automated attacks are faster than people"),
    Feature("card_unique_merchants_24h", "card_velocity", "numeric", "Distinct merchants for the card in [ts-24h, ts)",
            "Fraudsters spread purchases across merchants"),
    Feature("card_unique_countries_24h", "card_velocity", "numeric", "Distinct payment countries for the card in [ts-24h, ts)",
            "A card cannot be in several countries the same day"),
    # --- customer behaviour -------------------------------------------------------------------
    Feature("cust_txn_count_24h", "customer_behaviour", "numeric", "Customer transactions (all cards) in [ts-24h, ts)",
            "Activity across all of the customer's cards"),
    Feature("cust_seconds_since_prev", "customer_behaviour", "numeric", "Seconds since the customer's previous transaction",
            "Unusually short gaps"),
    Feature("cust_txn_count_90d", "customer_behaviour", "numeric", "Customer transactions in [ts-90d, ts)",
            "History depth; profile features are unreliable without history"),
    Feature("amount_to_cust_median_90d", "customer_behaviour", "numeric", "amount / median amount of the customer in [ts-90d, ts)",
            "'High' is relative to each customer (hypothesis H2)"),
    Feature("amount_zscore_cust_90d", "customer_behaviour", "numeric", "z-score of log amount vs the customer's last 90 days",
            "Same idea on the log scale, accounts for spread"),
    Feature("merchant_new_for_customer", "customer_behaviour", "boolean", "No previous transaction of the customer at this merchant",
            "Fraud happens at merchants the victim never uses"),
    Feature("category_new_for_customer", "customer_behaviour", "boolean", "No previous transaction of the customer in this category",
            "Same at category level"),
    Feature("cust_unique_devices_30d", "customer_behaviour", "numeric", "Distinct devices of the customer in [ts-30d, ts)",
            "Account takeover adds devices"),
    # --- device -------------------------------------------------------------------------------
    Feature("device_new_for_customer", "device", "boolean", "First time this customer uses this device (NULL without device)",
            "Account takeover starts on a new device"),
    Feature("device_txn_count_1h", "device", "numeric", "Device transactions in [ts-1h, ts)",
            "Bots and fraud farms are busy"),
    Feature("device_unique_cards_24h", "device", "numeric", "Distinct cards used on the device in [ts-24h, ts)",
            "One device cashing out many stolen cards (fraud rings)"),
    Feature("device_unique_customers_30d", "device", "numeric", "Distinct customers on the device in [ts-30d, ts)",
            "Legit sharing is 1-2 people; rings are dozens"),
    # --- geography ----------------------------------------------------------------------------
    Feature("country_changed", "geo", "boolean", "Payment country differs from the customer's previous payment",
            "Location jumps"),
    Feature("new_country_for_customer_180d", "geo", "boolean", "No payment of the customer in this country in [ts-180d, ts)",
            "Unfamiliar country"),
    Feature("distance_from_prev_km", "geo", "numeric", "Distance between country centroids of this and the previous payment",
            "COUNTRY precision only: same country = 0 km"),
    Feature("implied_speed_kmh", "geo", "numeric", "distance_from_prev_km / hours since the previous payment",
            "Physical impossibility points to a cloned card"),
    Feature("impossible_travel", "geo", "boolean", "implied_speed_kmh > impossible_speed_kmh",
            "Card-present use abroad right after a use at home"),
    # --- labels known at ts -------------------------------------------------------------------
    Feature("card_prior_fraud_reports", "labels_as_of", "numeric", "Fraud reports on earlier transactions of the card, reported before ts",
            "A compromised card keeps being abused"),
    Feature("device_prior_fraud_reports", "labels_as_of", "numeric", "Fraud reports on earlier transactions from the device, reported before ts",
            "Devices used for fraud before"),
    Feature("merchant_fraud_reports_90d", "labels_as_of", "numeric", "Fraud reports for the merchant with reported_at in [ts-90d, ts)",
            "Merchants under attack or compromised"),
    Feature("merchant_txn_count_90d", "labels_as_of", "numeric", "Merchant transactions in [ts-90d, ts)",
            "Denominator of the merchant rate"),
    Feature("merchant_fraud_rate_90d_smoothed", "labels_as_of", "numeric", "Bayesian-smoothed rate of the two previous columns",
            "Stable even for small merchants"),
)
# fmt: on

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
NUMERIC = tuple(f.name for f in FEATURES if f.kind == "numeric")
BOOLEAN = tuple(f.name for f in FEATURES if f.kind == "boolean")
CATEGORICAL = tuple(f.name for f in FEATURES if f.kind == "categorical")

# Columns of features.transaction_features that are NOT model inputs.
KEY_COLUMNS = ("transaction_id", "ts", "customer_id", "card_id", "merchant_id", "device_id")
META_COLUMNS = ("feature_set_version",)

# Deliberately leaky variants (features.transaction_features_leaky), for the H8 experiment only.
LEAKY_FEATURES: dict[str, str] = {
    "leaky_card_txn_count_24h_centered": "card_txn_count_24h",
    "leaky_cust_amount_median_alltime": "amount_to_cust_median_90d",
    "leaky_device_unique_customers_alltime": "device_unique_customers_30d",
    "leaky_card_prior_fraud_naive": "card_prior_fraud_reports",
    "leaky_merchant_fraud_rate_alltime": "merchant_fraud_rate_90d_smoothed",
}


def render_markdown() -> str:
    lines = [
        "# Feature catalog",
        "",
        "Generated from `src/fraudplatform/features/catalog.py` (`fraud features catalog`).",
        "",
        "Rules for every history-based feature: only transactions with `ts` strictly before",
        'the current one (same-second transactions are not "before"), only labels with',
        "`reported_at < ts`, windows are `[ts - w, ts)`.",
        "",
    ]
    group = None
    for f in FEATURES:
        if f.group != group:
            group = f.group
            lines += [
                f"## {group.replace('_', ' ')}",
                "",
                "| feature | kind | definition | why it can predict fraud |",
                "|---|---|---|---|",
            ]
        lines.append(f"| `{f.name}` | {f.kind} | {f.definition} | {f.why} |")
        if f is FEATURES[-1] or FEATURES[FEATURES.index(f) + 1].group != group:
            lines.append("")
    lines += [
        "## Deliberately leaky variants (experiment only)",
        "",
        "| leaky feature | clean counterpart | what leaks |",
        "|---|---|---|",
        "| `leaky_card_txn_count_24h_centered` | `card_txn_count_24h` | window ±12h includes the future |",
        "| `leaky_cust_amount_median_alltime` | `amount_to_cust_median_90d` | median over the whole history, future included |",
        "| `leaky_device_unique_customers_alltime` | `device_unique_customers_30d` | global count over the whole extract |",
        "| `leaky_card_prior_fraud_naive` | `card_prior_fraud_reports` | earlier labels used before they were reported |",
        "| `leaky_merchant_fraud_rate_alltime` | `merchant_fraud_rate_90d_smoothed` | all labels, including the transaction's own |",
        "",
    ]
    return "\n".join(lines)
