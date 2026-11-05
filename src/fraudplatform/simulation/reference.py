"""Static reference data for the simulator: countries, currencies, categories and segments.

All weights here are SYNTHETIC parameters. In particular, ``attacker_weight`` controls where
cross-border fraud originates in the simulation; it is not a statement about real countries.
The main geographic risk mechanism is *mismatch* (cross-border use, impossible travel), not
nationality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    region: str
    currency: str
    lat: float
    lon: float
    utc_offset: int  # fixed offset, DST ignored (documented simplification)
    customer_weight: float  # share of the customer base
    travel_weight: float  # attractiveness as a travel destination
    online_weight: float  # where online merchants are registered
    attacker_weight: float  # synthetic: origin of cross-border fraud


COUNTRIES: tuple[Country, ...] = (
    Country("ES", "Spain", "EU", "EUR", 40.2, -3.7, 1, 0.700, 0.10, 0.35, 0.05),
    Country("PT", "Portugal", "EU", "EUR", 39.6, -8.0, 0, 0.060, 0.14, 0.02, 0.03),
    Country("FR", "France", "EU", "EUR", 46.6, 2.4, 1, 0.060, 0.16, 0.06, 0.06),
    Country("DE", "Germany", "EU", "EUR", 51.1, 10.4, 1, 0.040, 0.08, 0.06, 0.05),
    Country("IT", "Italy", "EU", "EUR", 42.8, 12.5, 1, 0.050, 0.14, 0.03, 0.05),
    Country("NL", "Netherlands", "EU", "EUR", 52.2, 5.3, 1, 0.020, 0.05, 0.10, 0.08),
    Country("IE", "Ireland", "EU", "EUR", 53.2, -8.2, 0, 0.010, 0.03, 0.14, 0.03),
    Country("GB", "United Kingdom", "Europe non-EU", "GBP", 54.0, -2.0, 0, 0.020, 0.10, 0.08, 0.10),
    Country("US", "United States", "Americas", "USD", 39.8, -98.6, -5, 0.010, 0.07, 0.12, 0.12),
    Country("MX", "Mexico", "Americas", "MXN", 23.6, -102.5, -6, 0.015, 0.04, 0.01, 0.08),
    Country("BR", "Brazil", "Americas", "BRL", -14.2, -51.9, -3, 0.005, 0.02, 0.01, 0.12),
    Country("MA", "Morocco", "Africa", "MAD", 31.8, -7.1, 1, 0.005, 0.05, 0.01, 0.08),
    Country("RO", "Romania", "EU", "RON", 45.9, 25.0, 2, 0.005, 0.01, 0.01, 0.08),
    Country("TR", "Turkey", "Europe non-EU", "TRY", 39.0, 35.2, 3, 0.005, 0.01, 0.01, 0.07),
)
COUNTRY_CODES = np.array([c.code for c in COUNTRIES])
COUNTRY_INDEX = {c.code: i for i, c in enumerate(COUNTRIES)}
UTC_OFFSETS = np.array([c.utc_offset for c in COUNTRIES], dtype=np.int64)


@dataclass(frozen=True)
class Currency:
    code: str
    eur_per_unit: float  # starting rate
    daily_vol: float
    annual_drift: float = 0.0


CURRENCIES: tuple[Currency, ...] = (
    Currency("EUR", 1.0, 0.0),
    Currency("GBP", 1.17, 0.004),
    Currency("USD", 0.92, 0.005),
    Currency("MXN", 0.052, 0.007),
    Currency("BRL", 0.17, 0.008),
    Currency("MAD", 0.092, 0.003),
    Currency("RON", 0.201, 0.002),
    Currency("TRY", 0.028, 0.006, annual_drift=-0.30),
)
CURRENCY_INDEX = {c.code: i for i, c in enumerate(CURRENCIES)}
COUNTRY_CURRENCY = np.array([CURRENCY_INDEX[c.currency] for c in COUNTRIES])


@dataclass(frozen=True)
class Category:
    name: str
    weight: float  # share of merchants
    online_share: float  # probability that a merchant of this category is online-only
    median_eur: float  # median legit ticket
    sigma: float  # lognormal spread of legit tickets
    fraud_appeal: float  # how attractive the category is to fraudsters (resale value)


CATEGORIES: tuple[Category, ...] = (
    Category("grocery", 0.22, 0.05, 35, 0.7, 0.6),
    Category("restaurants", 0.14, 0.10, 25, 0.6, 0.5),
    Category("fuel", 0.07, 0.00, 50, 0.4, 1.0),
    Category("transport", 0.07, 0.40, 8, 0.9, 0.6),
    Category("fashion", 0.08, 0.50, 60, 0.7, 1.0),
    Category("electronics", 0.04, 0.60, 180, 0.9, 2.5),
    Category("digital_goods", 0.07, 1.00, 12, 1.0, 2.0),
    Category("travel", 0.04, 0.80, 250, 0.9, 1.5),
    Category("health", 0.05, 0.10, 30, 0.7, 0.4),
    Category("utilities", 0.05, 0.90, 60, 0.5, 0.3),
    Category("entertainment", 0.06, 0.50, 30, 0.7, 1.2),
    Category("gift_cards", 0.02, 0.70, 50, 0.6, 3.0),
    Category("luxury", 0.01, 0.30, 400, 0.8, 2.0),
    Category("home", 0.04, 0.30, 70, 0.9, 0.7),
)
CATEGORY_NAMES = np.array([c.name for c in CATEGORIES])
CATEGORY_INDEX = {c.name: i for i, c in enumerate(CATEGORIES)}


@dataclass(frozen=True)
class Segment:
    name: str
    share: float
    monthly_rate: float  # mean legit transactions per month
    amount_mult: float
    travel_rate: float  # trips per month
    age: tuple[str, float, float]  # ("uniform", lo, hi) or ("normal", mean, sd)
    category_tilt: dict[str, float] = field(default_factory=dict)


# fmt: off
SEGMENTS: tuple[Segment, ...] = (
    Segment(
        "student", 0.15, 3.5, 0.6, 0.12, ("uniform", 18, 26),
        {"digital_goods": 2.0, "entertainment": 1.5, "luxury": 0.2, "travel": 0.6,
         "utilities": 0.5},
    ),
    Segment("standard", 0.55, 4.0, 1.0, 0.10, ("normal", 44, 13)),
    Segment(
        "premium", 0.15, 6.0, 1.8, 0.25, ("normal", 48, 12),
        {"luxury": 3.0, "travel": 2.0, "restaurants": 1.3},
    ),
    Segment(
        "small_business", 0.15, 5.0, 1.4, 0.18, ("normal", 45, 11),
        {"utilities": 2.0, "travel": 1.5, "fuel": 1.5, "electronics": 1.3},
    ),
)
# fmt: on
SEGMENT_NAMES = np.array([s.name for s in SEGMENTS])


def category_tilt_matrix() -> np.ndarray:
    """(n_segments, n_categories) multipliers for how often each segment buys each category."""
    tilt = np.ones((len(SEGMENTS), len(CATEGORIES)))
    for i, seg in enumerate(SEGMENTS):
        for cat, mult in seg.category_tilt.items():
            tilt[i, CATEGORY_INDEX[cat]] = mult
    return tilt


PAYMENT_METHODS = np.array(["chip_pin", "contactless", "ecommerce", "mobile_wallet"])
PM_CHIP, PM_CONTACTLESS, PM_ECOMMERCE, PM_WALLET = range(4)
CARD_PRESENT_METHODS = (PM_CHIP, PM_CONTACTLESS)

CARD_TYPES = np.array(["debit", "credit", "prepaid", "virtual"])

# Device (type, operating system) combinations and their probabilities.
LEGIT_PRIMARY_DEVICES = (("mobile", "android", 0.55), ("mobile", "ios", 0.45))
LEGIT_SECONDARY_DEVICES = (
    ("desktop", "windows", 0.49),
    ("desktop", "macos", 0.13),
    ("desktop", "linux", 0.03),
    ("tablet", "ipados", 0.21),
    ("tablet", "android", 0.14),
)
FRAUD_DEVICES = (
    ("mobile", "android", 0.40),  # often emulators
    ("desktop", "windows", 0.30),
    ("desktop", "linux", 0.15),
    ("mobile", "ios", 0.10),
    ("tablet", "android", 0.05),
)
WALLET_FRAUD_DEVICES = (("mobile", "android", 0.6), ("mobile", "ios", 0.4))

# fmt: off
# Relative activity by LOCAL hour of day.
LEGIT_HOUR_WEIGHTS = np.array(
    [0.9, 0.5, 0.3, 0.2, 0.2, 0.3, 0.8, 1.8, 3.2, 4.2, 5.0, 5.6,
     6.5, 6.8, 5.8, 5.2, 5.4, 6.0, 6.8, 7.2, 6.6, 5.0, 3.2, 1.8]
)
NIGHT_OWL_HOUR_WEIGHTS = np.roll(LEGIT_HOUR_WEIGHTS, 4)
# Remote fraud happens on the attacker's clock, so in the victim's local time it is flatter.
REMOTE_FRAUD_HOUR_WEIGHTS = np.array(
    [5.0, 5.0, 5.0, 5.0, 5.0, 4.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
     4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.5, 5.0, 5.0]
)
# fmt: on
