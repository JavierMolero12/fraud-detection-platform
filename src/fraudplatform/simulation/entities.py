"""Generation of the static entities: customers, cards, merchants and devices.

Entities are held as column arrays indexed by an integer position; string IDs are only attached
when the dataset is written. Columns starting with an underscore are latent simulator state
(e.g. a merchant's true riskiness) and are never written to the output tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudplatform.config import SimulationConfig
from fraudplatform.simulation.randomness import (
    DAY,
    Calendar,
    RngFactory,
    lognormal_mean_one,
    weighted_choice,
)
from fraudplatform.simulation.reference import (
    CATEGORIES,
    CATEGORY_INDEX,
    COUNTRIES,
    LEGIT_PRIMARY_DEVICES,
    LEGIT_SECONDARY_DEVICES,
    SEGMENTS,
)

YEAR = 365.25 * DAY
NO_TIME = np.iinfo(np.int64).max  # "never happens" for optional events


def generate_customers(n: int, cal: Calendar, cfg: SimulationConfig, rng) -> pd.DataFrame:
    params = cfg.customers
    segment = weighted_choice(rng, [s.share for s in SEGMENTS], n)
    country = weighted_choice(rng, [c.customer_weight for c in COUNTRIES], n)

    age = np.empty(n)
    for i, seg in enumerate(SEGMENTS):
        mask = segment == i
        kind, a, b = seg.age
        if kind == "uniform":
            age[mask] = rng.uniform(a, b, mask.sum())
        else:
            age[mask] = np.clip(rng.normal(a, b, mask.sum()), 18.5, 88)
    dob_s = (cal.start_s - age * YEAR - rng.uniform(0, DAY, n)).astype(np.int64)

    before = rng.random(n) < params.share_created_before_start
    created_before = cal.start_s - rng.uniform(0, params.max_account_age_years * YEAR, n)
    created_during = rng.uniform(cal.start_s, cal.end_s - 14 * DAY, n)
    created_s = np.where(before, created_before, created_during).astype(np.int64)
    # Accounts are opened by adults. 18 calendar years can be up to 6575 days, more than
    # 18 * 365.25; the extra day guarantees the account opens on or after the 18th birthday
    # (v1.0.0 had 448 customers opening a few hours early; found by CUS_UNDERAGE_ACCOUNT).
    created_s = np.maximum(created_s, dob_s + int(18 * YEAR) + DAY)

    return pd.DataFrame(
        {
            "segment": segment,
            "country": country,
            "dob_s": dob_s,
            "created_s": created_s,
            "_activity": lognormal_mean_one(rng, params.activity_sigma, n),
            "_amount_mult": lognormal_mean_one(rng, params.amount_sigma, n),
            "_night_owl": rng.random(n) < params.night_owl_share,
        }
    )


def generate_cards(customers: pd.DataFrame, cal: Calendar, cfg: SimulationConfig, rng):
    """One card per customer at account opening, plus an optional second card issued later."""
    n = len(customers)
    created = customers["created_s"].to_numpy()
    issued1 = created + rng.integers(0, 3 * DAY, n)

    lo = issued1 + 30 * DAY
    hi = cal.end_s - 14 * DAY
    has2 = (rng.random(n) < cfg.cards.second_card_prob) & (lo < hi)
    draw = (lo + rng.random(n) * np.maximum(hi - lo, 0)).astype(np.int64)
    issued2 = np.where(has2, draw, NO_TIME)

    type1 = weighted_choice(rng, [0.60, 0.30, 0.10, 0.00], n)  # debit, credit, prepaid, virtual
    type2 = weighted_choice(rng, [0.15, 0.40, 0.10, 0.35], n)

    idx2 = np.flatnonzero(has2)
    cards = pd.DataFrame(
        {
            "customer": np.concatenate([np.arange(n), idx2]),
            "card_type": np.concatenate([type1, type2[idx2]]),
            "issued_s": np.concatenate([issued1, issued2[idx2]]),
        }
    )
    card2_of_customer = np.full(n, -1, dtype=np.int64)
    card2_of_customer[idx2] = np.arange(n, n + len(idx2))
    return cards, card2_of_customer


def generate_merchants(n: int, cal: Calendar, rng) -> pd.DataFrame:
    n_countries = len(COUNTRIES)
    if n < 3 * n_countries:
        raise ValueError(f"need at least {3 * n_countries} merchants")

    # Guarantee a few physical merchants in every country (travel and card cloning need them).
    reserved_cat = np.tile(
        [CATEGORY_INDEX["grocery"], CATEGORY_INDEX["restaurants"], CATEGORY_INDEX["fuel"]],
        n_countries,
    )
    reserved_country = np.repeat(np.arange(n_countries), 3)

    n_rand = n - len(reserved_cat)
    cat = np.concatenate(
        [reserved_cat, weighted_choice(rng, [c.weight for c in CATEGORIES], n_rand)]
    )
    online_share = np.array([c.online_share for c in CATEGORIES])
    online = rng.random(n) < online_share[cat]
    online[: len(reserved_cat)] = False

    physical_w = np.array([0.75 * c.customer_weight + 0.25 * c.travel_weight for c in COUNTRIES])
    online_w = np.array([c.online_weight for c in COUNTRIES])
    country = np.where(
        online, weighted_choice(rng, online_w, n), weighted_choice(rng, physical_w, n)
    )
    country[: len(reserved_cat)] = reserved_country

    appeal = np.array([c.fraud_appeal for c in CATEGORIES])
    latent_risk = appeal[cat] * rng.lognormal(0, 0.5, n)
    # The published risk rating is a noisy external view of the latent risk, fixed at onboarding.
    rating = np.log(latent_risk) + rng.normal(0, 0.6, n)
    q60, q90 = np.quantile(rating, [0.6, 0.9])
    risk_level = np.where(rating >= q90, "high", np.where(rating >= q60, "medium", "low"))

    new = rng.random(n) < 0.10
    new[: len(reserved_cat)] = False
    onboarded_s = np.where(
        new,
        rng.uniform(cal.start_s, cal.end_s - 30 * DAY, n),
        cal.start_s - rng.uniform(30 * DAY, 6 * YEAR, n),
    ).astype(np.int64)

    testable_cat = np.isin(cat, [CATEGORY_INDEX["digital_goods"], CATEGORY_INDEX["entertainment"]])
    testable = online & testable_cat & ~new
    if testable.any():
        testable &= latent_risk >= np.quantile(latent_risk[testable], 0.4)

    return pd.DataFrame(
        {
            "category": cat,
            "online": online,
            "country": country,
            "risk_level": risk_level,
            "onboarded_s": onboarded_s,
            "_popularity": rng.lognormal(0, 1.2, n),
            "_latent_risk": latent_risk,
            "_testable": testable,
        }
    )


@dataclass
class DeviceRegistry:
    """Append-only list of devices; legit devices first, fraud devices are added by scenarios."""

    device_type: list[str] = field(default_factory=list)
    os: list[str] = field(default_factory=list)
    is_fraud_device: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.device_type)

    def add(self, rng, n: int, mix, fraud: bool) -> np.ndarray:
        choice = weighted_choice(rng, [w for _, _, w in mix], n)
        start = len(self)
        self.device_type.extend(mix[i][0] for i in choice)
        self.os.extend(mix[i][1] for i in choice)
        self.is_fraud_device.extend([fraud] * n)
        return np.arange(start, start + n)


@dataclass
class CustomerDevices:
    primary1: np.ndarray
    primary2: np.ndarray  # -1 if the customer never replaces their phone
    switch_s: np.ndarray  # when primary2 replaces primary1
    secondary: np.ndarray  # -1 if none

    def primary_at(self, customer: np.ndarray, ts: np.ndarray) -> np.ndarray:
        switched = ts >= self.switch_s[customer]
        return np.where(switched, self.primary2[customer], self.primary1[customer])


def generate_customer_devices(
    customers: pd.DataFrame, first_card_s: np.ndarray, cal: Calendar, cfg, reg, rng
) -> CustomerDevices:
    params = cfg.devices
    n = len(customers)
    primary1 = reg.add(rng, n, LEGIT_PRIMARY_DEVICES, fraud=False)

    upgrade = rng.random(n) < params.upgrade_prob
    window = np.maximum(cal.end_s - np.maximum(first_card_s, cal.start_s), DAY)
    draw = (np.maximum(first_card_s, cal.start_s) + rng.random(n) * window).astype(np.int64)
    switch_s = np.where(upgrade, draw, NO_TIME)
    primary2 = np.full(n, -1, dtype=np.int64)
    primary2[upgrade] = reg.add(rng, int(upgrade.sum()), LEGIT_PRIMARY_DEVICES, fraud=False)

    has_sec = rng.random(n) < params.secondary_prob
    secondary = np.full(n, -1, dtype=np.int64)
    secondary[has_sec] = reg.add(rng, int(has_sec.sum()), LEGIT_SECONDARY_DEVICES, fraud=False)

    # Families share a computer: the same device legitimately appears for two customers.
    country = customers["country"].to_numpy()
    sharers = np.flatnonzero(has_sec & (rng.random(n) < params.shared_secondary_prob))
    for i in sharers:
        candidates = np.flatnonzero(country == country[i])
        partner = candidates[rng.integers(len(candidates))]
        if partner != i:
            secondary[partner] = secondary[i]

    return CustomerDevices(primary1, primary2, switch_s, secondary)


@dataclass
class World:
    """Everything the transaction generators need to know about the static entities."""

    cal: Calendar
    cfg: SimulationConfig
    rngs: RngFactory
    customers: pd.DataFrame
    cards: pd.DataFrame
    card2_of_customer: np.ndarray
    merchants: pd.DataFrame
    devices: DeviceRegistry
    customer_devices: CustomerDevices
    preferred: np.ndarray | None = None  # (n_customers, k) usual merchants, set by legit module

    @property
    def first_card_s(self) -> np.ndarray:
        return self.cards["issued_s"].to_numpy()[: len(self.customers)]


def build_world(cfg: SimulationConfig, scale: str, rngs: RngFactory) -> World:
    sc = cfg.scale(scale)
    cal = Calendar(sc.start, sc.end)
    customers = generate_customers(sc.n_customers, cal, cfg, rngs.get("customers"))
    cards, card2 = generate_cards(customers, cal, cfg, rngs.get("cards"))
    merchants = generate_merchants(sc.n_merchants, cal, rngs.get("merchants"))
    registry = DeviceRegistry()
    first_card_s = cards["issued_s"].to_numpy()[: len(customers)]
    cust_devices = generate_customer_devices(
        customers, first_card_s, cal, cfg, registry, rngs.get("devices")
    )
    return World(cal, cfg, rngs, customers, cards, card2, merchants, registry, cust_devices)
