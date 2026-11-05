"""Legitimate transaction stream.

Each customer is a non-homogeneous Poisson process: a segment base rate scaled by a personal
activity level, day seasonality and a local hour-of-day curve. Customers mostly buy from a small
personal set of merchants (Zipf), occasionally explore, travel abroad, change phones and make
unusually large purchases. Those last behaviours are deliberate: they are legitimate events that
look suspicious, which is what makes false positives realistic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraudplatform.simulation.entities import World
from fraudplatform.simulation.randomness import DAY, HOUR, weighted_choice
from fraudplatform.simulation.reference import (
    CATEGORIES,
    COUNTRIES,
    LEGIT_HOUR_WEIGHTS,
    NIGHT_OWL_HOUR_WEIGHTS,
    PM_CHIP,
    PM_CONTACTLESS,
    PM_ECOMMERCE,
    PM_WALLET,
    SEGMENTS,
    UTC_OFFSETS,
    category_tilt_matrix,
)

log = logging.getLogger(__name__)

TXN_COLUMNS = [
    "customer",
    "card",
    "merchant",
    "device",
    "ts",
    "amount_eur",
    "country",
    "payment_method",
    "scenario",
    "incident",
]
LEGIT = "legit"
MONTH_DAYS = 30.44


def build_preferred_merchants(world: World, rng) -> np.ndarray:
    """Each customer's usual merchants: physical ones in their home country plus online ones."""
    params = world.cfg.legit
    m = world.merchants
    c = world.customers
    tilt = category_tilt_matrix()
    established = m["onboarded_s"].to_numpy() < world.cal.start_s
    online = m["online"].to_numpy()
    base_w = m["_popularity"].to_numpy() * np.where(online, params.online_preference, 1.0)

    preferred = np.empty((len(c), params.preferred_merchants), dtype=np.int64)
    for (seg, country), group in c.groupby(["segment", "country"]).groups.items():
        candidates = np.flatnonzero(established & (online | (m["country"].to_numpy() == country)))
        w = base_w[candidates] * tilt[seg, m["category"].to_numpy()[candidates]]
        draws = rng.choice(candidates, size=(len(group), params.preferred_merchants), p=w / w.sum())
        preferred[np.asarray(group)] = draws
    return preferred


def _sample_by_country(rng, loc: np.ndarray, pools: dict[int, tuple[np.ndarray, np.ndarray]]):
    out = np.empty(len(loc), dtype=np.int64)
    for country in np.unique(loc):
        rows = np.flatnonzero(loc == country)
        cands, p = pools[country]
        out[rows] = rng.choice(cands, size=len(rows), p=p)
    return out


def _pool(mask: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cands = np.flatnonzero(mask)
    w = weights[cands]
    return cands, w / w.sum()


def generate_legit(world: World) -> pd.DataFrame:
    rng = world.rngs.get("legit")
    cfg = world.cfg
    cal = world.cal
    cust = world.customers
    merch = world.merchants
    n_cust = len(cust)

    world.preferred = build_preferred_merchants(world, world.rngs.get("preferred"))

    # 1. How many transactions each customer makes while they hold a card.
    first_card = world.first_card_s
    active_from = np.maximum(first_card, cal.start_s)
    active_days = np.maximum((cal.end_s - active_from) / DAY, 0)
    seg = cust["segment"].to_numpy()
    rate = np.array([s.monthly_rate for s in SEGMENTS])[seg] * cust["_activity"].to_numpy()
    n_txn = rng.poisson(rate * active_days / MONTH_DAYS)
    c = np.repeat(np.arange(n_cust), n_txn)
    n = len(c)
    log.info("legit: %s customers -> %s transactions", n_cust, n)

    # 2. Which day, and which local hour.
    day = cal.sample_days(rng, cal.day_index(active_from)[c], np.full(n, cal.n_days - 1))
    night_owl = cust["_night_owl"].to_numpy()[c]
    hour = np.where(
        night_owl,
        weighted_choice(rng, NIGHT_OWL_HOUR_WEIGHTS, n),
        weighted_choice(rng, LEGIT_HOUR_WEIGHTS, n),
    )
    local_s = hour * HOUR + rng.integers(0, HOUR, n)

    # 3. Where the customer is: at home, or travelling.
    home = cust["country"].to_numpy()
    loc = home[c].copy()
    trip_rate = np.array([s.travel_rate for s in SEGMENTS])[seg]
    n_trips = rng.poisson(trip_rate * active_days / MONTH_DAYS)
    trip_c = np.repeat(np.arange(n_cust), n_trips)
    if len(trip_c):
        lo_day = cal.day_index(active_from)[trip_c]
        trip_start = lo_day + (rng.random(len(trip_c)) * (cal.n_days - lo_day)).astype(np.int64)
        trip_end = trip_start + rng.integers(2, 15, len(trip_c))
        travel_w = np.array([ct.travel_weight for ct in COUNTRIES])
        dest = np.empty(len(trip_c), dtype=np.int64)
        trip_home = home[trip_c]
        for h in np.unique(trip_home):
            rows = np.flatnonzero(trip_home == h)
            w = travel_w.copy()
            w[h] = 0  # a trip goes abroad
            dest[rows] = rng.choice(len(w), size=len(rows), p=w / w.sum())
        trips = pd.DataFrame({"c": trip_c, "day": trip_start, "end": trip_end, "dest": dest})
        txn = pd.DataFrame({"c": c, "day": day, "row": np.arange(n)})
        merged = pd.merge_asof(
            txn.sort_values("day"), trips.sort_values("day"), on="day", by="c", direction="backward"
        ).sort_values("row")
        in_trip = (merged["end"].to_numpy() >= day) & merged["dest"].notna().to_numpy()
        loc[in_trip] = merged["dest"].to_numpy()[in_trip].astype(np.int64)
    else:
        in_trip = np.zeros(n, dtype=bool)

    ts = cal.day_start_s[day] + local_s - UTC_OFFSETS[loc] * HOUR
    too_early = ts < first_card[c] + 60
    ts[too_early] = first_card[c][too_early] + rng.integers(60, 6 * HOUR, too_early.sum())
    # Local midnight in a UTC+ country can fall on the previous UTC day; keep the period strict.
    ts = np.clip(ts, cal.start_s, cal.end_s)

    # 4. Which merchant.
    params = cfg.legit
    k = params.preferred_merchants
    zipf = 1.0 / np.arange(1, k + 1) ** params.zipf_exponent
    merchant = world.preferred[c, weighted_choice(rng, zipf, n)]

    m_online = merch["online"].to_numpy()
    m_country = merch["country"].to_numpy()
    pop = merch["_popularity"].to_numpy()
    n_countries = len(COUNTRIES)
    explore_pools = {i: _pool(m_online | (m_country == i), pop) for i in range(n_countries)}
    physical_pools = {i: _pool(~m_online & (m_country == i), pop) for i in range(n_countries)}

    explore = rng.random(n) < params.explore_prob
    merchant[explore] = _sample_by_country(rng, loc[explore], explore_pools)
    # While travelling, physical purchases happen in the destination country.
    abroad_physical = in_trip & ~m_online[merchant] & (m_country[merchant] != loc)
    merchant[abroad_physical] = _sample_by_country(rng, loc[abroad_physical], physical_pools)
    # A merchant cannot be used before it is onboarded.
    not_yet = merch["onboarded_s"].to_numpy()[merchant] > ts
    merchant[not_yet] = world.preferred[c[not_yet], 0]

    drift_s = int(np.datetime64(cfg.drift.start, "s").astype(np.int64))
    after_drift = ts >= drift_s
    online_pool = _pool(m_online & (merch["onboarded_s"].to_numpy() < drift_s), pop)
    to_online = after_drift & ~m_online[merchant] & (rng.random(n) < cfg.drift.online_shift_prob)
    merchant[to_online] = rng.choice(online_pool[0], size=to_online.sum(), p=online_pool[1])

    # 5. Payment method, transaction country, card and device.
    online_txn = m_online[merchant]
    wallet_share = np.where(after_drift, cfg.drift.wallet_share_physical, 0.10)
    u = rng.random(n)
    pm = np.where(
        online_txn,
        np.where(u < 0.8, PM_ECOMMERCE, PM_WALLET),
        np.where(
            u < wallet_share,
            PM_WALLET,
            np.where(u < wallet_share + (1 - wallet_share) * 0.6, PM_CONTACTLESS, PM_CHIP),
        ),
    )
    country = np.where(online_txn, loc, m_country[merchant])

    card = c.copy()  # the first card of customer i has index i
    card2 = world.card2_of_customer[c]
    card2_issued = np.where(card2 >= 0, world.cards["issued_s"].to_numpy()[card2], np.inf)
    use2 = (card2 >= 0) & (ts >= card2_issued) & (rng.random(n) < cfg.cards.second_card_usage)
    card[use2] = card2[use2]

    devs = world.customer_devices
    device = np.full(n, -1, dtype=np.int64)
    needs_device = (pm == PM_ECOMMERCE) | (pm == PM_WALLET)
    device[needs_device] = devs.primary_at(c[needs_device], ts[needs_device])
    use_secondary = (
        (pm == PM_ECOMMERCE)
        & (devs.secondary[c] >= 0)
        & (rng.random(n) < cfg.devices.secondary_usage)
    )
    device[use_secondary] = devs.secondary[c[use_secondary]]

    # 6. Amount in EUR.
    cat = merch["category"].to_numpy()[merchant]
    median = np.array([x.median_eur for x in CATEGORIES])[cat]
    sigma = np.array([x.sigma for x in CATEGORIES])[cat]
    seg_mult = np.array([s.amount_mult for s in SEGMENTS])[seg[c]]
    amount = median * seg_mult * cust["_amount_mult"].to_numpy()[c] * rng.lognormal(0, sigma)
    splurge = rng.random(n) < params.splurge_prob
    amount[splurge] *= rng.uniform(*params.splurge_multiplier, splurge.sum())
    amount[after_drift] *= cfg.drift.amount_inflation
    amount = np.round(np.maximum(amount, 0.5), 2)

    assert not (pm[device < 0] >= PM_ECOMMERCE).any()
    assert set(np.unique(pm)) <= {PM_CHIP, PM_CONTACTLESS, PM_ECOMMERCE, PM_WALLET}
    return pd.DataFrame(
        {
            "customer": c,
            "card": card,
            "merchant": merchant,
            "device": device,
            "ts": ts,
            "amount_eur": amount,
            "country": country,
            "payment_method": pm,
            "scenario": LEGIT,
            "incident": -1,
        }
    )[TXN_COLUMNS]
