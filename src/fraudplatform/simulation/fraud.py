"""Fraud scenarios.

Fraud is generated as *incidents* of behaviour (a card-testing burst, an account takeover, ...)
rather than by flipping a label with a formula over the columns. Each scenario leaves a different,
partial footprint in the data; one of them (stealth) leaves almost none on purpose, so that the
problem has an irreducible error floor.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraudplatform.config import DRIFT_SCENARIO, SCENARIOS
from fraudplatform.simulation.entities import World
from fraudplatform.simulation.legit import TXN_COLUMNS
from fraudplatform.simulation.randomness import DAY, HOUR, weighted_choice
from fraudplatform.simulation.reference import (
    CATEGORIES,
    CATEGORY_INDEX,
    COUNTRIES,
    FRAUD_DEVICES,
    LEGIT_HOUR_WEIGHTS,
    PM_CHIP,
    PM_CONTACTLESS,
    PM_ECOMMERCE,
    PM_WALLET,
    REMOTE_FRAUD_HOUR_WEIGHTS,
    SEGMENTS,
    UTC_OFFSETS,
    WALLET_FRAUD_DEVICES,
)

log = logging.getLogger(__name__)
MINUTE = 60

# Average fraud transactions per incident; used to turn a target number of fraud transactions
# into a number of incidents.
MEAN_TXNS_PER_INCIDENT = {
    "card_testing": 8.2,
    "account_takeover": 4.0,
    "card_cloning": 3.5,
    "fraud_ring": 2.0,
    "merchant_compromise": 2.5,
    "stealth": 1.5,
    DRIFT_SCENARIO: 4.0,
}
CARD_TYPE_FRAUD_WEIGHT = np.array([1.0, 1.3, 2.0, 1.6])  # debit, credit, prepaid, virtual


class FraudGenerator:
    def __init__(self, world: World, legit: pd.DataFrame):
        self.world = world
        self.legit = legit
        self.cfg = world.cfg
        self.cal = world.cal

        m = world.merchants
        self.m_cat = m["category"].to_numpy()
        self.m_online = m["online"].to_numpy()
        self.m_country = m["country"].to_numpy()
        self.m_weight = m["_latent_risk"].to_numpy() * np.sqrt(m["_popularity"].to_numpy())
        self.m_established = m["onboarded_s"].to_numpy() < self.cal.start_s
        self.m_testable = m["_testable"].to_numpy()

        cards = world.cards
        cust = world.customers
        self.card_customer = cards["customer"].to_numpy()
        self.card_issued = cards["issued_s"].to_numpy()
        joined_during = cust["created_s"].to_numpy()[self.card_customer] >= self.cal.start_s
        w = CARD_TYPE_FRAUD_WEIGHT[cards["card_type"].to_numpy()] * np.where(
            joined_during, 1.5, 1.0
        )
        self.card_p = w / w.sum()

        self.home = cust["country"].to_numpy()
        self.segment = cust["segment"].to_numpy()
        self.amount_mult = cust["_amount_mult"].to_numpy()
        self.attacker_w = np.array([c.attacker_weight for c in COUNTRIES])

        self._pool_cache: dict = {}
        self._parts: list[dict[str, np.ndarray]] = []
        self._next_incident = 0
        self.target_txns = len(legit) * self.cfg.fraud.target_rate

    # ----------------------------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------------------------
    def _n_incidents(self, scenario: str, txn_share: float) -> int:
        return max(1, round(self.target_txns * txn_share / MEAN_TXNS_PER_INCIDENT[scenario]))

    def _new_incident(self) -> int:
        self._next_incident += 1
        return self._next_incident - 1

    def _victims(self, rng, t: np.ndarray) -> np.ndarray:
        """Cards targeted at times t; a card must have been issued at least a day before."""
        cards = rng.choice(len(self.card_p), size=len(t), p=self.card_p)
        for _ in range(100):
            bad = self.card_issued[cards] > t - DAY
            if not bad.any():
                return cards
            cards[bad] = rng.choice(len(self.card_p), size=bad.sum(), p=self.card_p)
        raise RuntimeError("could not find eligible victim cards")

    def _attacker_country(self, rng, home: int, home_prob: float | None = None) -> int:
        p = self.cfg.fraud.attacker_home_country_prob if home_prob is None else home_prob
        if rng.random() < p:
            return int(home)
        w = self.attacker_w.copy()
        w[home] = 0
        return int(rng.choice(len(w), p=w / w.sum()))

    def _pool(self, cat: int | None, online: bool | None, country: int | None):
        key = (cat, online, country)
        if key not in self._pool_cache:
            mask = self.m_established.copy()
            if cat is not None:
                mask &= self.m_cat == cat
            if online is not None:
                mask &= self.m_online == online
            if country is not None:
                mask &= self.m_country == country
            cands = np.flatnonzero(mask)
            w = self.m_weight[cands]
            self._pool_cache[key] = (cands, w / w.sum()) if len(cands) else None
        return self._pool_cache[key]

    def _merchants(self, rng, n: int, cat_probs: dict[str, float], online: bool, country=None):
        names = list(cat_probs)
        cats = weighted_choice(rng, list(cat_probs.values()), n)
        out = np.empty(n, dtype=np.int64)
        for ci in np.unique(cats):
            rows = np.flatnonzero(cats == ci)
            cat = CATEGORY_INDEX[names[ci]]
            pool = (
                self._pool(cat, online, country)
                or self._pool(None, online, country)
                or self._pool(cat, online, None)
                or self._pool(None, online, None)
            )
            out[rows] = rng.choice(pool[0], size=len(rows), p=pool[1])
        return out

    def _typical_amount(self, customer: int, merchants: np.ndarray) -> np.ndarray:
        median = np.array([c.median_eur for c in CATEGORIES])[self.m_cat[merchants]]
        seg_mult = SEGMENTS[self.segment[customer]].amount_mult
        return median * seg_mult * self.amount_mult[customer]

    def _retime(self, rng, t: np.ndarray, country: np.ndarray, hour_weights) -> np.ndarray:
        """Keep the (UTC) day of t but draw a new local hour in the given country's timezone."""
        day_start = t - t % DAY
        local = weighted_choice(rng, hour_weights, len(t)) * HOUR + rng.integers(0, HOUR, len(t))
        return day_start + local - UTC_OFFSETS[country] * HOUR

    @staticmethod
    def _sequence(rng, t0: float, n: int, mean_gap: float, min_gap: float) -> np.ndarray:
        gaps = min_gap + rng.exponential(mean_gap, n)
        gaps[0] = 0
        return (t0 + np.cumsum(gaps)).astype(np.int64)

    def _emit(self, scenario, incident, customer, card, merchant, device, ts, amount, country, pm):
        n = len(ts)
        self._parts.append(
            {
                "customer": np.full(n, customer, dtype=np.int64),
                "card": np.full(n, card, dtype=np.int64),
                "merchant": np.asarray(merchant, dtype=np.int64),
                "device": np.broadcast_to(np.asarray(device, dtype=np.int64), (n,)).copy(),
                "ts": np.asarray(ts, dtype=np.int64),
                "amount_eur": np.round(np.maximum(np.asarray(amount, dtype=float), 0.5), 2),
                "country": np.full(n, country, dtype=np.int64),
                "payment_method": np.broadcast_to(np.asarray(pm, dtype=np.int64), (n,)).copy(),
                "scenario": np.full(n, scenario, dtype=object),
                "incident": np.full(n, incident, dtype=np.int64),
            }
        )

    def _remote_times(self, rng, n: int, lo_s: int, hi_s: int) -> np.ndarray:
        return self.cal.sample_times(
            rng, np.full(n, lo_s), np.full(n, hi_s), REMOTE_FRAUD_HOUR_WEIGHTS, np.full(n, 1)
        )

    # ----------------------------------------------------------------------------------------
    # scenarios
    # ----------------------------------------------------------------------------------------
    def card_testing(self, n_incidents: int) -> None:
        """Burst of micro-charges to check the card works, then a few large cash-out purchases."""
        rng = self.world.rngs.get("fraud.card_testing")
        t0 = self._remote_times(rng, n_incidents, self.cal.start_s, self.cal.end_s - 2 * DAY)
        cards = self._victims(rng, t0)
        testable = np.flatnonzero(self.m_testable)
        if len(testable) == 0:
            testable = self._pool(None, True, None)[0]
        bots: list[int] = []
        for t, card in zip(t0, cards, strict=True):
            customer = self.card_customer[card]
            country = self._attacker_country(rng, self.home[customer])
            if bots and rng.random() < 0.3:
                device = bots[rng.integers(len(bots))]  # testing scripts reuse their machines
            else:
                device = int(self.world.devices.add(rng, 1, FRAUD_DEVICES, fraud=True)[0])
                bots.append(device)
            incident = self._new_incident()

            n_test = int(rng.integers(3, 11))
            ts = self._sequence(rng, t, n_test, mean_gap=MINUTE, min_gap=5)
            targets = rng.choice(testable, size=rng.integers(1, 3))
            merchant = targets[rng.integers(0, len(targets), n_test)]
            amount = np.where(rng.random(n_test) < 0.2, 1.0, rng.uniform(0.5, 5.0, n_test))
            self._emit(
                "card_testing",
                incident,
                customer,
                card,
                merchant,
                device,
                ts,
                amount,
                country,
                PM_ECOMMERCE,
            )

            if rng.random() < 0.85:
                n_big = int(rng.integers(1, 4))
                start = ts[-1] + rng.exponential(6 * HOUR)
                ts_big = self._sequence(rng, start, n_big, mean_gap=30 * MINUTE, min_gap=5 * MINUTE)
                merch = self._merchants(
                    rng,
                    n_big,
                    {"electronics": 0.4, "gift_cards": 0.3, "digital_goods": 0.15, "luxury": 0.15},
                    online=True,
                )
                amount = rng.lognormal(np.log(350), 0.7, n_big)
                self._emit(
                    "card_testing",
                    incident,
                    customer,
                    card,
                    merch,
                    device,
                    ts_big,
                    amount,
                    country,
                    PM_ECOMMERCE,
                )

    def account_takeover(self, n_incidents: int) -> None:
        """Attacker logs in from a new device and buys resellable goods within hours."""
        rng = self.world.rngs.get("fraud.account_takeover")
        t0 = self._remote_times(rng, n_incidents, self.cal.start_s, self.cal.end_s - DAY)
        cards = self._victims(rng, t0)
        devices = self.world.devices.add(rng, n_incidents, FRAUD_DEVICES, fraud=True)
        for t, card, device in zip(t0, cards, devices, strict=True):
            customer = self.card_customer[card]
            country = self._attacker_country(rng, self.home[customer], home_prob=0.5)
            n = int(rng.integers(2, 7))
            ts = self._sequence(rng, t, n, mean_gap=40 * MINUTE, min_gap=2 * MINUTE)
            merch = self._merchants(
                rng,
                n,
                {
                    "electronics": 0.35,
                    "gift_cards": 0.2,
                    "luxury": 0.15,
                    "travel": 0.15,
                    "fashion": 0.15,
                },
                online=True,
            )
            amount = np.maximum(
                self._typical_amount(customer, merch) * rng.lognormal(np.log(4), 0.5, n), 20
            )
            pm = np.where(rng.random(n) < 0.7, PM_ECOMMERCE, PM_WALLET)
            self._emit(
                "account_takeover",
                self._new_incident(),
                customer,
                card,
                merch,
                device,
                ts,
                amount,
                country,
                pm,
            )

    def card_cloning(self, n_incidents: int) -> None:
        """Counterfeit card used in shops abroad while the real customer is at home."""
        rng = self.world.rngs.get("fraud.card_cloning")
        t_day = self._remote_times(rng, n_incidents, self.cal.start_s, self.cal.end_s - DAY)
        cards = self._victims(rng, t_day)
        customers = self.card_customer[cards]
        clone_country = np.array(
            [self._attacker_country(rng, h, home_prob=0.0) for h in self.home[customers]]
        )
        t0 = np.maximum(
            self._retime(rng, t_day, clone_country, LEGIT_HOUR_WEIGHTS),
            self.card_issued[cards] + HOUR,
        )
        for t, card, customer, country in zip(t0, cards, customers, clone_country, strict=True):
            n = int(rng.integers(2, 6))
            ts = self._sequence(rng, t, n, mean_gap=50 * MINUTE, min_gap=3 * MINUTE)
            merch = self._merchants(
                rng,
                n,
                {
                    "fuel": 0.25,
                    "electronics": 0.25,
                    "luxury": 0.15,
                    "grocery": 0.2,
                    "fashion": 0.15,
                },
                online=False,
                country=int(country),
            )
            amount = rng.lognormal(np.log(120), 0.8, n)
            pm = np.where(rng.random(n) < 0.7, PM_CHIP, PM_CONTACTLESS)
            self._emit(
                "card_cloning",
                self._new_incident(),
                customer,
                card,
                merch,
                -1,
                ts,
                amount,
                self.m_country[merch[0]],
                pm,
            )

    def fraud_ring(self, n_txns: float) -> None:
        """A few devices, each cashing out many stolen cards from unrelated customers."""
        rng = self.world.rngs.get("fraud.fraud_ring")
        n_victims = max(1, round(n_txns / MEAN_TXNS_PER_INCIDENT["fraud_ring"]))
        n_devices = max(1, round(n_victims / self.cfg.fraud.cards_per_ring_device))
        devices = self.world.devices.add(rng, n_devices, FRAUD_DEVICES, fraud=True)
        per_device = rng.multinomial(n_victims, np.full(n_devices, 1 / n_devices))
        for device, k in zip(devices, per_device, strict=True):
            if k == 0:
                continue
            w = self.attacker_w
            country = int(rng.choice(len(w), p=w / w.sum()))
            start = rng.uniform(self.cal.start_s, max(self.cal.start_s, self.cal.end_s - 20 * DAY))
            duration = rng.uniform(20, 60) * DAY
            t_victims = np.minimum(start + rng.random(k) * duration, self.cal.end_s - HOUR)
            for t, card in zip(t_victims, self._victims(rng, t_victims), strict=True):
                n = int(rng.integers(1, 4))
                ts = self._sequence(rng, t, n, mean_gap=30 * MINUTE, min_gap=MINUTE)
                merch = self._merchants(
                    rng,
                    n,
                    {
                        "digital_goods": 0.3,
                        "gift_cards": 0.3,
                        "electronics": 0.25,
                        "entertainment": 0.15,
                    },
                    online=True,
                )
                amount = rng.lognormal(np.log(90), 0.6, n)
                self._emit(
                    "fraud_ring",
                    self._new_incident(),
                    self.card_customer[card],
                    card,
                    merch,
                    device,
                    ts,
                    amount,
                    country,
                    PM_ECOMMERCE,
                )

    def merchant_compromise(self, n_txns: float) -> None:
        """Cards skimmed at a compromised merchant are defrauded online weeks later."""
        rng = self.world.rngs.get("fraud.merchant_compromise")
        legit = self.legit
        risky_cats = [
            CATEGORY_INDEX[c] for c in ("restaurants", "fuel", "grocery", "entertainment")
        ]
        physical = legit[~self.m_online[legit["merchant"].to_numpy()]]
        physical = physical[np.isin(self.m_cat[physical["merchant"].to_numpy()], risky_cats)]
        volume = physical["merchant"].value_counts()
        n_comp = self.cfg.fraud.compromised_merchants
        top = volume.index.to_numpy()[: 3 * n_comp]
        if len(top) == 0:
            log.warning("merchant_compromise: no candidate merchants, scenario skipped")
            return
        compromised = rng.choice(top, size=min(n_comp, len(top)), replace=False)

        lo = self.cal.start_s + 14 * DAY
        hi = max(lo, self.cal.end_s - 60 * DAY)
        visits = []
        for merchant in compromised:
            w_start = rng.uniform(lo, hi)
            v = physical[
                (physical["merchant"] == merchant)
                & (physical["ts"] >= w_start)
                & (physical["ts"] < w_start + 21 * DAY)
            ]
            v = v.drop_duplicates("card")[["card", "ts"]].assign(source=merchant)
            visits.append(v)
        visits = pd.concat(visits, ignore_index=True)
        n_victims = round(n_txns / MEAN_TXNS_PER_INCIDENT["merchant_compromise"])
        if len(visits) == 0 or n_victims == 0:
            log.warning("merchant_compromise: no skimmed cards, scenario skipped")
            return
        victims = visits.iloc[
            rng.choice(len(visits), size=min(n_victims, len(visits)), replace=False)
        ]

        # Stolen card data is sold to a few buyers per breach; each buyer uses one device.
        buyers = {}
        for merchant in compromised:
            n_buyers = int(rng.integers(1, 4))
            devs = self.world.devices.add(rng, n_buyers, FRAUD_DEVICES, fraud=True)
            buyers[merchant] = [
                (int(d), self._attacker_country(rng, 0, home_prob=0.0)) for d in devs
            ]

        for card, visit_ts, source in victims.itertuples(index=False):
            t = visit_ts + rng.uniform(7, 42) * DAY
            if t > self.cal.end_s - HOUR:
                continue
            device, country = buyers[source][rng.integers(len(buyers[source]))]
            customer = self.card_customer[card]
            if rng.random() < self.cfg.fraud.attacker_home_country_prob:
                country = int(self.home[customer])
            n = int(rng.integers(1, 5))
            ts = self._sequence(rng, t, n, mean_gap=3 * HOUR, min_gap=5 * MINUTE)
            merch = self._merchants(
                rng,
                n,
                {"electronics": 0.35, "gift_cards": 0.25, "digital_goods": 0.2, "fashion": 0.2},
                online=True,
            )
            amount = rng.lognormal(np.log(150), 0.8, n)
            self._emit(
                "merchant_compromise",
                self._new_incident(),
                customer,
                card,
                merch,
                device,
                ts,
                amount,
                country,
                PM_ECOMMERCE,
            )

    def stealth(self, n_incidents: int) -> None:
        """Fraud that looks like the customer: usual merchants, usual device, usual amounts."""
        rng = self.world.rngs.get("fraud.stealth")
        t_day = self._remote_times(rng, n_incidents, self.cal.start_s, self.cal.end_s - 4 * DAY)
        cards = self._victims(rng, t_day)
        customers = self.card_customer[cards]
        t0 = np.maximum(
            self._retime(rng, t_day, self.home[customers], LEGIT_HOUR_WEIGHTS),
            self.card_issued[cards] + HOUR,
        )
        k = self.world.preferred.shape[1]
        zipf = 1.0 / np.arange(1, k + 1) ** self.cfg.legit.zipf_exponent
        devs = self.world.customer_devices
        for t, card, customer in zip(t0, cards, customers, strict=True):
            n = 1 if rng.random() < 0.5 else 2
            ts = self._sequence(rng, t, n, mean_gap=DAY, min_gap=HOUR)
            merch = self.world.preferred[customer, weighted_choice(rng, zipf, n)]
            amount = self._typical_amount(customer, merch) * rng.uniform(1.0, 2.5, n)
            online = self.m_online[merch]
            device = np.where(online, devs.primary_at(np.full(n, customer), ts), -1)
            pm = np.where(
                online, PM_ECOMMERCE, np.where(rng.random(n) < 0.6, PM_CONTACTLESS, PM_CHIP)
            )
            self._emit(
                "stealth",
                self._new_incident(),
                customer,
                card,
                merch,
                device,
                ts,
                amount,
                self.home[customer],
                pm,
            )

    def wallet_provisioning(self) -> None:
        """Drift scenario: stolen cards are added to a mobile wallet and used in local shops."""
        drift_s = int(np.datetime64(self.cfg.drift.start, "s").astype(np.int64))
        if drift_s >= self.cal.end_s - 3 * DAY:
            return
        rng = self.world.rngs.get("fraud.wallet_provisioning")
        per_10k = self.cfg.drift.wallet_fraud_incidents_per_10k_customers
        n_incidents = round(per_10k * len(self.world.customers) / 10_000)
        if n_incidents == 0:
            return
        # Linear ramp-up: the density of incident times grows from 0 at the drift start.
        span = self.cal.end_s - 3 * DAY - drift_s
        t_day = (drift_s + np.sqrt(rng.random(n_incidents)) * span).astype(np.int64)
        cards = self._victims(rng, t_day)
        customers = self.card_customer[cards]
        t0 = np.maximum(self._retime(rng, t_day, self.home[customers], LEGIT_HOUR_WEIGHTS), drift_s)
        devices = self.world.devices.add(rng, n_incidents, WALLET_FRAUD_DEVICES, fraud=True)
        for t, card, customer, device in zip(t0, cards, customers, devices, strict=True):
            n = int(rng.integers(2, 7))
            ts = self._sequence(rng, t, n, mean_gap=5 * HOUR, min_gap=10 * MINUTE)
            merch = self._merchants(
                rng,
                n,
                {"electronics": 0.3, "grocery": 0.2, "fuel": 0.15, "luxury": 0.15, "fashion": 0.2},
                online=False,
                country=int(self.home[customer]),
            )
            amount = rng.lognormal(np.log(160), 0.7, n)
            self._emit(
                DRIFT_SCENARIO,
                self._new_incident(),
                customer,
                card,
                merch,
                device,
                ts,
                amount,
                self.m_country[merch[0]],
                PM_WALLET,
            )

    # ----------------------------------------------------------------------------------------
    def generate(self) -> pd.DataFrame:
        mix = self.cfg.fraud.scenario_mix
        self.card_testing(self._n_incidents("card_testing", mix["card_testing"]))
        self.account_takeover(self._n_incidents("account_takeover", mix["account_takeover"]))
        self.card_cloning(self._n_incidents("card_cloning", mix["card_cloning"]))
        self.fraud_ring(self.target_txns * mix["fraud_ring"])
        self.merchant_compromise(self.target_txns * mix["merchant_compromise"])
        self.stealth(self._n_incidents("stealth", mix["stealth"]))
        self.wallet_provisioning()

        fraud = pd.DataFrame(
            {col: np.concatenate([p[col] for p in self._parts]) for col in TXN_COLUMNS}
        )
        before = len(fraud)
        fraud = fraud[(fraud["ts"] <= self.cal.end_s) & (fraud["ts"] >= self.cal.start_s)]
        log.info(
            "fraud: %s transactions (%s dropped outside the period)",
            len(fraud),
            before - len(fraud),
        )
        assert set(fraud["scenario"]) <= set(SCENARIOS) | {DRIFT_SCENARIO}
        return fraud.reset_index(drop=True)
