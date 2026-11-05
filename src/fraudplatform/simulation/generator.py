"""Simulator entry point: builds the world, the transaction streams, labels and the raw extract."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudplatform.config import SimulationConfig
from fraudplatform.simulation.defects import DefectInjector
from fraudplatform.simulation.entities import World, build_world
from fraudplatform.simulation.fraud import FraudGenerator
from fraudplatform.simulation.labels import generate_labels
from fraudplatform.simulation.legit import LEGIT, generate_legit
from fraudplatform.simulation.randomness import DAY, RngFactory, random_ids, to_date, to_iso
from fraudplatform.simulation.reference import (
    CARD_TYPES,
    CATEGORY_NAMES,
    COUNTRIES,
    COUNTRY_CODES,
    COUNTRY_CURRENCY,
    CURRENCIES,
    PAYMENT_METHODS,
    SEGMENT_NAMES,
)

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    scale: str
    seed: int
    simulator_version: str
    raw: dict[str, pd.DataFrame]  # tables as delivered by the "source system" (with defects)
    reference: dict[str, pd.DataFrame]  # trusted master data: countries, currencies, fx rates
    ground_truth: pd.DataFrame  # simulator-only: true label and scenario per transaction
    dq_defects: pd.DataFrame  # simulator-only: every injected data quality defect
    clean_transactions: pd.DataFrame  # typed transactions before defect injection (for checks)


def generate_fx(world: World, rng) -> pd.DataFrame:
    """Daily EUR rates: geometric random walk with optional drift, one row per currency and day."""
    cal = world.cal
    days = cal.days
    t_years = np.arange(len(days)) / 365.25
    frames = []
    for cur in CURRENCIES:
        shocks = rng.normal(0, cur.daily_vol, len(days))
        log_rate = np.log(cur.eur_per_unit) + cur.annual_drift * t_years + np.cumsum(shocks)
        frames.append(
            pd.DataFrame(
                {
                    "currency_code": cur.code,
                    "rate_date": np.datetime_as_string(days),
                    "eur_per_unit": np.round(np.exp(log_rate), 6),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _local_amounts(txns: pd.DataFrame, world: World, fx: pd.DataFrame):
    """Convert EUR amounts to the merchant's local currency using that day's rate."""
    m_country = world.merchants["country"].to_numpy()[txns["merchant"].to_numpy()]
    currency = COUNTRY_CURRENCY[m_country]
    day = world.cal.day_index(txns["ts"].to_numpy())
    rates = fx["eur_per_unit"].to_numpy().reshape(len(CURRENCIES), -1)
    amount = np.round(txns["amount_eur"].to_numpy() / rates[currency, day], 2)
    return np.maximum(amount, 0.01), np.array([c.code for c in CURRENCIES])[currency]


def _reference_tables(fx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    countries = pd.DataFrame(
        {
            "country_code": [c.code for c in COUNTRIES],
            "country_name": [c.name for c in COUNTRIES],
            "region": [c.region for c in COUNTRIES],
            "currency_code": [c.currency for c in COUNTRIES],
            "centroid_lat": [c.lat for c in COUNTRIES],
            "centroid_lon": [c.lon for c in COUNTRIES],
            "utc_offset_hours": [c.utc_offset for c in COUNTRIES],
        }
    )
    currencies = pd.DataFrame({"currency_code": [c.code for c in CURRENCIES]})
    return {"currencies": currencies, "countries": countries, "fx_rates": fx}


def simulate(cfg: SimulationConfig, scale: str, seed: int | None = None) -> SimulationResult:
    seed = cfg.seed if seed is None else seed
    rngs = RngFactory(seed)
    world = build_world(cfg, scale, rngs)
    log.info(
        "world: %s customers, %s cards, %s merchants",
        len(world.customers),
        len(world.cards),
        len(world.merchants),
    )

    legit = generate_legit(world)
    fraud = FraudGenerator(world, legit).generate()
    txns = pd.concat([legit, fraud], ignore_index=True)
    txns = txns.sort_values(["ts", "card"], kind="stable").reset_index(drop=True)
    txns["transaction_id"] = random_ids(rngs.get("ids.transactions"), len(txns), "TX", 16)

    labels = generate_labels(txns, cfg, rngs.get("labels"))
    fx = generate_fx(world, rngs.get("fx"))
    amount, currency = _local_amounts(txns, world, fx)

    # Devices only exist from their first use; their country is the country of that first use.
    used = txns[txns["device"] >= 0]
    first_use = used.groupby("device").agg(first_seen_s=("ts", "min"))
    first_row = used.sort_values("ts", kind="stable").drop_duplicates("device")
    first_use["country"] = first_row.set_index("device")["country"]
    device_ids = random_ids(rngs.get("ids.devices"), len(world.devices), "D", 12)

    customers, cards, merchants = world.customers, world.cards, world.merchants
    customer_ids = np.array([f"C{i:07d}" for i in range(len(customers))])
    card_ids = np.array([f"K{i:08d}" for i in range(len(cards))])
    merchant_ids = np.array([f"M{i:07d}" for i in range(len(merchants))])

    dev = txns["device"].to_numpy()
    clean = pd.DataFrame(
        {
            "transaction_id": txns["transaction_id"],
            "customer_id": customer_ids[txns["customer"]],
            "card_id": card_ids[txns["card"]],
            "merchant_id": merchant_ids[txns["merchant"]],
            "device_id": np.where(dev >= 0, device_ids[np.maximum(dev, 0)], ""),
            "ts_s": txns["ts"],
            "amount": amount,
            "currency": currency,
            "amount_eur": txns["amount_eur"],
            "country": COUNTRY_CODES[txns["country"]],
            "payment_method": PAYMENT_METHODS[txns["payment_method"]],
            "is_fraud_true": txns["scenario"] != LEGIT,
            "scenario": txns["scenario"],
        }
    )

    raw_txns = pd.DataFrame(
        {
            "transaction_id": clean["transaction_id"],
            "customer_id": clean["customer_id"],
            "card_id": clean["card_id"],
            "merchant_id": clean["merchant_id"],
            "device_id": clean["device_id"],
            "timestamp": to_iso(clean["ts_s"].to_numpy()),
            "amount": np.char.mod("%.2f", amount),
            "currency": currency,
            "country": clean["country"],
            "payment_method": clean["payment_method"],
        }
    )
    raw_customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "date_of_birth": to_date(customers["dob_s"].to_numpy()),
            "country": COUNTRY_CODES[customers["country"]],
            "account_creation_date": to_iso(customers["created_s"].to_numpy()),
            "customer_segment": SEGMENT_NAMES[customers["segment"]],
        }
    )
    raw_cards = pd.DataFrame(
        {
            "card_id": card_ids,
            "customer_id": customer_ids[cards["customer"]],
            "card_type": CARD_TYPES[cards["card_type"]],
            "card_creation_date": to_iso(cards["issued_s"].to_numpy()),
        }
    )
    raw_merchants = pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "merchant_category": CATEGORY_NAMES[merchants["category"]],
            "channel": np.where(merchants["online"], "online", "physical"),
            "country": COUNTRY_CODES[merchants["country"]],
            "risk_level": merchants["risk_level"],
            "onboarding_date": to_iso(merchants["onboarded_s"].to_numpy()),
        }
    )
    used_idx = first_use.index.to_numpy()
    raw_devices = pd.DataFrame(
        {
            "device_id": device_ids[used_idx],
            "device_type": np.array(world.devices.device_type)[used_idx],
            "operating_system": np.array(world.devices.os)[used_idx],
            "country": COUNTRY_CODES[first_use["country"].to_numpy()],
            "first_seen_at": to_iso(first_use["first_seen_s"].to_numpy()),
        }
    )
    raw_labels = pd.DataFrame(
        {
            "transaction_id": clean["transaction_id"].to_numpy()[labels["row"].to_numpy()],
            "is_fraud": np.where(labels["is_fraud"], "true", "false"),
            "reported_at": to_iso(labels["reported_s"].to_numpy()),
            "label_source": labels["label_source"],
        }
    )

    injector = DefectInjector(cfg.defects, rngs.get("defects"))
    card_issued = pd.Series(cards["issued_s"].to_numpy(), index=card_ids)
    raw = {
        "customers": injector.customers(raw_customers),
        "cards": injector.cards(raw_cards),
        "merchants": raw_merchants,
        "devices": raw_devices,
        "transactions": injector.transactions(raw_txns, card_issued, raw_cards),
        "fraud_labels": raw_labels,
    }

    ground_truth = pd.DataFrame(
        {
            "transaction_id": clean["transaction_id"],
            "is_fraud_true": clean["is_fraud_true"],
            "scenario": clean["scenario"],
            "incident_id": np.where(txns["incident"] >= 0, txns["incident"], -1),
        }
    )
    log.info(
        "done: %s transactions, %s fraud (%.3f%%), %s labels",
        len(clean),
        int(clean["is_fraud_true"].sum()),
        100 * clean["is_fraud_true"].mean(),
        len(raw_labels),
    )
    assert (labels["reported_s"].to_numpy() - txns["ts"].to_numpy()[labels["row"]] > 0).all()
    assert (
        labels["reported_s"].to_numpy() - txns["ts"].to_numpy()[labels["row"]]
        <= cfg.labels.max_delay_days * DAY
    ).all()

    return SimulationResult(
        scale=scale,
        seed=seed,
        simulator_version=cfg.simulator_version,
        raw=raw,
        reference=_reference_tables(fx),
        ground_truth=ground_truth,
        dq_defects=injector.defect_log(),
        clean_transactions=clean,
    )
