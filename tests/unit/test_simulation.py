"""Invariants of the simulated dataset.

The *clean* transactions (before defect injection) must satisfy every business rule; the raw
extract must contain exactly the defects that were logged.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraudplatform.config import DRIFT_SCENARIO, SCENARIOS, SimulationConfig
from fraudplatform.simulation.generator import simulate
from fraudplatform.simulation.randomness import DAY, iso_to_epoch, random_ids
from fraudplatform.simulation.writer import write_dataset

CARD_PRESENT = ("chip_pin", "contactless")


def _period_s(cfg: SimulationConfig, scale: str) -> tuple[int, int]:
    sc = cfg.scale(scale)
    start = int(pd.Timestamp(sc.start, tz="UTC").timestamp())
    end = int(pd.Timestamp(sc.end, tz="UTC").timestamp()) + DAY - 1
    return start, end


# ------------------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------------------
def test_same_seed_produces_identical_data(sim_config, tiny):
    again = simulate(sim_config, "tiny")
    for name, table in tiny.raw.items():
        pd.testing.assert_frame_equal(table, again.raw[name], obj=name)


def test_different_seed_produces_different_data(sim_config, tiny):
    other = simulate(sim_config, "tiny", seed=7)
    assert not tiny.raw["transactions"]["amount"].equals(other.raw["transactions"]["amount"])


def test_written_dataset_hash_is_reproducible(tiny, tmp_path):
    first = json.loads((write_dataset(tiny, tmp_path / "a") / "manifest.json").read_text())
    second = json.loads((write_dataset(tiny, tmp_path / "b") / "manifest.json").read_text())
    assert first["dataset_hash"] == second["dataset_hash"]
    assert first["files"]["raw/transactions.csv"]["rows"] == len(tiny.raw["transactions"])


# ------------------------------------------------------------------------------------------
# Clean data satisfies the business rules
# ------------------------------------------------------------------------------------------
def test_identifiers_are_unique(small):
    clean = small.clean_transactions
    assert clean["transaction_id"].is_unique
    assert small.ground_truth["transaction_id"].is_unique


def test_identifiers_do_not_encode_time(small):
    """Random IDs: sorting by ID must not reproduce the time order (no leakage through IDs)."""
    clean = small.clean_transactions.sort_values("transaction_id")
    rank_corr = np.corrcoef(np.arange(len(clean)), clean["ts_s"].rank().to_numpy())[0, 1]
    assert abs(rank_corr) < 0.05


def test_referential_integrity(small):
    clean = small.clean_transactions
    customers = small.raw["customers"].drop_duplicates("customer_id")
    cards = small.raw["cards"]
    cards = cards[cards["customer_id"].isin(customers["customer_id"])]
    merchants = small.raw["merchants"]
    devices = small.raw["devices"]

    assert clean["customer_id"].isin(customers["customer_id"]).all()
    assert clean["merchant_id"].isin(merchants["merchant_id"]).all()
    owner = cards.set_index("card_id")["customer_id"]
    assert (owner.loc[clean["card_id"]].to_numpy() == clean["customer_id"].to_numpy()).all()
    with_device = clean[clean["device_id"] != ""]
    assert with_device["device_id"].isin(devices["device_id"]).all()


def test_amounts_are_positive(small):
    clean = small.clean_transactions
    assert (clean["amount"] > 0).all()
    assert (clean["amount_eur"] > 0).all()


def test_transactions_inside_simulation_period(sim_config, small):
    start, end = _period_s(sim_config, "small")
    assert small.clean_transactions["ts_s"].between(start, end).all()


def test_device_present_exactly_for_card_not_present(small):
    clean = small.clean_transactions
    card_present = clean["payment_method"].isin(CARD_PRESENT)
    assert (clean.loc[card_present, "device_id"] == "").all()
    assert (clean.loc[~card_present, "device_id"] != "").all()


def test_no_transaction_before_card_issue_or_account_creation(small):
    clean = small.clean_transactions
    cards = small.raw["cards"].set_index("card_id")
    issued = iso_to_epoch(cards.loc[clean["card_id"], "card_creation_date"])
    assert (clean["ts_s"].to_numpy() >= issued).all()

    customers = small.raw["customers"].drop_duplicates("customer_id").set_index("customer_id")
    created = iso_to_epoch(customers.loc[clean["customer_id"], "account_creation_date"])
    assert (clean["ts_s"].to_numpy() >= created).all()


def test_device_first_seen_is_first_use(small):
    clean = small.clean_transactions
    first_use = clean[clean["device_id"] != ""].groupby("device_id")["ts_s"].min()
    devices = small.raw["devices"].set_index("device_id")
    first_seen = pd.Series(iso_to_epoch(devices["first_seen_at"]), index=devices.index)
    assert (first_seen.loc[first_use.index] == first_use).all()


# ------------------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------------------
def test_labels_arrive_after_transaction_within_window(sim_config, small):
    labels = small.raw["fraud_labels"]
    ts = small.clean_transactions.set_index("transaction_id").loc[labels["transaction_id"], "ts_s"]
    delay = iso_to_epoch(labels["reported_at"]) - ts.to_numpy()
    assert (delay > 0).all()
    assert (delay <= sim_config.labels.max_delay_days * DAY).all()


def test_labels_are_noisy_like_real_chargebacks(small):
    """Some fraud is never labelled and some legit is labelled as fraud: labels != truth."""
    labelled = small.raw["fraud_labels"]
    labelled = set(labelled.loc[labelled["is_fraud"] == "true", "transaction_id"])
    truth = small.ground_truth
    true_fraud = set(truth.loc[truth["is_fraud_true"], "transaction_id"])
    assert true_fraud - labelled, "expected unreported fraud"
    assert labelled - true_fraud, "expected friendly fraud (legit labelled as fraud)"


# ------------------------------------------------------------------------------------------
# Fraud generation
# ------------------------------------------------------------------------------------------
def test_fraud_rate_is_a_minority_class_in_target_range(small):
    rate = small.ground_truth["is_fraud_true"].mean()
    assert 0.005 <= rate <= 0.015


def test_all_base_scenarios_are_generated(small):
    assert set(SCENARIOS) <= set(small.ground_truth["scenario"])


def test_drift_scenario_only_after_drift_start(sim_config):
    cfg = sim_config.model_copy(deep=True)
    cfg.drift.start = date(2025, 2, 1)
    result = simulate(cfg, "tiny")
    clean = result.clean_transactions
    drift = clean[clean["scenario"] == DRIFT_SCENARIO]
    assert len(drift) > 0
    assert (drift["ts_s"] >= int(pd.Timestamp("2025-02-01", tz="UTC").timestamp())).all()


def test_no_drift_scenario_when_drift_outside_period(small):
    assert DRIFT_SCENARIO not in set(small.ground_truth["scenario"])


def test_simulator_truth_is_not_in_the_raw_extract(small):
    for table in small.raw.values():
        assert not {"scenario", "is_fraud_true", "incident_id"} & set(table.columns)


# ------------------------------------------------------------------------------------------
# Injected data quality defects
# ------------------------------------------------------------------------------------------
def test_duplicate_ids_match_the_defect_log(small):
    raw = small.raw["transactions"]
    log = small.dq_defects
    dup_ids = raw["transaction_id"].value_counts()
    dup_ids = set(dup_ids[dup_ids > 1].index)
    logged = log[log["defect_type"].isin(["exact_duplicate", "duplicate_id_conflict"])]
    assert dup_ids == set(logged.loc[logged["table_name"] == "transactions", "record_key"])


def test_logged_invalid_amounts_are_invalid_in_raw(small):
    raw = small.raw["transactions"].drop_duplicates("transaction_id").set_index("transaction_id")
    keys = small.dq_defects.query("defect_type == 'invalid_amount'")["record_key"]
    amounts = pd.to_numeric(raw.loc[keys, "amount"].replace("", np.nan), errors="coerce")
    assert len(keys) > 0
    assert (amounts.isna() | (amounts <= 0)).all()


def test_unlogged_rows_are_clean(small):
    """Rows without a logged defect must equal the clean data (no accidental corruption)."""
    raw = small.raw["transactions"]
    logged = set(small.dq_defects.query("table_name == 'transactions'")["record_key"])
    untouched = raw[~raw["transaction_id"].isin(logged)].set_index("transaction_id")
    clean = small.clean_transactions.set_index("transaction_id").loc[untouched.index]
    assert np.allclose(untouched["amount"].astype(float), clean["amount"])
    assert (untouched["customer_id"] == clean["customer_id"]).all()
    assert (iso_to_epoch(untouched["timestamp"]) == clean["ts_s"].to_numpy()).all()


# ------------------------------------------------------------------------------------------
# Configuration and helpers
# ------------------------------------------------------------------------------------------
def test_config_rejects_scenario_mix_not_summing_to_one(sim_config):
    data = sim_config.model_dump()
    data["fraud"]["scenario_mix"]["stealth"] = 0.5
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate(data)


def test_random_ids_are_unique():
    ids = random_ids(np.random.default_rng(0), 50_000, "X", 6)
    assert len(set(ids)) == 50_000
    assert all(len(i) == 7 for i in ids)
