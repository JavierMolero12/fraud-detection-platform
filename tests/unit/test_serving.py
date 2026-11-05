"""Serving pieces that need no database: input validation, frame conversion, decisions."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraudplatform.config import (
    RiskLevels,
    load_cost_config,
    load_serving_config,
)
from fraudplatform.decision.costs import APPROVE, BLOCK, REVIEW, CostModel, bayes_actions
from fraudplatform.features.catalog import BOOLEAN, CATEGORICAL, FEATURE_NAMES
from fraudplatform.models.pipelines import model_columns
from fraudplatform.serving.api import TransactionRequest
from fraudplatform.serving.online import logged_features
from fraudplatform.serving.scorer import Bundle, Scorer

VALID = {
    "transaction_id": "T-1",
    "customer_id": "C0000001",
    "card_id": "K00000001",
    "merchant_id": "M0000001",
    "timestamp": "2026-03-01T10:00:00+01:00",
    "amount": 25.5,
    "currency": "EUR",
    "country": "ES",
    "payment_method": "ecommerce",
    "device_id": "D0000001",
}


def _bundle(capacity: int = 60) -> Bundle:
    costs = load_cost_config()
    return Bundle(
        model=None,
        policy={"calibration": {"method": "none"}},
        columns=model_columns("tree"),
        costs=CostModel.from_config(costs),
        capacity=capacity,
        model_version="m",
        policy_version="p",
    )


# --- request validation -------------------------------------------------------------------


def test_a_valid_request_is_accepted():
    req = TransactionRequest(**VALID)
    assert req.incoming().ts.utcoffset() == timedelta(hours=1)


@pytest.mark.parametrize(
    "change",
    [
        {"timestamp": "2026-03-01T10:00:00"},  # no time zone: ambiguous
        {"timestamp": (datetime.now(UTC) + timedelta(hours=1)).isoformat()},  # future
        {"amount": 0},
        {"amount": -3},
        {"currency": "eur"},
        {"country": "ESP"},
        {"payment_method": "cash"},
        {"transaction_id": ""},
        {"customer_id": "C 1; DROP TABLE"},
        {"unexpected": 1},  # extra fields are refused, not silently ignored
        {"payment_method": "chip_pin"},  # card present but a device is attached
        {"payment_method": "mobile_wallet", "device_id": None},
        {"device_id": None, "device_type": "mobile"},
        {"device_type": "smart_fridge"},
    ],
)
def test_invalid_requests_are_rejected(change):
    with pytest.raises(ValidationError):
        TransactionRequest(**(VALID | change))


def test_card_present_without_device_and_online_without_device_are_valid():
    TransactionRequest(**(VALID | {"payment_method": "contactless", "device_id": None}))
    TransactionRequest(**(VALID | {"device_id": None}))


# --- logged features ------------------------------------------------------------------------


def test_logged_features_are_plain_json_with_unknowns_as_null():
    row = {}
    for name in FEATURE_NAMES:
        row[name] = "x" if name in CATEGORICAL else (1.0 if name in BOOLEAN else 2)
    row |= {"is_night": 0.0, "device_new_for_customer": np.nan, "device_age_days": np.nan}
    values = logged_features(pd.Series(row))
    assert list(values) == list(FEATURE_NAMES)
    assert values["is_night"] is False and values["is_card_present"] is True
    assert values["device_new_for_customer"] is None and values["device_age_days"] is None
    assert values["payment_method"] == "x" and values["card_txn_count_1h"] == 2
    json.dumps(values, allow_nan=False)  # valid JSON: no NaN tokens


# --- risk levels and decisions --------------------------------------------------------------


def test_risk_levels_bands():
    scorer = Scorer(_bundle(), RiskLevels(medium_from=0.01, high_from=0.25))
    levels = scorer.risk_level([0.0, 0.0099, 0.01, 0.2499, 0.25, 1.0])
    assert levels.tolist() == ["low", "low", "medium", "medium", "high", "high"]


def test_risk_levels_must_be_ordered():
    with pytest.raises(ValidationError):
        RiskLevels(medium_from=0.3, high_from=0.2)


def test_online_decisions_equal_batch_decisions_when_capacity_does_not_bind():
    rng = np.random.default_rng(0)
    n = 5000
    q = rng.beta(0.3, 3, n)
    amount = np.exp(rng.normal(3.5, 1.5, n))
    scorer = Scorer(_bundle(capacity=n), RiskLevels(medium_from=0.01, high_from=0.25))
    batch = bayes_actions(q, amount, np.zeros(n), scorer.bundle.costs, n)
    online = [
        scorer.preferred_actions(float(a), float(b))[0] for a, b in zip(q, amount, strict=True)
    ]
    assert (np.array(online) == batch).all()
    assert {APPROVE, REVIEW, BLOCK} <= set(online)  # the sample exercises every action


def test_fallback_when_no_review_slot_is_the_best_of_approve_and_block():
    scorer = Scorer(_bundle(), RiskLevels(medium_from=0.01, high_from=0.25))
    costs = scorer.bundle.costs
    rng = np.random.default_rng(1)
    for q, amount in zip(rng.uniform(0, 1, 2000), np.exp(rng.normal(4, 1.5, 2000)), strict=True):
        best, fallback = scorer.preferred_actions(float(q), float(amount))
        expected = costs.expected([q], [amount])[0]
        assert fallback in (APPROVE, BLOCK)
        assert expected[fallback] == min(expected[APPROVE], expected[BLOCK])
        assert expected[best] == expected.min()


# --- configuration and bundle ---------------------------------------------------------------


def test_bundle_dir_can_be_overridden_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_BUNDLE_DIR", str(tmp_path))
    assert load_serving_config().bundle_dir == tmp_path


def test_missing_bundle_fails_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraud policy optimise"):
        Bundle.load(tmp_path)


def test_only_the_expected_cost_policy_is_served(tmp_path):
    (tmp_path / "model.skops").write_bytes(b"")
    (tmp_path / "policy.json").write_text(json.dumps({"policy": {"type": "thresholds"}}))
    with pytest.raises(ValueError, match="expected-cost"):
        Bundle.load(tmp_path)


# --- server launcher ------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux only: elsewhere Python reads the protocol from the descriptor, and several "
    "API workers only run on Linux anyway (decision 026)",
)
def test_the_listening_socket_is_declared_tcp_so_connections_get_tcp_nodelay():
    import socket

    from fraudplatform.serving.server import TcpConfig

    sock = TcpConfig("fraudplatform.serving.api:create_app", host="127.0.0.1", port=0).bind_socket()
    try:
        assert sock.proto == socket.IPPROTO_TCP  # asyncio sets TCP_NODELAY only in this case
        assert sock.getsockname()[1] > 0
    finally:
        sock.close()


# --- payment simulator ----------------------------------------------------------------------


def test_the_simulator_explains_every_scenario_and_shows_only_real_features():
    from fraudplatform.config import DRIFT_SCENARIO, SCENARIOS
    from fraudplatform.serving import simulator

    assert set(simulator.SCENARIOS) == {"legit", *SCENARIOS, DRIFT_SCENARIO}
    assert {key for key, _ in simulator.FEATURES} <= set(FEATURE_NAMES)
    assert simulator.PAGE.is_file()
