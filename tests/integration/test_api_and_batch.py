"""API and batch scoring against a real database, with a small model trained on the test data.

The bundle here is NOT the Phase 8 model (that one is not in the repository): the demo bundle
(`fraud serve demo-bundle`), a quick HistGradientBoosting trained on the first 60% of the small
dataset and saved in the same format. What is tested is the serving machinery, not model
quality.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from fraudplatform.config import (
    RiskLevels,
    ServingConfig,
    load_feature_config,
)
from fraudplatform.db import load_dataset
from fraudplatform.eda.data import load_features
from fraudplatform.features.builder import build_features
from fraudplatform.features.catalog import FEATURE_NAMES
from fraudplatform.models.pipelines import prepare
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.serving.api import create_app
from fraudplatform.serving.batch import OUTPUT_COLUMNS, run_batch, score_period
from fraudplatform.serving.checks import check_api, check_passed, verify_bundle
from fraudplatform.serving.demo import build_demo_bundle
from fraudplatform.serving.online import score_payment
from fraudplatform.serving.online_features import IncomingTransaction, OnlineFeatureBuilder
from fraudplatform.serving.scorer import Bundle, Scorer
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration
RISK = RiskLevels(medium_from=0.01, high_from=0.25)


@pytest.fixture(scope="module")
def bundle_dir(conn, small, tmp_path_factory):
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    build_features(conn, load_feature_config())
    return build_demo_bundle(conn, tmp_path_factory.mktemp("bundle"))


@pytest.fixture(scope="module")
def client(bundle_dir, db_url):
    app = create_app(ServingConfig(bundle_dir=bundle_dir, risk_levels=RISK), db_url, pool_size=4)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scorer(bundle_dir):
    return Scorer(Bundle.load(bundle_dir), RISK)


def _stored(conn, where: str, limit: int, params=()) -> list[dict]:
    cur = conn.execute(
        "SELECT transaction_id, customer_id, card_id, merchant_id, ts, amount, currency_code, "
        f"country_code, payment_method, device_id FROM core.transactions WHERE {where} "
        f"ORDER BY md5(transaction_id) LIMIT {limit}",
        params,
    )
    return [dict(zip([d.name for d in cur.description], r, strict=True)) for r in cur.fetchall()]


def _body(row: dict, prefix: str = "api-") -> dict:
    body = {
        "transaction_id": prefix + row["transaction_id"],
        "customer_id": row["customer_id"],
        "card_id": row["card_id"],
        "merchant_id": row["merchant_id"],
        "timestamp": row["ts"].isoformat(),
        "amount": float(row["amount"]),
        "currency": row["currency_code"],
        "country": row["country_code"],
        "payment_method": row["payment_method"],
    }
    if row["device_id"] is not None:
        body["device_id"] = row["device_id"]
    return body


def _busiest_days(conn, n: int) -> tuple[date, date]:
    start = conn.execute(
        "SELECT (ts AT TIME ZONE 'UTC')::date AS d FROM core.transactions "
        "GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1"
    ).fetchone()[0]
    return start, start + timedelta(days=n)


def test_api_gives_the_same_probabilities_and_decisions_as_batch(conn, client, scorer):
    start, end = _busiest_days(conn, 2)
    rows = _stored(conn, "ts >= %s AND ts < %s", 150, (start, end))
    responses = {}
    for row in rows:
        r = client.post("/predict", json=_body(row))
        assert r.status_code == 200, r.text
        responses[row["transaction_id"]] = r.json()

    batch, _ = score_period(conn, scorer, start, end)
    batch = batch.set_index("transaction_id").loc[list(responses)]
    api = pd.DataFrame(responses.values()).set_index(batch.index)
    np.testing.assert_allclose(api["fraud_probability"], batch["fraud_probability"], rtol=1e-12)
    assert (api["decision"] == batch["decision"]).all()
    assert (api["risk_level"] == batch["risk_level"]).all()
    assert set(api["model_version"]) == {scorer.bundle.model_version}


def test_every_api_decision_is_logged_with_its_features(conn, client, scorer):
    row = _stored(conn, "payment_method = 'ecommerce'", 1)[0]
    body = _body(row, prefix="log-")
    response = client.post("/predict", json=body).json()
    logged = conn.execute(
        "SELECT source, model_version, policy_version, fraud_probability, decision, "
        "amount_eur, transaction_ts, latency_ms, features "
        "FROM scoring.predictions WHERE transaction_id = %s",
        (body["transaction_id"],),
    ).fetchall()
    assert len(logged) == 1
    source, model_v, policy_v, p, decision, amount_eur, ts, latency, features = logged[0]
    assert (source, model_v, policy_v) == (
        "api",
        *(response[k] for k in ("model_version", "policy_version")),
    )
    assert p == response["fraud_probability"] and decision == response["decision"]
    assert ts == row["ts"] and latency > 0
    assert set(features) == set(FEATURE_NAMES)  # jsonb does not keep key order
    txn = IncomingTransaction(
        *[row[k] for k in ("customer_id", "card_id", "merchant_id", "ts")],
        float(row["amount"]),
        row["currency_code"],
        row["country_code"],
        row["payment_method"],
        row["device_id"],
    )
    frame = OnlineFeatureBuilder(load_feature_config()).frame(conn, txn)
    assert features["amount_eur"] == frame["amount_eur"].iloc[0] == float(amount_eur)


def test_a_retried_request_returns_the_logged_decision(conn, client):
    body = _body(_stored(conn, "true", 1)[0], prefix="retry-")
    first = client.post("/predict", json=body).json()
    second = client.post("/predict", json=body | {"amount": 9999.0}).json()
    assert first == second
    count = conn.execute(
        "SELECT count(*) FROM scoring.predictions WHERE transaction_id = %s",
        (body["transaction_id"],),
    ).fetchone()[0]
    assert count == 1


def test_a_query_over_the_time_budget_gets_503_not_a_wait(conn, bundle_dir, db_url):
    # 1 ms: the feature query cannot finish in time, so PostgreSQL cancels it.
    serving = ServingConfig(bundle_dir=bundle_dir, risk_levels=RISK, statement_timeout_ms=1)
    body = _body(_stored(conn, "true", 1)[0], prefix="slow-")
    with TestClient(create_app(serving, db_url, pool_size=1)) as slow:
        r = slow.post("/predict", json=body)
    assert r.status_code == 503
    assert r.json()["detail"] == "scoring exceeded its time budget"


def test_unknown_entities_and_currencies_are_rejected(conn, client):
    body = _body(_stored(conn, "true", 1)[0], prefix="bad-")
    for change in (
        {"customer_id": "C_UNKNOWN"},
        {"merchant_id": "M_UNKNOWN"},
        {"card_id": "K_UNKNOWN"},
        {"country": "ZZ"},
        {"currency": "JPY"},
    ):
        r = client.post("/predict", json=body | change)
        assert r.status_code == 422, (change, r.text)
    assert client.post("/predict", json=body | {"amount": -1}).status_code == 422


def test_stale_exchange_rates_are_refused_but_eur_needs_none(conn, client):
    row = _stored(conn, "payment_method = 'ecommerce'", 1)[0]
    last_rate = conn.execute("SELECT max(rate_date) FROM core.fx_rates").fetchone()[0]
    late = datetime.combine(last_rate + timedelta(days=30), datetime.min.time(), UTC)
    body = _body(row, prefix="fx-") | {"timestamp": late.isoformat()}
    assert client.post("/predict", json=body | {"currency": "USD"}).status_code == 422
    assert client.post("/predict", json=body | {"currency": "EUR"}).status_code == 200


def test_a_device_never_seen_before(conn, client):
    row = _stored(conn, "payment_method = 'ecommerce' AND device_id IS NOT NULL", 1)[0]
    body = _body(row, prefix="newdev-") | {
        "device_id": "D_NEVER_SEEN",
        "device_type": "desktop",
        "operating_system": "linux",
    }
    assert client.post("/predict", json=body).status_code == 200
    features = conn.execute(
        "SELECT features FROM scoring.predictions WHERE transaction_id = %s",
        (body["transaction_id"],),
    ).fetchone()[0]
    assert features["device_type"] == "desktop" and features["operating_system"] == "linux"
    assert features["device_age_days"] == 0
    assert features["device_new_for_customer"] is True
    assert features["device_txn_count_1h"] == 0


def test_health_and_model_endpoints(client, scorer):
    assert client.get("/health").json() == {
        "status": "ok",
        "model_version": scorer.bundle.model_version,
    }
    info = client.get("/model").json()
    assert info["policy_version"] == scorer.bundle.policy_version
    assert info["review_capacity_per_day"] == 60
    assert info["risk_levels"] == {"medium_from": 0.01, "high_from": 0.25}


def test_step_timings_cover_the_whole_path(conn, db_url, scorer):
    row = _stored(conn, "payment_method = 'ecommerce'", 1)[0]
    txn = IncomingTransaction(
        *[row[k] for k in ("customer_id", "card_id", "merchant_id", "ts")],
        float(row["amount"]),
        row["currency_code"],
        row["country_code"],
        row["payment_method"],
        row["device_id"],
    )
    timings: dict[str, float] = {}
    with psycopg.connect(db_url, autocommit=True) as c:
        txid = "timing-" + row["transaction_id"]
        score_payment(
            c, scorer, OnlineFeatureBuilder(load_feature_config()), txid, txn, timings=timings
        )
        logged_ms = c.execute(
            "SELECT latency_ms FROM scoring.predictions WHERE transaction_id = %s", (txid,)
        ).fetchone()[0]
    assert list(timings) == [
        "idempotency_check",
        "exchange_rate",
        "features_sql",
        "prepare",
        "predict_calibrate",
        "decide",
        "write_log",
    ]
    assert all(ms >= 0 for ms in timings.values())
    # the logged latency is the time to the decision: every step except writing the log
    to_decision = sum(ms for step, ms in timings.items() if step != "write_log")
    assert logged_ms == pytest.approx(to_decision, rel=0.05, abs=0.5)


class _FixedProbability(Scorer):
    def probability(self, frame):
        return np.full(len(frame), 0.15)


def test_review_capacity_holds_under_concurrent_requests(conn, db_url, bundle_dir):
    """10 simultaneous payments that all deserve a review, 3 slots: exactly 3 reviews.

    q = 0.15 on 100 EUR: review is the cheapest action, block the second best.
    """
    bundle = dataclasses.replace(Bundle.load(bundle_dir), capacity=3)
    scorer = _FixedProbability(bundle, RISK)
    features = OnlineFeatureBuilder(load_feature_config())
    best, fallback = scorer.preferred_actions(0.15, 100.0)
    assert (best, fallback) == (1, 2)  # review, block

    row = _stored(conn, "payment_method = 'ecommerce' AND device_id IS NOT NULL", 1)[0]
    day = datetime(2030, 1, 1, 12, tzinfo=UTC)  # a day nobody else in this module scores
    barrier = threading.Barrier(10)
    decisions = []

    def pay(i: int, when: datetime) -> None:
        txn = IncomingTransaction(
            row["customer_id"],
            row["card_id"],
            row["merchant_id"],
            when,
            100.0,
            "EUR",
            row["country_code"],
            "ecommerce",
            row["device_id"],
        )
        with psycopg.connect(db_url, autocommit=True) as c:
            barrier.wait()
            decisions.append(score_payment(c, scorer, features, f"cap-{when:%j}-{i}", txn).decision)

    threads = [threading.Thread(target=pay, args=(i, day)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(decisions) == ["block"] * 7 + ["review"] * 3

    # the next day has its own slots
    barrier = threading.Barrier(1)
    decisions.clear()
    pay(0, day + timedelta(days=1))
    assert decisions == ["review"]


def test_batch_output_and_rerun_replaces_earlier_rows(conn, scorer, tmp_path):
    start, end = _busiest_days(conn, 3)
    expected_rows = conn.execute(
        "SELECT count(*) FROM features.transaction_features WHERE ts >= %s AND ts < %s",
        (start, end),
    ).fetchone()[0]
    count_sql = (
        "SELECT count(*) FROM scoring.predictions WHERE source = 'batch' "
        "AND transaction_ts >= %s AND transaction_ts < %s"
    )
    for _ in range(2):
        summary = run_batch(conn, scorer, start, end, tmp_path / "out.csv", tmp_path / "s.json")
        assert conn.execute(count_sql, (start, end)).fetchone()[0] == expected_rows
    out = pd.read_csv(tmp_path / "out.csv")
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == summary["rows"] == expected_rows
    assert out["transaction_id"].is_unique
    assert out["fraud_probability"].between(0, 1).all()
    assert set(out["decision"]) <= {"approve", "review", "block"}
    assert json.loads((tmp_path / "s.json").read_text())["rows"] == expected_rows


def test_check_api_passes_against_the_served_bundle_and_cleans_up(conn, client, scorer):
    result = check_api(conn, "http://testserver", scorer, payments=30, client=client)
    assert check_passed(result), result
    assert result["payments"] == 30
    leftovers = conn.execute(
        "SELECT count(*) FROM scoring.predictions WHERE transaction_id LIKE 'check-%'"
    ).fetchone()[0]
    assert leftovers == 0


def test_check_api_fails_when_the_api_serves_another_model(conn, client, bundle_dir, tmp_path):
    other = build_demo_bundle(conn, tmp_path / "other", max_iter=5)
    result = check_api(conn, "http://testserver", Scorer(Bundle.load(other), RISK), 20, client)
    assert result["served_model_version"] != result["local_model_version"]
    assert not check_passed(result)


def test_verify_bundle_tolerates_last_bit_rounding_but_not_a_different_model(
    conn, scorer, bundle_dir, tmp_path
):
    start, end = (date.fromisoformat(d) for d in scorer.bundle.policy["chosen_on"])
    df = prepare(load_features(conn, start, end, with_label=False))
    p_raw = scorer.bundle.model.predict_proba(df[scorer.bundle.columns])[:, 1]
    pd.DataFrame({"p_raw": p_raw}).to_parquet(bundle_dir / "validation_scores.parquet")
    exact = verify_bundle(conn, scorer, bundle_dir)
    assert exact["identical"] and exact["passed"] and exact["differing"] == 0

    def verify_with(saved):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        pd.DataFrame({"p_raw": saved}).to_parquet(folder / "validation_scores.parquet")
        return verify_bundle(conn, scorer, folder)

    one_ulp = p_raw.copy()
    one_ulp[0] = np.nextafter(one_ulp[0], 1)  # what another OS's exp() can do
    result = verify_with(one_ulp)
    assert not result["identical"] and result["passed"]
    assert result["differing"] == 1 and result["max_ulps"] == 1

    other_model = p_raw.copy()
    other_model[0] += 1e-9  # far beyond rounding
    assert not verify_with(other_model)["passed"]
