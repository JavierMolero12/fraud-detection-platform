"""The payment simulator against a real database and the real API, with the small demo bundle."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from fraudplatform.config import RiskLevels, ServingConfig, load_feature_config
from fraudplatform.db import load_dataset
from fraudplatform.decision.costs import CostModel
from fraudplatform.features.builder import build_features
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.serving.api import create_app as create_api
from fraudplatform.serving.batch import run_batch
from fraudplatform.serving.demo import build_demo_bundle
from fraudplatform.serving.scorer import Bundle, Scorer
from fraudplatform.serving.simulator import FEATURES, create_app
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration
RISK = RiskLevels(medium_from=0.01, high_from=0.25)
# Any period has a legitimate payment; the second condition matches nothing and is skipped.
CONDITIONS = ["g.scenario = 'legit'", "FALSE"]


@pytest.fixture(scope="module")
def api(conn, small, db_url, tmp_path_factory):
    """The scoring API, with batch scores of the last two loaded days to pick payments from."""
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    build_features(conn, load_feature_config())
    bundle_dir = build_demo_bundle(conn, tmp_path_factory.mktemp("bundle"))
    last = conn.execute("SELECT max(ts) FROM features.transaction_features").fetchone()[0].date()
    scorer = Scorer(Bundle.load(bundle_dir), RISK)
    out = tmp_path_factory.mktemp("batch") / "batch.csv"
    run_batch(conn, scorer, last - timedelta(days=1), last + timedelta(days=1), out)
    serving = ServingConfig(bundle_dir=bundle_dir, risk_levels=RISK)
    with TestClient(create_api(serving, db_url, pool_size=2)) as client:
        yield client


@pytest.fixture(scope="module")
def simulator(api, db_url):
    with TestClient(create_app(api, db_url, CONDITIONS)) as client:
        yield client


def _left_behind(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM scoring.predictions WHERE transaction_id LIKE 'ui-%'"
    ).fetchone()[0]


def test_the_page_lists_the_payments_without_their_truth(simulator):
    assert simulator.get("/").text.startswith("<!doctype html>")
    cases = simulator.get("/api/cases").json()
    assert len(cases) == 1  # the condition that matches nothing is skipped
    assert set(cases[0]) == {"id", "payment", "merchant_category"}
    assert "transaction_id" not in cases[0]["payment"]  # each send gets a new one


def test_a_send_shows_the_batch_score_its_costs_and_leaves_no_prediction(conn, simulator, api):
    r = simulator.post("/api/score", json={"case": 0}).json()
    sent = r["request"]
    assert r["status"] == 200
    assert sent["transaction_id"].startswith("ui-")
    assert r["response"]["transaction_id"] == sent["transaction_id"]

    batch = conn.execute(
        "SELECT p.fraud_probability, p.decision FROM scoring.predictions p "
        "JOIN core.transactions t USING (transaction_id) "
        "WHERE p.source = 'batch' AND t.card_id = %s AND t.ts = %s",
        (sent["card_id"], sent["timestamp"]),
    ).fetchone()
    q = r["response"]["fraud_probability"]
    assert (q, r["response"]["decision"]) == (batch[0], batch[1])

    costs = CostModel(**api.get("/model").json()["costs"]).expected([q], [r["amount_eur"]])[0]
    assert [r["costs"][k] for k in ("approve", "review", "block")] == pytest.approx(costs)
    assert [f["key"] for f in r["features"]] == [key for key, _ in FEATURES]
    assert r["truth"] == {
        "is_fraud": False,
        "title": "Pago legítimo",
        "explanation": "Lo hizo el propio cliente.",
        "amount_changed": False,
    }
    assert _left_behind(conn) == 0


def test_a_changed_amount_is_sent_and_flagged(conn, simulator):
    r = simulator.post("/api/score", json={"case": 0, "amount": 12.5}).json()
    assert r["status"] == 200
    assert r["request"]["amount"] == 12.5
    assert r["truth"]["amount_changed"] is True
    assert _left_behind(conn) == 0


def test_an_unknown_case_is_404_and_a_bad_amount_422(simulator):
    assert simulator.post("/api/score", json={"case": 99}).status_code == 404
    assert simulator.post("/api/score", json={"case": 0, "amount": -1}).status_code == 422


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _over_budget(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"detail": "scoring exceeded its time budget"})


@pytest.mark.parametrize(
    ("handler", "status", "detail"),
    [
        (_refused, 502, "la API no responde en http://api"),
        (_over_budget, 503, "scoring exceeded its time budget"),
    ],
)
def test_api_failures_reach_the_page_as_a_status_and_a_reason(api, db_url, handler, status, detail):
    transport = httpx.MockTransport(handler)
    with (
        httpx.Client(base_url="http://api", transport=transport) as fake,
        TestClient(create_app(fake, db_url, CONDITIONS)) as simulator,
    ):
        r = simulator.post("/api/score", json={"case": 0}).json()
    assert (r["status"], r["detail"]) == (status, detail)
