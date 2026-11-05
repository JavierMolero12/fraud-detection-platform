"""Payment simulator: a web page that sends stored payments to the scoring API, step by step.

`fraud serve simulator` serves it on http://127.0.0.1:8001; the API must be running. At start it
picks one payment of each kind from the batch scores of the simulated production period, with
the generator's truth. For each send the page shows the API's answer, the features the API
logged, the expected cost of each action and, on request, the truth, which the API never reads.
The amount can be changed before sending. The page's text is in Spanish.

Every prediction the API logs for the simulator (ids `ui-...`) is deleted right after its answer,
so a demo leaves scoring.predictions and the day's review count as they were.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from fraudplatform.decision.costs import CostModel
from fraudplatform.serving.bench import payload

PAGE = Path(__file__).with_name("simulator.html")
ID_PREFIX = "ui-"

# One example of each kind: a condition on the batch score (p) and the generator's truth (g).
CASES = [
    "g.scenario = 'legit' AND p.decision = 'approve' AND p.amount_eur BETWEEN 20 AND 60",
    "g.scenario = 'legit' AND p.decision = 'review' AND p.fraud_probability < 0.01 "
    "AND p.amount_eur > 2000",
    "g.scenario = 'account_takeover' AND p.decision = 'block' AND p.amount_eur > 300",
    "g.scenario = 'card_testing' AND p.decision = 'block' AND p.amount_eur < 5",
    "g.scenario = 'stealth' AND p.decision = 'approve'",
    "g.scenario = 'card_cloning' AND p.decision = 'block'",
    "g.scenario = 'fraud_ring' AND p.decision = 'block'",
    "g.scenario = 'wallet_provisioning' AND p.decision = 'block'",
    "g.scenario = 'wallet_provisioning' AND p.decision = 'approve'",
    "g.scenario = 'legit' AND p.decision = 'block'",
    "g.scenario = 'merchant_compromise' AND p.decision = 'block'",
    "g.scenario = 'legit' AND p.decision = 'approve' AND p.amount_eur > 500",
]

# What the page says about each scenario of the generator once the truth is revealed.
SCENARIOS = {
    "legit": ("Pago legítimo", "Lo hizo el propio cliente."),
    "account_takeover": (
        "Fraude: robo de cuenta",
        "Alguien tomó el control de la cuenta del cliente.",
    ),
    "card_testing": (
        "Fraude: prueba de tarjetas",
        "Micro-cargos para comprobar si una tarjeta robada funciona.",
    ),
    "stealth": (
        "Fraude sigiloso",
        "Imita el comportamiento normal del cliente: casi invisible por construcción.",
    ),
    "card_cloning": ("Fraude: tarjeta clonada", "Una copia de la tarjeta usada por otra persona."),
    "fraud_ring": (
        "Fraude: red organizada",
        "Varias tarjetas y cuentas que comparten dispositivos.",
    ),
    "wallet_provisioning": (
        "Fraude: alta en wallet",
        "Una tarjeta robada añadida a un móvil. Es el fraude nuevo de la deriva de marzo.",
    ),
    "merchant_compromise": (
        "Fraude: comercio comprometido",
        "Tarjetas robadas en un comercio y usadas después.",
    ),
}

# The 14 of the 50 features the page shows, with their labels.
FEATURES = [
    ("amount_eur", "Importe en euros"),
    ("payment_method", "Canal de pago"),
    ("merchant_category", "Tipo de comercio"),
    ("local_hour", "Hora local"),
    ("cross_border", "Fuera del país del cliente"),
    ("card_txn_count_1h", "Usos de la tarjeta en la última hora"),
    ("card_micro_count_1h", "Micro-cargos en la última hora"),
    ("card_txn_count_24h", "Usos de la tarjeta en 24 horas"),
    ("amount_to_cust_median_90d", "Importe / importe habitual del cliente"),
    ("device_new_for_customer", "Dispositivo nuevo para el cliente"),
    ("device_age_days", "Antigüedad del dispositivo (días)"),
    ("device_unique_customers_30d", "Clientes distintos en el dispositivo (30 días)"),
    ("new_country_for_customer_180d", "País nuevo para el cliente (180 días)"),
    ("merchant_fraud_rate_90d_smoothed", "Tasa de fraude del comercio (90 días)"),
]

# The first ten columns are bench.SAMPLE's, so bench.payload turns them into a request.
CASE = """
SELECT t.transaction_id, t.customer_id, t.card_id, t.merchant_id, t.ts, t.amount,
       t.currency_code, t.country_code, t.payment_method, t.device_id,
       g.scenario, g.is_fraud_true, m.merchant_category
FROM scoring.predictions p
JOIN sim.ground_truth g USING (transaction_id)
JOIN core.transactions t USING (transaction_id)
JOIN core.merchants m ON m.merchant_id = t.merchant_id
WHERE p.source = 'batch' AND {condition}
ORDER BY md5(t.transaction_id)
LIMIT 1
"""

DELETE_LOGGED = """
DELETE FROM scoring.predictions WHERE source = 'api' AND transaction_id = %s
RETURNING features, amount_eur
"""


def load_cases(conn: psycopg.Connection, conditions: list[str]) -> list[dict]:
    """The first batch-scored payment (in a fixed order) for each condition; none, skipped."""
    cases = []
    for condition in conditions:
        row = conn.execute(CASE.format(condition=condition)).fetchone()
        if row is None:
            continue
        request = payload(row[:10], prefix="")
        del request["transaction_id"]  # each send gets a new one
        scenario, is_fraud, category = row[10:]
        cases.append(
            {
                "id": len(cases),
                "payment": request,
                "merchant_category": category,
                "scenario": scenario,
                "is_fraud": bool(is_fraud),
            }
        )
    return cases


class ScoreRequest(BaseModel):
    case: int
    amount: float | None = Field(default=None, gt=0, le=1_000_000)


def _detail(response: httpx.Response):
    try:
        return response.json().get("detail")
    except ValueError:
        return response.text


def create_app(api: httpx.Client, db_url: str, conditions: list[str] = CASES) -> FastAPI:
    """`api`: a client whose base URL is the scoring API (tests pass the API's TestClient).

    One client for the whole run: creating one per request cost ~200 ms on Windows, which the
    page then showed as API time.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with psycopg.connect(db_url) as conn:
            app.state.cases = load_cases(conn, conditions)
        app.state.model = None  # GET /model of the API, read at the first answer
        yield

    app = FastAPI(title="Payment simulator", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/cases")
    def cases(request: Request) -> list[dict]:
        """The payments without their truth: the page reveals it after the answer."""
        return [
            {"id": c["id"], "payment": c["payment"], "merchant_category": c["merchant_category"]}
            for c in request.app.state.cases
        ]

    @app.post("/api/score")
    def score(body: ScoreRequest, request: Request) -> dict:
        state = request.app.state
        if not 0 <= body.case < len(state.cases):
            raise HTTPException(status_code=404, detail="unknown case")
        case = state.cases[body.case]
        sent = {"transaction_id": f"{ID_PREFIX}{uuid.uuid4().hex[:12]}", **case["payment"]}
        if body.amount is not None:
            sent["amount"] = body.amount
        start = time.perf_counter()
        try:
            response = api.post("/predict", json=sent)
        except httpx.TransportError:
            ms = (time.perf_counter() - start) * 1000
            detail = f"la API no responde en {str(api.base_url).rstrip('/')}"
            return {"status": 502, "detail": detail, "ms": ms, "request": sent}
        ms = (time.perf_counter() - start) * 1000
        if response.status_code != 200:
            return {
                "status": response.status_code,
                "detail": _detail(response),
                "ms": ms,
                "request": sent,
            }
        answer = response.json()
        with psycopg.connect(db_url, autocommit=True) as conn:
            features, amount_eur = conn.execute(DELETE_LOGGED, (sent["transaction_id"],)).fetchone()
        if state.model is None:
            state.model = api.get("/model").json()
        amount_eur = float(amount_eur)
        q = answer["fraud_probability"]
        expected = CostModel(**state.model["costs"]).expected([q], [amount_eur])[0]
        approve, review, block = (float(x) for x in expected)
        levels = state.model["risk_levels"]
        title, explanation = SCENARIOS[case["scenario"]]
        return {
            "status": 200,
            "ms": ms,
            "request": sent,
            "response": answer,
            "amount_eur": amount_eur,
            "features": [
                {"key": key, "label": label, "value": features.get(key)} for key, label in FEATURES
            ],
            "costs": {"approve": approve, "review": review, "block": block},
            "cost_model": state.model["costs"],
            "risk_levels": {"medium": levels["medium_from"], "high": levels["high_from"]},
            "truth": {
                "is_fraud": case["is_fraud"],
                "title": title,
                "explanation": explanation,
                "amount_changed": body.amount is not None
                and body.amount != case["payment"]["amount"],
            },
        }

    return app


def run_simulator(api_url: str, db_url: str, host: str, port: int) -> None:
    import logging

    import uvicorn

    logging.getLogger("httpx").setLevel(logging.WARNING)  # the API's tab already logs each request
    with httpx.Client(base_url=api_url, timeout=10) as api:
        uvicorn.run(create_app(api, db_url), host=host, port=port, log_level="warning")
