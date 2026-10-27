"""Real-time scoring API.

POST /predict   score one payment at authorisation time
GET  /health    liveness + database reachability + model loaded
GET  /model     what is deployed: versions, policy, calibration, training period, lineage

Run with `fraud serve api` (uvicorn). Endpoints are plain `def`: FastAPI runs them in its
thread pool, and each request takes its own connection from a psycopg pool. Every statement on
those connections has a time budget (serving.yaml): past it, /predict answers 503 instead of
holding the authorisation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import psycopg
from fastapi import FastAPI, HTTPException, Request
from psycopg_pool import ConnectionPool, PoolTimeout
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from threadpoolctl import threadpool_limits

from fraudplatform.config import ServingConfig, load_feature_config, load_serving_config
from fraudplatform.db import database_url
from fraudplatform.serving.online import score_payment
from fraudplatform.serving.online_features import (
    IncomingTransaction,
    OnlineFeatureBuilder,
    UnknownEntity,
)
from fraudplatform.serving.scorer import Bundle, Scorer

log = logging.getLogger(__name__)

CARD_PRESENT = ("chip_pin", "contactless")
MAX_CLOCK_SKEW = timedelta(minutes=5)
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")]


class TransactionRequest(BaseModel):
    """A payment as the authorisation system sends it (before ingestion into core)."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: Identifier
    customer_id: Identifier
    card_id: Identifier
    merchant_id: Identifier
    timestamp: AwareDatetime
    amount: float = Field(gt=0, le=1_000_000, description="In the payment currency")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    country: str = Field(
        pattern=r"^[A-Z]{2}$", description="Terminal country (card present) or IP country"
    )
    payment_method: Literal["chip_pin", "contactless", "ecommerce", "mobile_wallet"]
    device_id: Identifier | None = None
    device_type: Literal["mobile", "desktop", "tablet"] | None = None
    operating_system: Literal["android", "ios", "ipados", "windows", "macos", "linux"] | None = None

    @field_validator("timestamp")
    @classmethod
    def _not_in_the_future(cls, ts: datetime) -> datetime:
        if ts > datetime.now(UTC) + MAX_CLOCK_SKEW:
            raise ValueError("timestamp is in the future")
        return ts

    @model_validator(mode="after")
    def _device_matches_channel(self) -> TransactionRequest:
        has_device = any((self.device_id, self.device_type, self.operating_system))
        if self.payment_method in CARD_PRESENT and has_device:
            raise ValueError("a card-present payment does not come from a device")
        if self.payment_method == "mobile_wallet" and self.device_id is None:
            raise ValueError("a mobile wallet payment needs device_id")
        if has_device and self.device_id is None:
            raise ValueError("device_type / operating_system given without device_id")
        return self

    def incoming(self) -> IncomingTransaction:
        return IncomingTransaction(
            customer_id=self.customer_id,
            card_id=self.card_id,
            merchant_id=self.merchant_id,
            ts=self.timestamp,
            amount=self.amount,
            currency=self.currency,
            country=self.country,
            payment_method=self.payment_method,
            device_id=self.device_id,
            device_type=self.device_type,
            operating_system=self.operating_system,
        )


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(description="Calibrated probability that it is fraud")
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approve", "review", "block"]
    model_version: str
    policy_version: str
    scoring_timestamp: datetime


def create_app(
    serving: ServingConfig | None = None, db_url: str | None = None, pool_size: int = 8
) -> FastAPI:
    serving = serving or load_serving_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One row per prediction gains nothing from OpenMP threads (measured: 8.8 vs 8.9 ms),
        # and several worker processes x 20 threads each would oversubscribe the CPU.
        threadpool_limits(limits=1, user_api="openmp")
        bundle = Bundle.load(serving.bundle_dir)
        app.state.scorer = Scorer(bundle, serving.risk_levels)
        app.state.features = OnlineFeatureBuilder(load_feature_config())
        app.state.pool = ConnectionPool(
            db_url or database_url(),
            min_size=1,
            max_size=pool_size,
            kwargs={
                "autocommit": True,
                "options": f"-c statement_timeout={serving.statement_timeout_ms}",
            },
            timeout=5,
            open=True,
        )
        log.info("serving model %s, policy %s", bundle.model_version, bundle.policy_version)
        try:
            yield
        finally:
            app.state.pool.close()

    app = FastAPI(title="Fraud risk scoring", version="1.0.0", lifespan=lifespan)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payment: TransactionRequest, request: Request) -> PredictionResponse:
        state = request.app.state
        try:
            with state.pool.connection() as conn:
                result = score_payment(
                    conn,
                    state.scorer,
                    state.features,
                    payment.transaction_id,
                    payment.incoming(),
                    log_features=serving.log_features,
                )
        except UnknownEntity as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except psycopg.errors.QueryCanceled as exc:
            log.warning("scoring %s exceeded the time budget", payment.transaction_id)
            raise HTTPException(status_code=503, detail="scoring exceeded its time budget") from exc
        except (psycopg.OperationalError, PoolTimeout) as exc:
            log.exception("database unavailable")
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return PredictionResponse(**result.__dict__)

    @app.get("/health")
    def health(request: Request) -> dict:
        state = request.app.state
        try:
            with state.pool.connection() as conn:
                conn.execute("SELECT 1")
        except (psycopg.OperationalError, PoolTimeout) as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ok", "model_version": state.scorer.bundle.model_version}

    @app.get("/model")
    def model(request: Request) -> dict:
        bundle = request.app.state.scorer.bundle
        policy = bundle.policy
        return {
            "model_version": bundle.model_version,
            "policy_version": bundle.policy_version,
            "model_family": policy["model"]["family"],
            "trained_on": policy["model"]["trained_on"],
            "policy": policy["policy_name"],
            "policy_chosen_on": policy["chosen_on"],
            "calibration": policy["calibration"]["method"],
            "costs": policy["costs"],
            "review_capacity_per_day": bundle.capacity,
            "risk_levels": serving.risk_levels.model_dump(),
            "lineage": policy["lineage"],
        }

    return app
