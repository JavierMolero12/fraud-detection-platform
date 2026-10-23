"""Typed loading of the YAML configuration files in ``configs/``.

Configuration is validated with pydantic so that a typo in a YAML file fails loudly at start-up
instead of silently producing a different dataset.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(os.environ.get("FRAUD_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"
SQL_DIR = PROJECT_ROOT / "sql"

SCENARIOS = (
    "card_testing",
    "account_takeover",
    "card_cloning",
    "fraud_ring",
    "merchant_compromise",
    "stealth",
)
DRIFT_SCENARIO = "wallet_provisioning"


class ScaleConfig(BaseModel):
    n_customers: int = Field(gt=0)
    n_merchants: int = Field(gt=0)
    start: date
    end: date

    @model_validator(mode="after")
    def _check_period(self) -> ScaleConfig:
        if (self.end - self.start).days < 28:
            raise ValueError("simulation period must cover at least 28 days")
        return self


class CustomerParams(BaseModel):
    share_created_before_start: float = Field(ge=0, le=1)
    max_account_age_years: float = Field(gt=0)
    activity_sigma: float = Field(ge=0)
    amount_sigma: float = Field(ge=0)
    night_owl_share: float = Field(ge=0, le=1)


class CardParams(BaseModel):
    second_card_prob: float = Field(ge=0, le=1)
    second_card_usage: float = Field(ge=0, le=1)


class DeviceParams(BaseModel):
    upgrade_prob: float = Field(ge=0, le=1)
    secondary_prob: float = Field(ge=0, le=1)
    secondary_usage: float = Field(ge=0, le=1)
    shared_secondary_prob: float = Field(ge=0, le=1)


class LegitParams(BaseModel):
    preferred_merchants: int = Field(gt=0)
    zipf_exponent: float = Field(gt=0)
    explore_prob: float = Field(ge=0, le=1)
    online_preference: float = Field(gt=0)
    splurge_prob: float = Field(ge=0, le=1)
    splurge_multiplier: tuple[float, float]


class FraudParams(BaseModel):
    target_rate: float = Field(gt=0, lt=0.2)
    scenario_mix: dict[str, float]
    attacker_home_country_prob: float = Field(ge=0, le=1)
    compromised_merchants: int = Field(gt=0)
    cards_per_ring_device: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_mix(self) -> FraudParams:
        if set(self.scenario_mix) != set(SCENARIOS):
            raise ValueError(f"scenario_mix must define exactly {SCENARIOS}")
        if abs(sum(self.scenario_mix.values()) - 1.0) > 1e-6:
            raise ValueError("scenario_mix must sum to 1")
        return self


class LabelParams(BaseModel):
    max_delay_days: int = Field(gt=0)
    delay_sigma: float = Field(gt=0)
    report_prob: dict[str, float]
    median_delay_days: dict[str, float]
    card_testing_micro_report_prob: float = Field(ge=0, le=1)
    friendly_fraud_rate: float = Field(ge=0, le=1)
    cleared_dispute_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check_scenarios(self) -> LabelParams:
        missing = set(SCENARIOS + (DRIFT_SCENARIO,)) - set(self.median_delay_days)
        if missing:
            raise ValueError(f"median_delay_days missing scenarios: {sorted(missing)}")
        if "default" not in self.report_prob:
            raise ValueError("report_prob needs a 'default' entry")
        return self

    def report_prob_for(self, scenario: str) -> float:
        return self.report_prob.get(scenario, self.report_prob["default"])


class DriftParams(BaseModel):
    start: date
    online_shift_prob: float = Field(ge=0, le=1)
    wallet_share_physical: float = Field(ge=0, le=1)
    amount_inflation: float = Field(gt=0)
    wallet_fraud_incidents_per_10k_customers: float = Field(ge=0)


class DefectParams(BaseModel):
    exact_duplicate: float = Field(ge=0, le=0.05)
    duplicate_id_conflict: float = Field(ge=0, le=0.05)
    invalid_amount: float = Field(ge=0, le=0.05)
    invalid_currency: float = Field(ge=0, le=0.05)
    invalid_timestamp: float = Field(ge=0, le=0.05)
    missing_customer_id: float = Field(ge=0, le=0.05)
    orphan_reference: float = Field(ge=0, le=0.05)
    country_mismatch: float = Field(ge=0, le=0.05)
    sentinel_device: float = Field(ge=0, le=0.05)
    customer_duplicate: float = Field(ge=0, le=0.05)
    customer_missing_country: float = Field(ge=0, le=0.05)
    card_orphan_customer: float = Field(ge=0, le=0.05)


class SimulationConfig(BaseModel):
    simulator_version: str
    seed: int
    scales: dict[str, ScaleConfig]
    customers: CustomerParams
    cards: CardParams
    devices: DeviceParams
    legit: LegitParams
    fraud: FraudParams
    labels: LabelParams
    drift: DriftParams
    defects: DefectParams

    def scale(self, name: str) -> ScaleConfig:
        if name not in self.scales:
            raise KeyError(f"unknown scale '{name}', expected one of {sorted(self.scales)}")
        return self.scales[name]


class Period(BaseModel):
    start: date
    end: date  # exclusive

    @model_validator(mode="after")
    def _check(self) -> Period:
        if self.end <= self.start:
            raise ValueError("period end must be after start")
        return self


class Splits(BaseModel):
    train: Period
    validation: Period
    test: Period
    production: Period

    @model_validator(mode="after")
    def _contiguous(self) -> Splits:
        periods = [self.train, self.validation, self.test, self.production]
        for earlier, later in zip(periods, periods[1:], strict=False):
            if earlier.end != later.start:
                raise ValueError("splits must be contiguous and in time order")
        return self

    def items(self) -> list[tuple[str, Period]]:
        return [
            (name, getattr(self, name)) for name in ("train", "validation", "test", "production")
        ]


def load_splits(path: Path | None = None) -> Splits:
    path = path or CONFIG_DIR / "splits.yaml"
    with open(path, encoding="utf-8") as fh:
        return Splits.model_validate(yaml.safe_load(fh))


class FeatureConfig(BaseModel):
    feature_set_version: str
    micro_amount_eur: float = Field(gt=0)
    night_hours: tuple[int, int]
    impossible_speed_kmh: float = Field(gt=0)
    merchant_rate_prior: float = Field(gt=0, lt=1)
    merchant_rate_prior_strength: float = Field(gt=0)
    warmup_days: int = Field(ge=0)

    def sql_settings(self) -> dict[str, str]:
        """Values exposed to the feature SQL via current_setting('features.<name>')."""
        return {
            "features.version": self.feature_set_version,
            "features.micro_amount_eur": str(self.micro_amount_eur),
            "features.night_start": str(self.night_hours[0]),
            "features.night_end": str(self.night_hours[1]),
            "features.impossible_speed_kmh": str(self.impossible_speed_kmh),
            "features.merchant_rate_prior": str(self.merchant_rate_prior),
            "features.merchant_rate_prior_strength": str(self.merchant_rate_prior_strength),
        }


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    path = path or CONFIG_DIR / "features.yaml"
    with open(path, encoding="utf-8") as fh:
        return FeatureConfig.model_validate(yaml.safe_load(fh))


class EvaluationParams(BaseModel):
    alert_rate: float = Field(gt=0, lt=1)
    bootstrap_reps: int = Field(ge=0)
    bootstrap_seed: int


class RuleParams(BaseModel):
    burst_card_txn_1h: int = Field(ge=1)
    card_testing_micro_1h: int = Field(ge=1)
    new_device_amount_eur: float = Field(ge=0)
    device_many_cards_24h: int = Field(ge=1)
    risky_categories: list[str]


class CrossValidationParams(BaseModel):
    test_months: list[str]  # "YYYY-MM", in increasing order
    gap_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> CrossValidationParams:
        if self.test_months != sorted(self.test_months) or len(set(self.test_months)) != len(
            self.test_months
        ):
            raise ValueError("test_months must be unique and in increasing order")
        return self


class ModelConfig(BaseModel):
    random_state: int
    evaluation: EvaluationParams
    rules: RuleParams
    models: dict[str, dict]
    cross_validation: CrossValidationParams | None = None
    tuning: dict | None = None


def load_model_config(path: Path | None = None) -> ModelConfig:
    path = path or CONFIG_DIR / "model.yaml"
    with open(path, encoding="utf-8") as fh:
        return ModelConfig.model_validate(yaml.safe_load(fh))


def load_simulation_config(path: Path | None = None) -> SimulationConfig:
    path = path or CONFIG_DIR / "simulation.yaml"
    with open(path, encoding="utf-8") as fh:
        return SimulationConfig.model_validate(yaml.safe_load(fh))
