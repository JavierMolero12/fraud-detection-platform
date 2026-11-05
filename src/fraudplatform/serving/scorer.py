"""The scorer shared by the API (one payment at a time) and batch scoring (a whole period).

Both use the frozen Phase 8 bundle unchanged (data/processed/phase8): the HistGradientBoosting
pipeline, the Platt calibrator and the expected-cost policy P2 with its daily review capacity.
Nothing here is refitted or re-tuned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.pipeline import Pipeline

from fraudplatform.config import RiskLevels
from fraudplatform.decision.calibration import apply_calibration_dict
from fraudplatform.decision.costs import APPROVE, BLOCK, REVIEW, CostModel, bayes_actions
from fraudplatform.models.pipelines import SKOPS_TRUSTED_TYPES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Bundle:
    model: Pipeline
    policy: dict
    columns: list[str]
    costs: CostModel
    capacity: int
    model_version: str  # first 12 hex digits of sha256(model.skops)
    policy_version: str  # first 12 hex digits of sha256(policy.json)

    @classmethod
    def load(cls, bundle_dir: Path) -> Bundle:
        model_path, policy_path = bundle_dir / "model.skops", bundle_dir / "policy.json"
        if not (model_path.exists() and policy_path.exists()):
            raise FileNotFoundError(
                f"no model bundle in {bundle_dir}: run `fraud policy optimise` (Phase 8)"
            )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy["policy"]["type"] != "expected_cost":
            raise ValueError(f"serving implements the expected-cost policy, got {policy['policy']}")
        return cls(
            model=sio.load(model_path, trusted=SKOPS_TRUSTED_TYPES),
            policy=policy,
            columns=list(policy["model"]["columns"]),
            costs=CostModel(**policy["costs"]),
            capacity=int(policy["review_capacity_per_day"]),
            model_version=_sha256(model_path)[:12],
            policy_version=_sha256(policy_path)[:12],
        )


class Scorer:
    def __init__(self, bundle: Bundle, risk: RiskLevels):
        self.bundle = bundle
        self.risk = risk

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probability."""
        raw = self.bundle.model.predict_proba(frame[self.bundle.columns])[:, 1]
        return apply_calibration_dict(self.bundle.policy["calibration"], raw)

    def risk_level(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.select(
            [q >= self.risk.high_from, q >= self.risk.medium_from], ["high", "medium"], "low"
        )

    def batch_actions(self, q, amount, day) -> np.ndarray:
        """A whole period at once: review slots of each day go to the largest benefits."""
        return bayes_actions(q, amount, day, self.bundle.costs, self.bundle.capacity)

    def preferred_actions(self, q: float, amount_eur: float) -> tuple[int, int]:
        """(best action, what to do instead if no review slot is left today).

        Same rule as bayes_actions: review only if it is strictly cheaper than the best of
        approve / block. Online, slots are given first come, first served.
        """
        expected = self.bundle.costs.expected([q], [amount_eur])[0]
        no_review = BLOCK if expected[BLOCK] < expected[APPROVE] else APPROVE
        return (REVIEW if expected[REVIEW] < expected[no_review] else no_review), no_review
