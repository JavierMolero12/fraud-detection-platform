"""Cost model and decision policies (docs/phase1_design.md, section 10).

Three actions per transaction: approve, review (analyst / step-up), block. Two policies:

P1  global thresholds   score >= t_block -> block; t_review <= score < t_block -> review
P2  per-transaction     action = argmin expected cost given p and the amount
                        (only meaningful if p is a calibrated probability)

Both respect a daily review capacity: if more transactions qualify for review than analysts
can handle, the highest-priority ones are reviewed and the rest are approved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudplatform.config import CostConfig

APPROVE, REVIEW, BLOCK = 0, 1, 2
ACTION_NAMES = np.array(["approve", "review", "block"])


@dataclass(frozen=True)
class CostModel:
    chargeback_fee: float
    review_cost: float
    reviewer_recall: float
    review_friction: float
    lost_margin_rate: float
    block_friction: float

    @classmethod
    def from_config(cls, cfg: CostConfig) -> CostModel:
        return cls(
            cfg.chargeback_fee,
            cfg.review_cost,
            cfg.reviewer_recall,
            cfg.review_friction,
            cfg.lost_margin_rate,
            cfg.block_friction,
        )

    def _components(self, amount):
        a = np.asarray(amount, dtype=float)
        loss = a + self.chargeback_fee  # what an approved fraud costs
        wrongful_block = self.lost_margin_rate * a + self.block_friction
        return loss, wrongful_block

    def realized(self, y, amount) -> np.ndarray:
        """(n, 3) cost of each action given the TRUE label (used to evaluate a policy)."""
        y = np.asarray(y, dtype=float)
        loss, wrongful_block = self._components(amount)
        return np.column_stack(
            [
                y * loss,
                self.review_cost
                + y * (1 - self.reviewer_recall) * loss
                + (1 - y) * self.review_friction,
                (1 - y) * wrongful_block,
            ]
        )

    def expected(self, p, amount) -> np.ndarray:
        """(n, 3) EXPECTED cost of each action given a fraud probability (used to decide)."""
        return self.realized(np.asarray(p, dtype=float), amount)  # linear in y

    def breakeven(self, amount: float) -> dict[str, float]:
        """Probabilities where the best action changes for a given amount (no capacity limit)."""
        loss, wrongful = self._components(amount)
        f_r, c_r, rho = self.review_friction, self.review_cost, self.reviewer_recall
        approve_review = (c_r + f_r) / (rho * loss + f_r)
        review_block = (wrongful - c_r - f_r) / ((1 - rho) * loss - f_r + wrongful)
        approve_block = wrongful / (loss + wrongful)
        return {
            "approve_review": float(approve_review),
            "review_block": float(review_block),
            "approve_block": float(approve_block),
        }


def within_day_rank(priority, day) -> np.ndarray:
    """1 = highest priority of its day. Ties broken by row order (deterministic)."""
    return (
        pd.Series(np.asarray(priority, dtype=float))
        .groupby(np.asarray(day))
        .rank(method="first", ascending=False)
        .to_numpy()
    )


def threshold_actions(
    score, day, t_review: float, t_block: float, capacity: int, rank=None, day_codes=None
) -> np.ndarray:
    """Policy P1. `rank`/`day_codes` can be precomputed to evaluate many thresholds fast."""
    score = np.asarray(score, dtype=float)
    if rank is None:
        rank = within_day_rank(score, day)
    if day_codes is None:
        day_codes = pd.factorize(np.asarray(day))[0]
    block = score >= t_block
    band = (score >= t_review) & ~block
    # Rows of a day sorted by score: first the blocked ones, then the review band. The band
    # rows that fit in the capacity are those ranked within n_blocked + capacity.
    n_blocked = np.bincount(day_codes[block], minlength=day_codes.max() + 1)
    reviewed = band & (rank <= n_blocked[day_codes] + capacity)
    actions = np.full(len(score), APPROVE)
    actions[block] = BLOCK
    actions[reviewed] = REVIEW
    return actions


def bayes_actions(p, amount, day, costs: CostModel, capacity: int) -> np.ndarray:
    """Policy P2: cheapest expected action per transaction; review slots go to the
    transactions where reviewing saves the most compared with their best other option."""
    expected = costs.expected(p, amount)
    no_review = np.where(expected[:, BLOCK] < expected[:, APPROVE], BLOCK, APPROVE)
    benefit = expected[np.arange(len(no_review)), no_review] - expected[:, REVIEW]
    candidate = benefit > 0
    rank = within_day_rank(np.where(candidate, benefit, -np.inf), day)
    actions = no_review.copy()
    actions[candidate & (rank <= capacity)] = REVIEW
    return actions


def evaluate_actions(
    actions, y, amount, realized: np.ndarray, day, reviewer_recall: float
) -> dict[str, float]:
    """Business view of a policy over a period."""
    y = np.asarray(y, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    actions = np.asarray(actions)
    n_days = len(np.unique(np.asarray(day)))
    cost = float(realized[np.arange(len(actions)), actions].sum())
    reviews_per_day = pd.Series(actions == REVIEW).groupby(np.asarray(day)).sum()
    blocked, reviewed = actions == BLOCK, actions == REVIEW
    fraud_eur = amount[y].sum()
    return {
        "total_cost_eur": cost,
        "cost_per_1000_txn": 1000 * cost / len(actions),
        "blocked": int(blocked.sum()),
        "reviewed": int(reviewed.sum()),
        "reviews_per_day_mean": float(reviews_per_day.mean()),
        "reviews_per_day_max": int(reviews_per_day.max()),
        "fraud_blocked": int((blocked & y).sum()),
        "fraud_reviewed": int((reviewed & y).sum()),
        "fraud_approved": int((actions == APPROVE)[y].sum()),
        "legit_blocked": int((blocked & ~y).sum()),
        "false_decline_rate": float((blocked & ~y).sum() / max((~y).sum(), 1)),
        "fraud_eur_stopped_share": float(
            (amount[blocked & y].sum() + reviewer_recall * amount[reviewed & y].sum())
            / max(fraud_eur, 1e-9)
        ),
        "days": n_days,
    }


def threshold_grid(
    score, realized: np.ndarray, day, capacity: int, review_grid, block_grid
) -> pd.DataFrame:
    """Total realised cost of every (t_review, t_block) pair with t_review < t_block."""
    score = np.asarray(score, dtype=float)
    rank = within_day_rank(score, day)
    day_codes = pd.factorize(np.asarray(day))[0]
    rows = []
    idx = np.arange(len(score))
    for t_block in block_grid:
        for t_review in review_grid:
            if t_review >= t_block:
                continue
            actions = threshold_actions(score, day, t_review, t_block, capacity, rank, day_codes)
            rows.append(
                {
                    "t_review": t_review,
                    "t_block": t_block,
                    "total_cost_eur": float(realized[idx, actions].sum()),
                    "reviewed": int((actions == REVIEW).sum()),
                    "blocked": int((actions == BLOCK).sum()),
                }
            )
    return pd.DataFrame(rows)
