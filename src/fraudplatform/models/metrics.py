"""Evaluation metrics for a rare positive class.

- PR-AUC (average precision): model selection. A random score gets ~ the base rate.
- ROC-AUC: reported, not used to select (it looks good even when precision is poor).
- Brier score / log loss: only for scores that claim to be probabilities.
- Alert metrics: precision and recall if we could flag `alert_rate` of all transactions,
  plus the share of fraud EUR caught. This is the operational view (review capacity).
- Confidence interval for PR-AUC by bootstrapping whole DAYS: transactions of one fraud
  campaign are not independent, so resampling rows would understate the uncertainty.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def top_k_mask(score, k: int, seed: int = 0) -> np.ndarray:
    """Boolean mask of the k highest scores. Ties are broken at random (seeded): a rules
    score with few distinct values must not get lucky from the row order."""
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def alert_metrics(y, score, amount, alert_rate: float, seed: int = 0) -> dict[str, float]:
    y = np.asarray(y, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    k = max(1, math.ceil(alert_rate * len(y)))
    alerts = top_k_mask(score, k, seed)
    caught = alerts & y
    return {
        "alerts": float(k),
        "precision_at_alert_rate": caught.sum() / k,
        "recall_at_alert_rate": caught.sum() / max(y.sum(), 1),
        "amount_recall_at_alert_rate": amount[caught].sum() / max(amount[y].sum(), 1e-9),
    }


def bootstrap_pr_auc(y, score, days, reps: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of PR-AUC, resampling whole days with replacement."""
    if reps == 0:
        return float("nan"), float("nan")
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    groups = pd.Series(np.arange(len(y))).groupby(np.asarray(days)).indices
    blocks = list(groups.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        picked = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in picked])
        if y[idx].any():
            values.append(average_precision_score(y[idx], score[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_pr_auc_difference(
    y, score_a, score_b, days, reps: int, seed: int
) -> dict[str, float]:
    """PR-AUC(a) - PR-AUC(b) with a PAIRED day-block bootstrap.

    Both models are evaluated on the same resampled days, so day-to-day noise cancels out.
    Comparing two separate intervals (overlap or not) is much less sensitive than this.
    """
    y = np.asarray(y, dtype=bool)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    blocks = list(pd.Series(np.arange(len(y))).groupby(np.asarray(days)).indices.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if y[idx].any():
            diffs.append(
                average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx])
            )
    diffs = np.asarray(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": float(average_precision_score(y, a) - average_precision_score(y, b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float((diffs > 0).mean()),
    }


def evaluate(
    y,
    score,
    amount,
    days,
    alert_rate: float,
    probabilistic: bool,
    bootstrap_reps: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    out = {
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if np.unique(score).size > 1 else 0.5,
    }
    out["pr_auc_ci_low"], out["pr_auc_ci_high"] = bootstrap_pr_auc(
        y, score, days, bootstrap_reps, seed
    )
    if probabilistic:
        p = np.clip(score, 1e-7, 1 - 1e-7)
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, p))
        out["mean_predicted"] = float(p.mean())
    out.update(alert_metrics(y, score, amount, alert_rate, seed))
    return out
