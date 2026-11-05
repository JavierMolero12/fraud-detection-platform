"""Probability calibration: does a score of 0.3 mean a 30% chance of fraud?

A policy that compares expected costs (P2) multiplies p by money, so a probability that is
2x too high doubles the expected loss it believes it avoids. Ranking metrics (PR-AUC) cannot
see this: calibration changes probabilities, not their order.

Methods (fitted on held-out scores, never on the model's training rows):
- platt    : logistic regression on logit(p). Two parameters, smooth, cannot fix non-monotone
             distortions but cannot overfit much either.
- isotonic : non-decreasing step function. Fits any monotone distortion, needs more data and
             can overfit (and creates ties in the score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("none", "platt", "isotonic")
EPS = 1e-7


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, p, y):
    if method == "none":
        return None
    if method == "platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), np.asarray(y, dtype=int))
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        )
    raise ValueError(f"unknown calibration method {method!r}")


def apply_calibrator(method: str, calibrator, p) -> np.ndarray:
    if method == "none":
        return np.asarray(p, dtype=float)
    if method == "platt":
        return calibrator.predict_proba(_logit(p))[:, 1]
    return calibrator.predict(np.asarray(p, dtype=float))


def reliability_table(p, y, bins: int = 15) -> pd.DataFrame:
    """Mean predicted vs observed fraud rate in equal-count bins of the predicted probability.

    Equal-count bins, because with 0.7% fraud almost all predictions are close to 0 and
    equal-width bins would put 99% of the rows in the first bin.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    bucket = np.empty(len(p), dtype=int)
    bucket[order] = np.arange(len(p)) * bins // len(p)
    frame = pd.DataFrame({"bucket": bucket, "predicted": p, "observed": y})
    return frame.groupby("bucket").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size")
    )


def expected_calibration_error(p, y, bins: int = 15) -> float:
    """Count-weighted mean |predicted - observed| over equal-count bins."""
    table = reliability_table(p, y, bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["predicted"] - table["observed"]).abs()).sum())


def calibrator_to_dict(method: str, calibrator) -> dict:
    """Plain-JSON form of a fitted calibrator: the scoring service applies it with numpy only."""
    if method == "none":
        return {"method": "none"}
    if method == "platt":
        return {
            "method": "platt",
            "coef": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        }
    return {
        "method": "isotonic",
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
    }


def apply_calibration_dict(spec: dict, p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if spec["method"] == "none":
        return p
    if spec["method"] == "platt":
        z = spec["coef"] * _logit(p).ravel() + spec["intercept"]
        return 1 / (1 + np.exp(-z))
    # isotonic = piecewise-linear interpolation between thresholds, clipped at both ends
    return np.interp(p, spec["x"], spec["y"])
