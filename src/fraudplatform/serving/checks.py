"""Checks of a deployed scoring service and of the environment it runs in.

check_api      a running API (e.g. the Docker container) gives, for stored payments, exactly the
               probabilities and decisions that batch scoring gives with the same bundle.
verify_bundle  this environment (library versions, OS) reproduces the validation scores saved
               when the bundle was frozen (Phase 8), bit for bit.
"""

from __future__ import annotations

import platform
import uuid
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psycopg
import sklearn

from fraudplatform.eda.data import load_features
from fraudplatform.models.pipelines import prepare
from fraudplatform.serving.batch import score_period
from fraudplatform.serving.bench import SAMPLE, payload
from fraudplatform.serving.scorer import Scorer


def check_api(
    conn: psycopg.Connection,
    url: str,
    scorer: Scorer,
    payments: int,
    client: httpx.Client | None = None,
) -> dict:
    """POST `payments` stored payments of the last two loaded days to `url` and compare.

    `client` replaces the HTTP client (tests pass FastAPI's TestClient). The check rows are
    deleted from scoring.predictions afterwards.
    """
    run = uuid.uuid4().hex[:8]
    last = conn.execute("SELECT max(ts) FROM features.transaction_features").fetchone()[0]
    if last is None:
        raise ValueError("features.transaction_features is empty: run `fraud features build`")
    start, end = last.date() - timedelta(days=1), last.date() + timedelta(days=1)
    rows = conn.execute(SAMPLE, (start, end, run, payments)).fetchall()
    responses = {}
    http = client or httpx.Client(base_url=url, timeout=60)
    try:
        served = http.get("/model").json()
        for row in rows:
            r = http.post("/predict", json=payload(row, f"check-{run}-"))
            r.raise_for_status()
            responses[row[0]] = r.json()
    finally:
        if client is None:
            http.close()
        conn.execute(
            "DELETE FROM scoring.predictions WHERE source = 'api' AND transaction_id LIKE %s",
            (f"check-{run}-%",),
        )
        conn.commit()

    batch, _ = score_period(conn, scorer, start, end)
    batch = batch.set_index("transaction_id").loc[list(responses)]
    api = pd.DataFrame(responses.values(), index=batch.index)
    diff = (api["fraud_probability"] - batch["fraud_probability"]).abs()
    return {
        "url": url,
        "payments": len(responses),
        "period": [str(start), str(end)],
        "served_model_version": served["model_version"],
        "local_model_version": scorer.bundle.model_version,
        "max_abs_probability_difference": float(diff.max()),
        "same_decision": int((api["decision"] == batch["decision"]).sum()),
        "decisions": api["decision"].value_counts().to_dict(),
    }


def check_passed(result: dict) -> bool:
    return (
        result["payments"] > 0
        and result["served_model_version"] == result["local_model_version"]
        and result["max_abs_probability_difference"] == 0.0
        and result["same_decision"] == result["payments"]
    )


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Units in the last place between two arrays of non-negative float64 (e.g. probabilities)."""
    return np.abs(a.astype(np.float64).view(np.int64) - b.astype(np.float64).view(np.int64))


# Between operating systems the tree outputs are identical, but the sigmoid uses the platform's
# exp(): measured Windows -> Linux, 1,446 of 359,220 probabilities moved by 1-2 ulps and no
# decision changed (decision 027). More than this is a different model or different inputs.
MAX_ULPS_ACROSS_PLATFORMS = 4


def verify_bundle(conn: psycopg.Connection, scorer: Scorer, bundle_dir: Path) -> dict:
    """Re-score the period the policy was chosen on and compare with validation_scores.parquet.

    The saved file has no transaction ids, so the two score vectors are compared sorted: equal
    sorted vectors of 359,220 floats leave no room for a model that behaves differently.
    Only the raw model output is compared: the saved calibrated scores are cross-fitted
    (Phase 8), not the output of the shipped calibrator.
    `identical` is the bit-for-bit result (expected on the training OS); `passed` allows the
    cross-platform rounding of the sigmoid.
    """
    saved = pd.read_parquet(bundle_dir / "validation_scores.parquet", columns=["p_raw"])
    start, end = (date.fromisoformat(d) for d in scorer.bundle.policy["chosen_on"])
    df = prepare(load_features(conn, start, end, with_label=False))
    p_raw = scorer.bundle.model.predict_proba(df[scorer.bundle.columns])[:, 1]
    a, b = np.sort(p_raw), np.sort(saved["p_raw"].to_numpy())
    same_size = len(a) == len(b)
    ulps = ulp_distance(a, b) if same_size else None
    return {
        "period": [str(start), str(end)],
        "rescored": int(len(a)),
        "saved": int(len(b)),
        "identical": bool(same_size and np.array_equal(a, b)),
        "differing": int((a != b).sum()) if same_size else None,
        "max_ulps": int(ulps.max()) if same_size else None,
        "max_abs_difference": float(np.abs(a - b).max()) if same_size else None,
        "passed": bool(same_size and ulps.max() <= MAX_ULPS_ACROSS_PLATFORMS),
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
