"""Observed fraud labels: what the company learns, and when.

The ground truth (``is_fraud_true``) and the observed labels differ on purpose:
- labels arrive with a delay (chargebacks take days to weeks),
- some fraud is never reported,
- small card-testing charges are often never disputed,
- some legitimate purchases are disputed as fraud (friendly fraud),
- some disputes are investigated and cleared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudplatform.simulation.legit import LEGIT
from fraudplatform.simulation.randomness import DAY, weighted_choice
from fraudplatform.simulation.reference import PM_ECOMMERCE

SOURCES = np.array(["chargeback", "customer_report", "investigation"])


def _delays(rng, median_days, sigma: float, max_days: int) -> np.ndarray:
    days = rng.lognormal(np.log(median_days), sigma)
    return (np.clip(days, 0.5, max_days) * DAY).astype(np.int64)


def generate_labels(txns: pd.DataFrame, cfg, rng) -> pd.DataFrame:
    """Return one row per labelled transaction: row, is_fraud, reported_s, label_source.

    ``txns`` must have the columns ts, scenario, incident, amount_eur and payment_method.
    Transactions without a label row are, as far as the company knows, legitimate.
    """
    params = cfg.labels
    max_s = params.max_delay_days * DAY
    fraud = txns[txns["scenario"] != LEGIT]

    # Incident level: is it reported, when, and through which channel.
    incidents = fraud.groupby("incident").agg(scenario=("scenario", "first"), last_ts=("ts", "max"))
    scen = incidents["scenario"].to_numpy()
    report_p = np.array([params.report_prob_for(s) for s in scen])
    reported = rng.random(len(incidents)) < report_p
    median = np.array([params.median_delay_days[s] for s in scen])
    inc_report_s = incidents["last_ts"].to_numpy() + _delays(
        rng, median, params.delay_sigma, params.max_delay_days
    )
    inc_source = SOURCES[weighted_choice(rng, [0.6, 0.3, 0.1], len(incidents))]
    inc = pd.DataFrame(
        {"reported": reported, "inc_report_s": inc_report_s, "source": inc_source},
        index=incidents.index,
    )

    f = fraud[["ts", "scenario", "incident", "amount_eur"]].join(inc, on="incident")
    keep = f["reported"].to_numpy().copy()
    micro = (f["scenario"].to_numpy() == "card_testing") & (f["amount_eur"].to_numpy() < 5)
    keep &= ~micro | (rng.random(len(f)) < params.card_testing_micro_report_prob)
    f = f[keep]
    ts = f["ts"].to_numpy()
    reported_s = f["inc_report_s"].to_numpy() + rng.integers(0, 2 * DAY, len(f))
    fraud_labels = pd.DataFrame(
        {
            "row": f.index.to_numpy(),
            "is_fraud": True,
            "reported_s": np.clip(reported_s, ts + 1, ts + max_s),
            "label_source": f["source"].to_numpy(),
        }
    )

    # Label noise on legitimate transactions.
    legit = txns[txns["scenario"] == LEGIT]
    online = legit["payment_method"].to_numpy() == PM_ECOMMERCE
    u = rng.random(len(legit))
    friendly = u < params.friendly_fraud_rate * np.where(online, 1.6, 0.6)
    cleared = ~friendly & (rng.random(len(legit)) < params.cleared_dispute_rate)
    noise = []
    for mask, is_fraud, source, median_days in (
        (friendly, True, "chargeback", 30),
        (cleared, False, "investigation", 20),
    ):
        ts = legit["ts"].to_numpy()[mask]
        noise.append(
            pd.DataFrame(
                {
                    "row": legit.index.to_numpy()[mask],
                    "is_fraud": is_fraud,
                    "reported_s": ts
                    + _delays(rng, np.full(len(ts), median_days), 0.6, params.max_delay_days),
                    "label_source": source,
                }
            )
        )
    return pd.concat([fraud_labels, *noise], ignore_index=True).sort_values("row")
