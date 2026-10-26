"""How good was the monitoring? Judged AFTER the replay against what really happened.

This is the only monitoring module allowed to read the simulator's ground truth
(sim.ground_truth) and the injected drift date (configs/simulation.yaml). Nothing here feeds
back into the alerts; tests/unit/test_monitoring.py checks that the monitoring itself never
reads the simulator schema.

- timeline: per rule, alerts before the drift (false alarms with respect to it) and the first
  alert after it, in days;
- truth: per week, what monitoring could not see on the day (all labels as known today, the
  scenario of every fraud): when did the model really get worse?
- post-hoc PSI thresholds: what other thresholds WOULD have given. Designed after seeing the
  replay, so it is a hypothesis for the next version of the rules, not a result.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import psycopg

from fraudplatform.monitoring.drift import bin_counts, psi, shares
from fraudplatform.monitoring.run import CAUGHT, SCORE, MonitoringData

DRIFT_SCENARIO = "wallet_provisioning"


def alert_timeline(alerts: pd.DataFrame, drift_start: date) -> pd.DataFrame:
    a = alerts.assign(as_of=pd.to_datetime(alerts["as_of"]).dt.date)
    rows = []
    for (rule, severity), g in a.groupby(["rule", "severity"]):
        before = g[g["as_of"] < drift_start]
        after = g[g["as_of"] >= drift_start]
        first_after = after["as_of"].min() if len(after) else None
        rows.append(
            {
                "rule": rule,
                "severity": severity,
                "runs_alerting": int(g["as_of"].nunique()),
                "runs_before_drift": int(before["as_of"].nunique()),
                "first_alert": g["as_of"].min(),
                "first_after_drift": first_after,
                "days_after_drift": (first_after - drift_start).days if first_after else None,
                "subjects": ", ".join(sorted(g["subject"].unique())[:6]),
            }
        )
    return pd.DataFrame(rows).sort_values(["days_after_drift", "rule"], na_position="last")


def weekly_truth(conn: psycopg.Connection, data: MonitoringData) -> pd.DataFrame:
    """Per production week, with every label known today and the simulator's scenarios."""
    truth = pd.DataFrame(
        conn.execute(
            "SELECT transaction_id, is_fraud_true, scenario FROM sim.ground_truth "
            "WHERE transaction_id = ANY(%s)",
            (list(data.production["transaction_id"]),),
        ).fetchall(),
        columns=["transaction_id", "is_fraud_true", "scenario"],
    )
    p = data.production[["transaction_id", "ts", SCORE, "decision"]].merge(
        truth, on="transaction_id", how="left"
    )
    p["reported"] = p["transaction_id"].isin(data.fraud_reports["transaction_id"])
    p["caught"] = p["decision"].isin(CAUGHT)
    p["week"] = (p["ts"] - data.production_start).dt.days // 7
    rows = []
    for k, g in p.groupby("week"):
        reported = g["reported"]
        rows.append(
            {
                "week_start": (data.production_start + pd.Timedelta(days=7 * int(k))).date(),
                "payments": len(g),
                "true_frauds": int(g["is_fraud_true"].sum()),
                "drift_scenario_frauds": int((g["scenario"] == DRIFT_SCENARIO).sum()),
                "reported_frauds": int(reported.sum()),
                "observed_rate": float(reported.mean()),
                "mean_predicted": float(g[SCORE].mean()),
                "recall_reported": float(g.loc[reported, "caught"].mean()),
                "recall_drift_scenario": float(
                    g.loc[g["scenario"] == DRIFT_SCENARIO, "caught"].mean()
                )
                if (g["scenario"] == DRIFT_SCENARIO).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def reference_noise(data: MonitoringData, window_days: int) -> pd.DataFrame:
    """PSI of each reference week against the whole reference: the noise of a weekly window
    when nothing has changed. Post hoc: an empirical basis for thresholds (decision 029)."""
    ref = data.reference
    start = ref["ts"].min().normalize()
    week = (ref["ts"] - start).dt.days // window_days
    rows = []
    for k, g in ref.groupby(week):
        if len(g) < 0.5 * len(ref) / week.nunique():
            continue  # a partial last week
        for col in [*data.features, SCORE]:
            cur = shares(bin_counts(g[col], data.binnings[col]))
            rows.append(
                {"week": int(k), "feature": col, "psi": psi(data.reference_shares[col], cur)}
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    feature_drift: pd.DataFrame,
    drift_start: date,
    thresholds: list[float],
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Post hoc: for other PSI thresholds, false-alarm runs before the drift and first
    detection after it (any model input above the threshold)."""
    f = feature_drift[~feature_drift["feature"].isin(exclude)].assign(
        as_of=pd.to_datetime(feature_drift["as_of"]).dt.date
    )
    rows = []
    for t in thresholds:
        over = f[f["psi"] >= t]
        before = over[over["as_of"] < drift_start]["as_of"].nunique()
        after = over[over["as_of"] >= drift_start]
        first = after["as_of"].min() if len(after) else None
        features_first = (
            ", ".join(sorted(after.loc[after["as_of"] == first, "feature"])[:5]) if first else ""
        )
        rows.append(
            {
                "threshold": t,
                "runs_alerting_before_drift": int(before),
                "first_alert_after_drift": first,
                "days_after_drift": (first - drift_start).days if first else None,
                "features_at_first_alert": features_first,
            }
        )
    return pd.DataFrame(rows)


def summary_numbers(tables: dict[str, pd.DataFrame], drift_start: date) -> dict:
    m = tables["metrics"].pivot(index="as_of", columns="metric", values="value")
    m.index = pd.to_datetime(m.index).date
    pre = m[m.index < drift_start]
    post = m[m.index >= drift_start]
    return {
        "score_psi_max_before_drift": float(np.nanmax(pre["score_psi"])),
        "score_psi_max_after_drift": float(np.nanmax(post["score_psi"])),
        "oe_max_before_drift": float(np.nanmax(pre["recent_observed_to_expected"])),
        "oe_max_after_drift": float(np.nanmax(post["recent_observed_to_expected"])),
    }


def _first_days(timeline: pd.DataFrame, rules: tuple[str, ...], severity: str = "alert"):
    t = timeline[timeline["rule"].isin(rules) & (timeline["severity"] == severity)]
    days = t["days_after_drift"].dropna()
    return int(days.min()) if len(days) else None


def evaluate(
    tables: dict[str, pd.DataFrame],
    data: MonitoringData,
    drift_free: pd.DataFrame,
    drift_start: date,
    truth: pd.DataFrame,
    window_days: int,
    maturity_share: float,
    recall_drop: float,
    rolling_threshold: float = 0.03,
    posthoc_fixed_threshold: float = 0.05,
    noise_threshold: float = 0.05,
) -> dict:
    """Everything the report and the findings quote, computed in one place.

    `drift_free` holds model inputs of weeks with no injected drift that were never used to
    design the post-hoc rules (first half of 2025).
    """
    timeline = alert_timeline(tables["alerts"], drift_start)
    noise = reference_noise(data, window_days)
    noise_inputs = noise[noise["feature"] != SCORE]
    noisy = noisy_features(noise_inputs, noise_threshold)
    stable_noise = noise_inputs[~noise_inputs["feature"].isin(noisy)]

    fd = tables["feature_drift"].assign(
        as_of=pd.to_datetime(tables["feature_drift"]["as_of"]).dt.date
    )
    stable = fd[~fd["feature"].isin(noisy)]
    noisy_psi = fd.loc[fd["feature"].isin(noisy), "psi"]

    # post-hoc rules, each checked on production before the drift and on drift-free weeks
    free_fixed = weekly_psi(drift_free, data, window_days)
    roll_prod = rolling_psi(data.production, data, window_days, previous=4)
    roll_prod = roll_prod[~roll_prod["feature"].isin(noisy)]
    roll_free = rolling_psi(drift_free, data, window_days, previous=4)
    roll_free = roll_free[~roll_free["feature"].isin(noisy)]

    def fixed_row(threshold: float, fixed_in_advance: bool) -> dict:
        over = stable[stable["psi"] >= threshold]
        after = over[over["as_of"] >= drift_start]["as_of"]
        check = false_alarm_check(free_fixed, threshold, noisy)
        return {
            "rule": f"input PSI vs reference >= {threshold}"
            + (" (fixed in advance)" if fixed_in_advance else ""),
            "production runs alerting before drift": int(
                over[over["as_of"] < drift_start]["as_of"].nunique()
            ),
            "drift-free weeks alerting": f"{check['weeks_alerting']} of {check['weeks']}",
            "days to first alert after drift": (after.min() - drift_start).days
            if len(after)
            else None,
        }

    # a rolling window's alert is known when its week ends
    roll_alerts = roll_prod[roll_prod["psi"] >= rolling_threshold]["week_start"]
    roll_after = [d + pd.Timedelta(days=window_days).to_pytimedelta() for d in roll_alerts]
    roll_after = [d for d in roll_after if d >= drift_start]
    free_weeks = roll_free["week_start"].nunique()
    roll_free_alerting = roll_free[roll_free["psi"] >= rolling_threshold]["week_start"].nunique()
    pre_roll = roll_prod[
        [
            d + pd.Timedelta(days=window_days).to_pytimedelta() < drift_start
            for d in roll_prod["week_start"]
        ]
    ]

    m = tables["metrics"].pivot(index="as_of", columns="metric", values="value")
    m.index = pd.to_datetime(m.index).date
    p_only = m[(m["recent_observed_to_expected"] > 1) & (m["recent_poisson_p"] < 0.01)].index
    posthoc = pd.DataFrame(
        [
            fixed_row(0.10, True),
            fixed_row(posthoc_fixed_threshold, False),
            {
                "rule": f"input PSI vs previous 4 weeks >= {rolling_threshold}",
                "production runs alerting before drift": int(
                    pre_roll[pre_roll["psi"] >= rolling_threshold]["week_start"].nunique()
                ),
                "drift-free weeks alerting": f"{roll_free_alerting} of {free_weeks}",
                "days to first alert after drift": (min(roll_after) - drift_start).days
                if roll_after
                else None,
            },
            {
                "rule": "labels: observed > expected with Poisson p < 0.01 (no ratio threshold)",
                "production runs alerting before drift": int(sum(d < drift_start for d in p_only)),
                "drift-free weeks alerting": "n/a",
                "days to first alert after drift": min(
                    (d - drift_start).days for d in p_only if d >= drift_start
                )
                if any(d >= drift_start for d in p_only)
                else None,
            },
        ]
    )

    summary = summary_numbers(tables, drift_start) | {
        "stable_inputs": len(data.features) - len(noisy),
        "reference_week_noise_max": float(stable_noise["psi"].max()),
        "noisy_feature_psi_min": float(noisy_psi.min()) if len(noisy_psi) else float("nan"),
        "noisy_feature_psi_max": float(noisy_psi.max()) if len(noisy_psi) else float("nan"),
        "stable_psi_max_before_drift": float(stable[stable["as_of"] < drift_start]["psi"].max()),
        "stable_psi_max_after_drift": float(stable[stable["as_of"] >= drift_start]["psi"].max()),
        "rolling_max_drift_free": float(roll_free["psi"].max()),
        "rolling_first_days": (min(roll_after) - drift_start).days if roll_after else None,
        "rolling_drift_free_weeks": int(free_weeks),
        "first_rate_alert_days": _first_days(timeline, ("review_rate", "block_rate")),
        "rate_alert_runs_before_drift": int(
            timeline.loc[
                timeline["rule"].isin(("review_rate", "block_rate"))
                & (timeline["severity"] == "alert"),
                "runs_before_drift",
            ].sum()
        ),
        "first_label_alert_days": _first_days(timeline, ("label_calibration",)),
        "first_retrain_days": _first_days(timeline, ("retrain",), "action"),
        "first_mature_recall_days": _first_days(timeline, ("mature_recall",)),
    }
    return {
        "timeline": timeline,
        "truth": truth,
        "noisy_features": noisy,
        "rolling_production": roll_prod,
        "rolling_threshold": rolling_threshold,
        "posthoc": posthoc,
        "summary": summary,
        "maturity_share": maturity_share,
        "maturity_days": data.delay.age_for_share(maturity_share),
        "recall_drop": recall_drop,
    }


def noisy_features(noise: pd.DataFrame, threshold: float) -> list[str]:
    """Inputs whose weekly PSI already exceeds the threshold INSIDE the reference period: they
    drift by construction (an age that only grows) and cannot be watched against a fixed
    reference."""
    worst = noise.groupby("feature")["psi"].max()
    return sorted(worst[worst >= threshold].index)


def weekly_psi(frame: pd.DataFrame, data: MonitoringData, window_days: int) -> pd.DataFrame:
    """PSI of every model input per week of `frame` against the reference bins."""
    start = frame["ts"].min().normalize()
    week = (frame["ts"] - start).dt.days // window_days
    rows = []
    for k, g in frame.groupby(week):
        if len(g) < 0.5 * len(frame) / week.nunique():
            continue
        for col in data.features:
            cur = shares(bin_counts(g[col], data.binnings[col]))
            rows.append(
                {
                    "week_start": (start + pd.Timedelta(days=window_days * int(k))).date(),
                    "feature": col,
                    "psi": psi(data.reference_shares[col], cur),
                }
            )
    return pd.DataFrame(rows)


def false_alarm_check(
    weekly: pd.DataFrame, threshold: float, exclude: list[str]
) -> dict[str, object]:
    """Weeks without injected drift: every week with an input above the threshold is a false
    alarm of the (post-hoc) rule."""
    w = weekly[~weekly["feature"].isin(exclude)]
    over = w[w["psi"] >= threshold]
    return {
        "weeks": int(w["week_start"].nunique()),
        "weeks_alerting": int(over["week_start"].nunique()),
        "max_psi": float(w["psi"].max()),
        "feature_at_max": str(w.loc[w["psi"].idxmax(), "feature"]),
    }


def rolling_psi(
    frame: pd.DataFrame, data: MonitoringData, window_days: int, previous: int
) -> pd.DataFrame:
    """Post hoc: PSI of each week against the `previous` weeks before it (same bins). A step
    change stands out; slow evolution (ageing histories, seasonality) mostly cancels out."""
    start = frame["ts"].min().normalize()
    week = ((frame["ts"] - start).dt.days // window_days).to_numpy()
    full = [k for k in np.unique(week) if (week == k).sum() >= 0.5 * len(frame) / len(set(week))]
    counts = {
        col: {k: bin_counts(frame.loc[week == k, col], data.binnings[col]) for k in full}
        for col in data.features
    }
    rows = []
    for i, k in enumerate(full):
        if i < previous:
            continue
        for col in data.features:
            ref = sum(counts[col][j] for j in full[i - previous : i])
            rows.append(
                {
                    "week_start": (start + pd.Timedelta(days=window_days * int(k))).date(),
                    "feature": col,
                    "psi": psi(shares(ref), shares(counts[col][k])),
                }
            )
    return pd.DataFrame(rows)
