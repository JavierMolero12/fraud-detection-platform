"""One monitoring run as of a date, and the replay of all weekly runs since deployment.

A run on date D sees only payments with ts < D (their features, score and decision from the
scoring log) and fraud labels with reported_at < D. `monitor_as_of` enforces it by filtering
everything on D first; tests check that data after D cannot change a run.

Outputs per run (and tables in the `monitoring` schema):
- feature_drift   PSI (and KS) of every model input, last week vs the reference period;
- metrics         score PSI, review / block rates, delay-adjusted calibration (observed vs
                  expected reported frauds among recent payments);
- label_cohorts   weekly cohorts: frauds known so far, expected so far, recall, maturity;
- alerts          the rules of configs/monitoring.yaml applied to the above.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import psycopg

from fraudplatform.config import MonitoringConfig, Period
from fraudplatform.decision.costs import ACTION_NAMES
from fraudplatform.eda.data import load_features
from fraudplatform.features.catalog import BOOLEAN, CATEGORICAL
from fraudplatform.models.pipelines import prepare
from fraudplatform.monitoring.drift import (
    Binning,
    bin_counts,
    fit_binning,
    ks_statistic,
    psi,
    shares,
)
from fraudplatform.monitoring.labels import (
    DelayDistribution,
    expected_reported,
    poisson_two_sided_p,
    recall_drop_p,
)
from fraudplatform.serving.scorer import Scorer

log = logging.getLogger(__name__)
CAUGHT = ("review", "block")
SCORE = "fraud_probability"


@dataclass
class MonitoringData:
    """Everything a monitoring run needs, loaded once for a whole replay."""

    features: list[str]
    reference: pd.DataFrame  # model inputs, score, decision and (eventual) label
    production: pd.DataFrame  # ts, model inputs, score and decision from the scoring log
    fraud_reports: pd.DataFrame  # transaction_id, reported_at (frauds only)
    delay: DelayDistribution
    production_start: pd.Timestamp
    binnings: dict[str, Binning]
    reference_shares: dict[str, np.ndarray]
    reference_metrics: dict[str, float]


def _utc(day: date) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def load_monitoring_data(
    conn: psycopg.Connection, cfg: MonitoringConfig, scorer: Scorer, production: Period
) -> MonitoringData:
    features = list(scorer.bundle.columns)

    # reference: validation period, scored and decided by the deployed bundle
    ref = prepare(load_features(conn, cfg.reference.start, cfg.reference.end))
    ref[SCORE] = scorer.probability(ref)
    actions = scorer.batch_actions(
        ref[SCORE].to_numpy(), ref["amount_eur"].to_numpy(float), ref["ts"].dt.date.to_numpy()
    )
    ref["decision"] = ACTION_NAMES[actions]

    # production: features + what the batch scoring log recorded for this model version
    prod = prepare(load_features(conn, production.start, production.end, with_label=False))
    scored = pd.DataFrame(
        conn.execute(
            "SELECT transaction_id, fraud_probability, decision FROM scoring.predictions "
            "WHERE source = 'batch' AND model_version = %s "
            "AND transaction_ts >= %s AND transaction_ts < %s",
            (scorer.bundle.model_version, production.start, production.end),
        ).fetchall(),
        columns=["transaction_id", SCORE, "decision"],
    )
    if len(scored) != len(prod):
        raise ValueError(
            f"{len(scored)} logged scores for {len(prod)} production payments: "
            "run `fraud serve batch` for this period and bundle first"
        )
    prod = prod.merge(scored, on="transaction_id", how="inner")

    reports = pd.DataFrame(
        conn.execute(
            "SELECT l.transaction_id, l.reported_at FROM core.fraud_labels l "
            "JOIN core.transactions t USING (transaction_id) "
            "WHERE l.is_fraud AND t.ts >= %s",
            (production.start,),
        ).fetchall(),
        columns=["transaction_id", "reported_at"],
    )
    reports["reported_at"] = pd.to_datetime(reports["reported_at"], utc=True)

    lab = cfg.labels
    delays = [
        r[0]
        for r in conn.execute(
            "SELECT extract(epoch FROM l.reported_at - t.ts) / 86400.0 "
            "FROM core.fraud_labels l JOIN core.transactions t USING (transaction_id) "
            "WHERE l.is_fraud AND t.ts >= %s AND t.ts < %s AND l.reported_at < %s",
            (lab.delay_from, lab.delay_to, lab.known_before),
        ).fetchall()
    ]

    return assemble(features, ref, prod, reports, delays, production.start, cfg.input_drift.bins)


def assemble(
    features: list[str],
    reference: pd.DataFrame,
    production: pd.DataFrame,
    fraud_reports: pd.DataFrame,
    delays_days,
    production_start: date,
    bins: int,
) -> MonitoringData:
    """In-memory part of the loading: reference bins and metrics. `reference` needs the model
    inputs, the score, the decision and the label; `production` the same without the label."""
    binnings, ref_shares = {}, {}
    for col in [*features, SCORE]:
        numeric = col not in CATEGORICAL and col not in BOOLEAN
        binnings[col] = fit_binning(reference[col], numeric, bins)
        ref_shares[col] = shares(bin_counts(reference[col], binnings[col]))

    y = reference["is_fraud"].to_numpy(dtype=bool)
    caught = reference["decision"].isin(CAUGHT).to_numpy()
    reference_metrics = {
        "review_rate": float((reference["decision"] == "review").mean()),
        "block_rate": float((reference["decision"] == "block").mean()),
        "mean_score": float(reference[SCORE].mean()),
        "recall": float(caught[y].mean()),
    }
    return MonitoringData(
        features=features,
        reference=reference,
        production=production.sort_values("ts", ignore_index=True),
        fraud_reports=fraud_reports,
        delay=DelayDistribution.fit(np.asarray(delays_days, dtype=float)),
        production_start=_utc(production_start),
        binnings=binnings,
        reference_shares=ref_shares,
        reference_metrics=reference_metrics,
    )


@dataclass
class RunResult:
    as_of: date
    feature_drift: pd.DataFrame
    metrics: pd.DataFrame
    label_cohorts: pd.DataFrame
    alerts: pd.DataFrame


def _alert(rule, subject, severity, value, threshold, detail="") -> dict:
    return {
        "rule": rule,
        "subject": subject,
        "severity": severity,
        "value": float(value),
        "threshold": float(threshold),
        "detail": detail,
    }


def _level(value: float, warning: float, alert: float) -> str | None:
    return "alert" if value >= alert else "warning" if value >= warning else None


def monitor_as_of(data: MonitoringData, cfg: MonitoringConfig, as_of: date) -> RunResult:
    cutoff = _utc(as_of)
    seen = data.production[data.production["ts"] < cutoff]  # nothing from the future...
    reports = data.fraud_reports[data.fraud_reports["reported_at"] < cutoff]  # ...nor its labels
    known = seen["transaction_id"].isin(reports["transaction_id"]).to_numpy()
    alerts, metrics = [], []
    ref_m = data.reference_metrics

    # --- input and prediction drift on the last window -------------------------------------
    window = seen[seen["ts"] >= cutoff - pd.Timedelta(days=cfg.replay.drift_window_days)]
    drift_rows = []
    if len(window):
        for col in data.features:
            cur = shares(bin_counts(window[col], data.binnings[col]))
            value = psi(data.reference_shares[col], cur)
            ks = (
                ks_statistic(
                    data.reference[col].to_numpy(dtype=float), window[col].to_numpy(dtype=float)
                )
                if data.binnings[col].numeric
                else float("nan")
            )
            drift_rows.append({"feature": col, "payments": len(window), "psi": value, "ks": ks})
            p = cfg.input_drift
            if level := _level(value, p.psi_warning, p.psi_alert):
                threshold = p.psi_alert if level == "alert" else p.psi_warning
                alerts.append(_alert("input_psi", col, level, value, threshold))

        pd_cfg = cfg.prediction_drift
        score_psi = psi(
            data.reference_shares[SCORE],
            shares(bin_counts(window[SCORE], data.binnings[SCORE])),
        )
        metrics.append(("score_psi", score_psi, 0.0))
        if level := _level(score_psi, pd_cfg.psi_warning, pd_cfg.psi_alert):
            threshold = pd_cfg.psi_alert if level == "alert" else pd_cfg.psi_warning
            alerts.append(_alert("score_psi", SCORE, level, score_psi, threshold))
        metrics.append(("mean_score", float(window[SCORE].mean()), ref_m["mean_score"]))
        for name in ("review", "block"):
            rate = float((window["decision"] == name).mean())
            reference = ref_m[f"{name}_rate"]
            metrics.append((f"{name}_rate", rate, reference))
            change = abs(rate / reference - 1) if reference else float("inf")
            if level := _level(change, pd_cfg.rate_warning, pd_cfg.rate_alert):
                threshold = pd_cfg.rate_alert if level == "alert" else pd_cfg.rate_warning
                detail = f"{rate:.4%} vs {reference:.4%} in the reference"
                alerts.append(_alert(f"{name}_rate", name, level, change, threshold, detail))

    # --- delay-adjusted calibration on recent payments ----------------------------------------
    lab = cfg.labels
    recent = seen["ts"] >= cutoff - pd.Timedelta(days=lab.recent_days)
    if recent.any():
        age = (cutoff - seen.loc[recent, "ts"]).dt.total_seconds().to_numpy() / 86400
        expected = expected_reported(seen.loc[recent, SCORE], age, data.delay)
        observed = int(known[recent.to_numpy()].sum())
        ratio = observed / expected if expected else float("nan")
        p_value = poisson_two_sided_p(observed, expected)
        metrics += [
            ("recent_expected_reports", expected, float("nan")),
            ("recent_observed_reports", float(observed), float("nan")),
            ("recent_observed_to_expected", ratio, 1.0),
            ("recent_poisson_p", p_value, float("nan")),
        ]
        outside = ratio >= lab.oe_alert_high or ratio <= lab.oe_alert_low
        if expected >= lab.min_expected and outside and p_value < lab.p_value:
            threshold = lab.oe_alert_high if ratio > 1 else lab.oe_alert_low
            detail = f"{observed} reported vs {expected:.1f} expected, Poisson p = {p_value:.2g}"
            alerts.append(_alert("label_calibration", "recent", "alert", ratio, threshold, detail))

    # --- weekly cohorts as known today ----------------------------------------------------------
    maturity_age = data.delay.age_for_share(lab.maturity_share)
    week = (seen["ts"] - data.production_start).dt.days // 7
    cohort_rows = []
    for k, idx in seen.groupby(week).groups.items():
        start = data.production_start + pd.Timedelta(days=7 * int(k))
        end = start + pd.Timedelta(days=7)
        if end > cutoff:
            continue  # the current, incomplete week
        rows = seen.loc[idx]
        is_known = rows["transaction_id"].isin(reports["transaction_id"]).to_numpy()
        age = (cutoff - rows["ts"]).dt.total_seconds().to_numpy() / 86400
        caught = rows["decision"].isin(CAUGHT).to_numpy()
        cohort_rows.append(
            {
                "cohort_start": start.date(),
                "payments": len(rows),
                "known_frauds": int(is_known.sum()),
                "expected_by_now": expected_reported(rows[SCORE], age, data.delay),
                "caught_known": int((is_known & caught).sum()),
                "mature": bool((cutoff - end).days >= maturity_age),
            }
        )
    cohorts = pd.DataFrame(cohort_rows)
    if len(cohorts) and cohorts["mature"].any():
        newest = cohorts[cohorts["mature"]].iloc[-1]
        frauds, caught_n = int(newest["known_frauds"]), int(newest["caught_known"])
        if frauds:
            recall = caught_n / frauds
            reference = ref_m["recall"]
            metrics.append(("mature_cohort_recall", recall, reference))
            p_value = recall_drop_p(caught_n, frauds, reference)
            if recall <= reference - lab.recall_drop_alert and p_value < lab.p_value:
                detail = (
                    f"cohort of {newest['cohort_start']}: caught {caught_n} of {frauds} "
                    f"known frauds, binomial p = {p_value:.2g}"
                )
                alerts.append(
                    _alert(
                        "mature_recall",
                        str(newest["cohort_start"]),
                        "alert",
                        recall,
                        reference - lab.recall_drop_alert,
                        detail,
                    )
                )

    return RunResult(
        as_of=as_of,
        feature_drift=pd.DataFrame(drift_rows, columns=["feature", "payments", "psi", "ks"]),
        metrics=pd.DataFrame(metrics, columns=["metric", "value", "reference"]),
        label_cohorts=cohorts,
        alerts=pd.DataFrame(
            alerts, columns=["rule", "subject", "severity", "value", "threshold", "detail"]
        ),
    )


LABEL_RULES = ("label_calibration", "mature_recall")


def retraining_actions(alerts: pd.DataFrame, as_of_dates: list[date], runs: int) -> pd.DataFrame:
    """A label alert in `runs` consecutive weekly runs recommends retraining.

    Input or score drift alone never does: it says the population changed, not that the model
    got worse, and retraining needs labels from the new regime anyway (decision 029).
    """
    flagged = set(alerts.loc[alerts["rule"].isin(LABEL_RULES), "as_of"])
    rows, streak = [], 0
    for day in as_of_dates:
        streak = streak + 1 if day in flagged else 0
        if streak >= runs:
            rules = sorted(
                set(alerts.loc[(alerts["as_of"] == day) & alerts["rule"].isin(LABEL_RULES), "rule"])
            )
            rows.append(
                {
                    "as_of": day,
                    "rule": "retrain",
                    "subject": "model",
                    "severity": "action",
                    "value": float(streak),
                    "threshold": float(runs),
                    "detail": f"label alerts in {streak} consecutive runs: {', '.join(rules)}",
                }
            )
    return pd.DataFrame(rows, columns=["as_of", *alerts.columns.drop("as_of")])


def replay(data: MonitoringData, cfg: MonitoringConfig) -> dict[str, pd.DataFrame]:
    dates = cfg.replay.as_of_dates()
    parts: dict[str, list[pd.DataFrame]] = {
        "feature_drift": [],
        "metrics": [],
        "label_cohorts": [],
        "alerts": [],
    }
    for day in dates:
        result = monitor_as_of(data, cfg, day)
        for name in parts:
            frame = getattr(result, name)
            if len(frame):
                parts[name].append(frame.assign(as_of=day))
        log.info("as of %s: %d alerts", day, len(result.alerts))
    tables = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in parts.items()
    }
    actions = retraining_actions(tables["alerts"], dates, cfg.retraining.consecutive_runs)
    if len(actions):
        tables["alerts"] = pd.concat([tables["alerts"], actions], ignore_index=True)
    return tables


TABLE_COLUMNS = {
    "feature_drift": ["as_of", "feature", "payments", "psi", "ks"],
    "metrics": ["as_of", "metric", "value", "reference"],
    "label_cohorts": [
        "as_of",
        "cohort_start",
        "payments",
        "known_frauds",
        "expected_by_now",
        "caught_known",
        "mature",
    ],
    "alerts": ["as_of", "rule", "subject", "severity", "value", "threshold", "detail"],
}


def read_tables(conn: psycopg.Connection, dates: list[date]) -> dict[str, pd.DataFrame]:
    """The stored results of these monitoring runs, as replay() returned them."""
    tables = {}
    for name, columns in TABLE_COLUMNS.items():
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM monitoring.{name} WHERE as_of = ANY(%s) "
            "ORDER BY as_of",
            (dates,),
        ).fetchall()
        tables[name] = pd.DataFrame(rows, columns=columns)
    for name in ("feature_drift", "metrics", "label_cohorts", "alerts"):
        frame = tables[name]
        for col in ("psi", "ks", "value", "reference", "threshold", "expected_by_now"):
            if col in frame:
                frame[col] = frame[col].astype(float)
    return tables


def write_tables(conn: psycopg.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """Replace the rows of the replayed dates in the monitoring schema (one transaction)."""
    dates = sorted({d for t in tables.values() if len(t) for d in t["as_of"]})
    with conn.transaction():
        for name, columns in TABLE_COLUMNS.items():
            conn.execute(f"DELETE FROM monitoring.{name} WHERE as_of = ANY(%s)", (dates,))
            frame = tables.get(name)
            if frame is None or not len(frame):
                continue
            buffer = io.StringIO()
            frame[columns].to_csv(buffer, index=False, header=False, lineterminator="\n")
            copy = f"COPY monitoring.{name} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)"
            with conn.cursor().copy(copy) as cp:
                cp.write(buffer.getvalue())
    conn.commit()
