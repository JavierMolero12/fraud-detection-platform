"""Monitoring on a small synthetic world: a stable reference, then production with a drift in
the payment mix and more fraud than the model expects from day 30."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudplatform.config import (
    InputDriftParams,
    LabelMonitoringParams,
    MonitoringConfig,
    MonitoringEvaluationParams,
    Period,
    PredictionDriftParams,
    ReplayParams,
    RetrainingParams,
)
from fraudplatform.monitoring import evaluation as ev
from fraudplatform.monitoring.drift import bin_counts, fit_binning, psi
from fraudplatform.monitoring.labels import (
    DelayDistribution,
    expected_reported,
    poisson_two_sided_p,
    recall_drop_p,
)
from fraudplatform.monitoring.report import build_report
from fraudplatform.monitoring.run import (
    assemble,
    monitor_as_of,
    replay,
    retraining_actions,
)

FEATURES = ["amount_eur", "payment_method", "is_night"]  # numeric, categorical, boolean
START = date(2026, 1, 1)
DRIFT = date(2026, 1, 31)


def _payments(rng, start: date, days: int, per_day: int, online_share, fraud_factor):
    n = days * per_day
    offsets = rng.uniform(0, days * 86400, n)
    ts = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.sort(offsets), unit="s")
    day = np.asarray((ts - pd.Timestamp(start, tz="UTC")).days)
    online = rng.random(n) < np.where(day >= (DRIFT - start).days, online_share[1], online_share[0])
    score = np.clip(rng.beta(0.5, 40, n), 1e-6, 0.99)
    is_fraud = rng.random(n) < np.clip(
        score * np.where(day >= (DRIFT - start).days, fraud_factor[1], fraud_factor[0]), 0, 1
    )
    decision = np.select([score > 0.3, score > 0.05], ["block", "review"], "approve")
    return pd.DataFrame(
        {
            "transaction_id": [f"T{start:%y%m}{i:07d}" for i in range(n)],
            "ts": ts,
            "amount_eur": rng.lognormal(3, 1, n),
            "payment_method": np.where(online, "ecommerce", "chip_pin"),
            "is_night": (rng.random(n) < 0.1).astype(float),
            "fraud_probability": score,
            "decision": decision,
            "is_fraud": is_fraud,
        }
    )


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(7)
    reference = _payments(rng, date(2025, 7, 1), 60, 400, (0.4, 0.4), (1.0, 1.0))
    production = _payments(rng, START, 60, 400, (0.4, 0.8), (1.0, 2.0))
    delays = rng.exponential(12, 5000)
    frauds = production[production["is_fraud"]]
    delay = pd.to_timedelta(rng.exponential(12, len(frauds)) * 86400, unit="s")
    reports = pd.DataFrame(
        {
            "transaction_id": frauds["transaction_id"].to_numpy(),
            "reported_at": frauds["ts"].reset_index(drop=True) + delay,
        }
    )
    data = assemble(
        FEATURES, reference, production.drop(columns="is_fraud"), reports, delays, START, 10
    )
    return data, reference, production


@pytest.fixture(scope="module")
def cfg():
    return MonitoringConfig(
        reference=Period(start=date(2025, 7, 1), end=date(2025, 8, 30)),
        replay=ReplayParams(
            first_as_of=START + timedelta(days=7),
            last_as_of=START + timedelta(days=84),
            every_days=7,
            drift_window_days=7,
        ),
        input_drift=InputDriftParams(bins=10, psi_warning=0.1, psi_alert=0.25),
        prediction_drift=PredictionDriftParams(
            psi_warning=0.1, psi_alert=0.25, rate_warning=0.25, rate_alert=0.5
        ),
        labels=LabelMonitoringParams(
            delay_from=date(2024, 1, 1),
            delay_to=date(2025, 1, 1),
            known_before=START,
            recent_days=28,
            oe_alert_high=1.25,
            oe_alert_low=0.8,
            p_value=0.01,
            min_expected=30,
            maturity_share=0.9,
            recall_drop_alert=0.05,
        ),
        retraining=RetrainingParams(consecutive_runs=2),
        evaluation=MonitoringEvaluationParams(
            drift_free=Period(start=date(2025, 7, 1), end=date(2025, 8, 30))
        ),
    )


# --- PSI ----------------------------------------------------------------------------------


def test_psi_is_zero_for_identical_shares_and_symmetric():
    a, b = np.array([0.2, 0.3, 0.5, 0.0]), np.array([0.5, 0.3, 0.2, 0.0])
    assert psi(a, a) == pytest.approx(0, abs=1e-12)
    assert psi(a, b) == pytest.approx(psi(b, a))
    assert psi(a, b) > 0.25


def test_an_empty_bin_gives_a_finite_psi():
    assert np.isfinite(psi(np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.4, 0.2])))


def test_zero_inflated_counts_keep_zero_in_its_own_bin():
    ref = pd.Series([0] * 80 + list(range(1, 21)), dtype=float)
    binning = fit_binning(ref, numeric=True, bins=10)
    counts = bin_counts(pd.Series([0.0, 0.0, 5.0]), binning)
    assert counts[0] == 2  # both zeros alone in the first bin
    assert counts.sum() == 3


def test_unseen_categories_and_missing_values_have_their_own_bins():
    binning = fit_binning(pd.Series(["a", "b", "a"]), numeric=False)
    counts = bin_counts(pd.Series(["a", "zzz", None, "b"]), binning)
    assert counts.tolist() == [1, 1, 1, 1]  # a, b, other, missing


# --- delayed labels -------------------------------------------------------------------------


def test_delay_distribution_and_expected_reports():
    delay = DelayDistribution.fit([1, 2, 3, 4, 10])
    assert delay.cdf([0, 2, 10, 99]).tolist() == [0.0, 0.4, 1.0, 1.0]
    assert delay.age_for_share(0.5) == pytest.approx(3)
    # two payments of p=0.5, one old enough for all reports, one too young for any
    assert expected_reported([0.5, 0.5], [100, 0], delay) == pytest.approx(0.5)


def test_statistical_checks():
    assert poisson_two_sided_p(100, 100) > 0.5
    assert poisson_two_sided_p(150, 100) < 1e-4
    assert recall_drop_p(70, 100, 0.85) < 0.01
    assert recall_drop_p(84, 100, 0.85) > 0.2


# --- runs -----------------------------------------------------------------------------------


def test_a_run_never_sees_the_future(world, cfg):
    data, _, _ = world
    as_of = DRIFT + timedelta(days=5)
    before = monitor_as_of(data, cfg, as_of)

    cutoff = pd.Timestamp(as_of, tz="UTC")
    future = data.production["ts"] >= cutoff
    altered = data.production.copy()
    altered.loc[future, "payment_method"] = "mobile_wallet"
    altered.loc[future, "fraud_probability"] = 0.9
    reports = data.fraud_reports[data.fraud_reports["reported_at"] < cutoff]
    data_altered = type(data)(**{**data.__dict__, "production": altered, "fraud_reports": reports})
    after = monitor_as_of(data_altered, cfg, as_of)

    for name in ("feature_drift", "metrics", "label_cohorts", "alerts"):
        pd.testing.assert_frame_equal(getattr(before, name), getattr(after, name))


def test_input_drift_appears_after_the_drift_only(world, cfg):
    data, _, _ = world
    early = monitor_as_of(data, cfg, DRIFT - timedelta(days=3)).feature_drift.set_index("feature")
    late = monitor_as_of(data, cfg, DRIFT + timedelta(days=14)).feature_drift.set_index("feature")
    assert early.loc["payment_method", "psi"] < 0.02
    assert late.loc["payment_method", "psi"] > 0.25
    assert late.loc["amount_eur", "psi"] < 0.05
    assert np.isnan(late.loc["payment_method", "ks"]) and late.loc["amount_eur", "ks"] < 0.1


def test_more_reported_fraud_than_expected_raises_the_label_alert(world, cfg):
    data, _, _ = world
    early = monitor_as_of(data, cfg, DRIFT - timedelta(days=1))
    late = monitor_as_of(data, cfg, DRIFT + timedelta(days=35))
    early_oe = early.metrics.set_index("metric").loc["recent_observed_to_expected", "value"]
    late_oe = late.metrics.set_index("metric").loc["recent_observed_to_expected", "value"]
    assert 0.8 < early_oe < 1.25
    assert late_oe > 1.5
    assert "label_calibration" not in set(early.alerts["rule"])
    assert "label_calibration" in set(late.alerts["rule"])


def test_only_cohorts_old_enough_are_mature(world, cfg):
    data, _, _ = world
    as_of = START + timedelta(days=56)
    cohorts = monitor_as_of(data, cfg, as_of).label_cohorts
    age_of_end = [(as_of - (c + timedelta(days=7))).days for c in cohorts["cohort_start"]]
    maturity = data.delay.age_for_share(cfg.labels.maturity_share)
    assert cohorts["mature"].tolist() == [a >= maturity for a in age_of_end]
    assert (cohorts["known_frauds"] <= cohorts["payments"]).all()


def test_retraining_needs_consecutive_label_alerts():
    days = [date(2026, 1, 1) + timedelta(days=7 * i) for i in range(5)]
    alerts = pd.DataFrame(
        {
            "as_of": [days[0], days[2], days[3], days[3], days[4]],
            "rule": ["label_calibration", "label_calibration", "input_psi", "mature_recall", "x"],
            "subject": ["recent"] * 5,
            "severity": ["alert"] * 5,
            "value": [1.0] * 5,
            "threshold": [1.0] * 5,
            "detail": [""] * 5,
        }
    )
    actions = retraining_actions(alerts, days, runs=2)
    assert actions["as_of"].tolist() == [days[3]]  # runs 2 and 3; input drift does not count


# --- evaluation and report ------------------------------------------------------------------


def test_replay_evaluation_and_report(world, cfg):
    data, reference, production = world
    tables = replay(data, cfg)
    timeline = ev.alert_timeline(tables["alerts"], DRIFT)
    psi_row = timeline[timeline["rule"] == "input_psi"].iloc[0]
    assert psi_row["runs_before_drift"] == 0 and psi_row["days_after_drift"] <= 14

    weeks = (production["ts"] - pd.Timestamp(START, tz="UTC")).dt.days // 7
    truth = pd.DataFrame(
        {
            "week_start": [START + timedelta(days=7 * int(k)) for k in sorted(weeks.unique())],
            "true_frauds": production.groupby(weeks)["is_fraud"].sum().to_numpy(),
            "drift_scenario_frauds": 0,
            "reported_frauds": 0,
            "recall_reported": 0.8,
            "recall_drift_scenario": np.nan,
            "mean_predicted": 0.01,
            "observed_rate": 0.01,
        }
    )
    result = ev.evaluate(
        tables, data, reference, DRIFT, truth, 7, 0.9, 0.05, rolling_threshold=0.03
    )
    assert result["summary"]["rolling_first_days"] is not None
    page = build_report(tables, result, data.reference_metrics, DRIFT, START, cfg.replay.last_as_of)
    assert page.startswith("<!doctype html>")
    assert page.count("<svg") == 7 and page.count("<details>") == 7
    assert "prefers-color-scheme: dark" in page


def test_rolling_reference_flags_a_step_not_a_stable_period(world):
    data, reference, production = world
    roll = ev.rolling_psi(production, data, 7, previous=4)
    step = roll[(roll["feature"] == "payment_method")].set_index("week_start")["psi"]
    assert step.max() > 0.25
    assert ev.rolling_psi(reference, data, 7, previous=4)["psi"].max() < 0.02


# --- the simulator's truth stays out of monitoring and models ---------------------------------

ALLOWED = {
    "simulation",  # writes it
    "db.py",  # loads it
    "analysis/errors.py",  # explains errors with it (Phase 9)
    "monitoring/evaluation.py",  # judges the alerts with it, afterwards
    "quality/report.py",  # recall of each data quality rule vs the injected defects (Phase 3)
    "sql/evaluation/",  # same, in SQL
}


def test_only_evaluation_code_reads_the_simulator_truth():
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"\bsim\.(ground_truth|dq_defects)\b|sim/ground_truth")
    offenders = []
    for path in [*(root / "src" / "fraudplatform").rglob("*.py"), *(root / "sql").rglob("*.sql")]:
        rel = path.relative_to(root).as_posix()
        if any(part in rel for part in ALLOWED) or rel.startswith("sql/schema/"):
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders


def test_test_files_have_unique_names():
    """pytest imports test files by base name: a duplicate in unit/ and integration/ breaks
    collection of the whole suite (it happened in phases 10 and 12)."""
    root = Path(__file__).resolve().parents[1]
    names = [p.name for p in root.rglob("test_*.py")]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)
