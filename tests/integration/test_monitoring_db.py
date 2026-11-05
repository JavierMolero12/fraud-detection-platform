"""Monitoring against a real database: the small dataset, the demo bundle, batch scores of the
last two months as "production"."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fraudplatform.config import (
    Period,
    RiskLevels,
    load_feature_config,
    load_monitoring_config,
)
from fraudplatform.db import load_dataset
from fraudplatform.features.builder import build_features
from fraudplatform.monitoring.run import (
    TABLE_COLUMNS,
    load_monitoring_data,
    read_tables,
    replay,
    write_tables,
)
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.serving.batch import score_period, write_predictions
from fraudplatform.serving.demo import build_demo_bundle
from fraudplatform.serving.scorer import Bundle, Scorer
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration

PRODUCTION = Period(start=date(2025, 5, 1), end=date(2025, 7, 1))


@pytest.fixture(scope="module")
def setup(conn, small, tmp_path_factory):
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    build_features(conn, load_feature_config())
    scorer = Scorer(
        Bundle.load(build_demo_bundle(conn, tmp_path_factory.mktemp("bundle"))),
        RiskLevels(medium_from=0.01, high_from=0.25),
    )
    base = load_monitoring_config()
    cfg = base.model_copy(
        update={
            "reference": Period(start=date(2025, 3, 1), end=date(2025, 5, 1)),
            "replay": base.replay.model_copy(
                update={"first_as_of": date(2025, 5, 15), "last_as_of": date(2025, 7, 31)}
            ),
            "labels": base.labels.model_copy(
                update={
                    "delay_from": date(2025, 1, 1),
                    "delay_to": date(2025, 3, 1),
                    "known_before": date(2025, 5, 1),
                    "min_expected": 5,
                }
            ),
        }
    )
    return cfg, scorer


def test_monitoring_needs_the_batch_scores_first(conn, setup):
    cfg, scorer = setup
    with pytest.raises(ValueError, match="fraud serve batch"):
        load_monitoring_data(conn, cfg, scorer, PRODUCTION)


def test_replay_writes_and_reads_back_the_monitoring_tables(conn, setup):
    cfg, scorer = setup
    _, rows = score_period(conn, scorer, PRODUCTION.start, PRODUCTION.end)
    write_predictions(conn, rows, PRODUCTION.start, PRODUCTION.end)

    data = load_monitoring_data(conn, cfg, scorer, PRODUCTION)
    assert len(data.production) == len(rows)
    assert set(data.features) == set(scorer.bundle.columns)
    assert 0 < data.reference_metrics["recall"] <= 1

    tables = replay(data, cfg)
    dates = cfg.replay.as_of_dates()
    assert set(tables["metrics"]["as_of"]) == set(dates)
    # drift windows stop with the payments, label cohorts keep maturing after them
    assert tables["feature_drift"]["as_of"].max() <= PRODUCTION.end + pd.Timedelta(days=7)
    assert tables["label_cohorts"]["as_of"].max() == dates[-1]

    write_tables(conn, tables)
    write_tables(conn, tables)  # a re-run replaces, it does not duplicate
    stored = read_tables(conn, dates)
    for name, columns in TABLE_COLUMNS.items():
        assert len(stored[name]) == len(tables[name]), name
        assert list(stored[name].columns) == columns
    pd.testing.assert_series_equal(
        stored["feature_drift"].sort_values(["as_of", "feature"])["psi"].reset_index(drop=True),
        tables["feature_drift"].sort_values(["as_of", "feature"])["psi"].reset_index(drop=True),
        check_names=False,
    )
