"""Command line interface: ``fraud <command>``. One entry point for every pipeline step."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import typer

from fraudplatform.config import PROJECT_ROOT, load_simulation_config
from fraudplatform.logs import setup_logging

app = typer.Typer(no_args_is_help=True, help="Fraud detection platform pipeline.")
db_app = typer.Typer(no_args_is_help=True, help="PostgreSQL schema and loading.")
dq_app = typer.Typer(no_args_is_help=True, help="Data quality: rules, quarantine, raw -> core.")
features_app = typer.Typer(no_args_is_help=True, help="Point-in-time feature engineering.")
models_app = typer.Typer(no_args_is_help=True, help="Baselines, models and evaluation.")
policy_app = typer.Typer(no_args_is_help=True, help="Calibration, costs and decision policy.")
analyse_app = typer.Typer(no_args_is_help=True, help="Explanations and error analysis.")
serve_app = typer.Typer(no_args_is_help=True, help="Real-time API and batch scoring.")
monitor_app = typer.Typer(no_args_is_help=True, help="Drift, delayed performance, alerts.")
app.add_typer(db_app, name="db")
app.add_typer(dq_app, name="dq")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(policy_app, name="policy")
app.add_typer(analyse_app, name="analyse")
app.add_typer(serve_app, name="serve")
app.add_typer(monitor_app, name="monitor")
log = logging.getLogger("fraudplatform")


@app.callback()
def main(log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING")] = "INFO") -> None:
    setup_logging(log_level)


@app.command()
def simulate(
    scale: Annotated[str, typer.Option(help="tiny | small | full")] = "small",
    seed: Annotated[int | None, typer.Option(help="Overrides the seed in the config")] = None,
    out_dir: Annotated[Path, typer.Option(help="Root folder for datasets")] = PROJECT_ROOT
    / "data"
    / "raw",
    config: Annotated[Path | None, typer.Option(help="Simulation YAML")] = None,
) -> None:
    """Generate a synthetic dataset (raw extract + reference data + simulator ground truth)."""
    from fraudplatform.simulation.generator import simulate as run_simulation
    from fraudplatform.simulation.writer import write_dataset

    started = time.perf_counter()
    result = run_simulation(load_simulation_config(config), scale, seed)
    path = write_dataset(result, out_dir)
    log.info("dataset written to %s in %.1fs", path, time.perf_counter() - started)
    typer.echo(path)


@db_app.command("init")
def db_init() -> None:
    """Create schemas, tables, constraints and indexes (idempotent)."""
    from fraudplatform.db import connect, init_schema

    with connect() as conn:
        init_schema(conn)


@db_app.command("load")
def db_load(dataset_dir: Annotated[Path, typer.Argument(help="Folder written by `simulate`")]):
    """Replace raw, reference and sim tables with the content of a dataset folder."""
    from fraudplatform.db import connect, load_dataset

    started = time.perf_counter()
    with connect() as conn:
        load_dataset(conn, dataset_dir)
    log.info("load finished in %.1fs", time.perf_counter() - started)


@dq_app.command("run")
def dq_run(
    report: Annotated[Path | None, typer.Option(help="Markdown report path")] = PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "dq_report.md",
) -> None:
    """Apply the data quality rules to raw, quarantine bad rows and rebuild core."""
    from fraudplatform.db import connect
    from fraudplatform.quality.report import build_report
    from fraudplatform.quality.runner import run_data_quality

    started = time.perf_counter()
    with connect() as conn:
        run_id = run_data_quality(conn)
        log.info("data quality run %s finished in %.1fs", run_id, time.perf_counter() - started)
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(build_report(conn, run_id), encoding="utf-8", newline="\n")
            typer.echo(report)


@features_app.command("build")
def features_build(
    config: Annotated[Path | None, typer.Option(help="Feature YAML")] = None,
) -> None:
    """Rebuild features.transaction_features (and the leaky variant) from core."""
    from fraudplatform.config import load_feature_config
    from fraudplatform.db import connect
    from fraudplatform.features.builder import build_features

    with connect() as conn:
        build_features(conn, load_feature_config(config))


@features_app.command("catalog")
def features_catalog(
    out: Annotated[Path, typer.Option(help="Markdown output")] = PROJECT_ROOT
    / "docs"
    / "feature_catalog.md",
) -> None:
    """Render the feature catalog to Markdown."""
    from fraudplatform.features.catalog import render_markdown

    out.write_text(render_markdown(), encoding="utf-8", newline="\n")
    typer.echo(out)


@models_app.command("compare")
def models_compare() -> None:
    """Fit baselines and models on the train period and compare them on validation (Phase 6)."""
    from fraudplatform.config import load_feature_config, load_model_config, load_splits
    from fraudplatform.db import connect
    from fraudplatform.models.comparison import run_comparison
    from fraudplatform.models.report import write_report
    from fraudplatform.models.tracking import lineage_tags

    cfg, feature_cfg, splits = load_model_config(), load_feature_config(), load_splits()
    with connect() as conn:
        results, scores = run_comparison(conn, cfg, feature_cfg, splits)
        tags = lineage_tags(conn)
    out = PROJECT_ROOT / "data" / "processed" / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "validation_scores.parquet", index=False)
    results.to_csv(out / "results.csv", index=False)
    context = {
        "dataset": f"v{tags['simulator_version']} `{tags['dataset_hash'][:12]}…`",
        "feature set": f"v{tags['feature_set_version']} (build {tags['feature_build_id']})",
        "code": f"`{tags['git_commit'][:10]}` (dirty: {tags['git_dirty']})",
    }
    report = write_report(
        results,
        scores,
        cfg.evaluation.alert_rate,
        PROJECT_ROOT / "reports" / "models" / "phase6_comparison.md",
        context,
    )
    typer.echo(report)


@models_app.command("tune")
def models_tune() -> None:
    """Temporal CV, hyperparameter search and leakage experiments (Phase 7)."""
    from fraudplatform.config import load_feature_config, load_model_config, load_splits
    from fraudplatform.db import connect
    from fraudplatform.models.tuning import run_phase7

    cfg, feature_cfg, splits = load_model_config(), load_feature_config(), load_splits()
    out = PROJECT_ROOT / "data" / "processed" / "phase7"
    started = time.perf_counter()
    with connect() as conn:
        run_phase7(conn, cfg, feature_cfg, splits, out)
    log.info("phase 7 finished in %.0fs; results in %s", time.perf_counter() - started, out)


@policy_app.command("optimise")
def policy_optimise() -> None:
    """Calibrate the selected model and choose the decision policy on validation (Phase 8)."""
    from fraudplatform.config import (
        load_cost_config,
        load_feature_config,
        load_model_config,
        load_splits,
    )
    from fraudplatform.db import connect
    from fraudplatform.decision.optimise import run_optimisation

    with connect() as conn:
        run_optimisation(
            conn,
            load_model_config(),
            load_feature_config(),
            load_cost_config(),
            load_splits(),
            PROJECT_ROOT / "data" / "processed" / "phase8",
        )


@policy_app.command("evaluate-test")
def policy_evaluate_test(
    force: Annotated[bool, typer.Option(help="Re-run although the test was already used")] = False,
) -> None:
    """Score the frozen model + policy on the test period. Runs ONCE (see test_evaluation.py)."""
    from fraudplatform.config import load_model_config, load_splits
    from fraudplatform.db import connect
    from fraudplatform.decision.test_evaluation import evaluate_on_test

    with connect() as conn:
        evaluate_on_test(
            conn,
            load_model_config(),
            load_splits(),
            PROJECT_ROOT / "data" / "processed" / "phase8",
            force=force,
        )


@analyse_app.command("validation")
def analyse_validation(
    quick: Annotated[bool, typer.Option(help="3 days only: smoke test of the whole job")] = False,
) -> None:
    """Explanations, error and segment analysis of the frozen bundle on validation (Phase 9)."""
    from fraudplatform.analysis.run import run_phase9
    from fraudplatform.config import load_model_config, load_splits
    from fraudplatform.db import connect

    with connect() as conn:
        run_phase9(
            conn,
            load_model_config(),
            load_splits(),
            PROJECT_ROOT / "data" / "processed" / "phase8",
            PROJECT_ROOT / "data" / "processed" / ("phase9_quick" if quick else "phase9"),
            quick=quick,
        )


def _period(start: str | None, end: str | None):
    """Default: the production split (after the test period)."""
    from datetime import date

    from fraudplatform.config import load_splits

    production = load_splits().production
    return (
        date.fromisoformat(start) if start else production.start,
        date.fromisoformat(end) if end else production.end,
    )


@serve_app.command("api")
def serve_api(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    workers: Annotated[
        int,
        typer.Option(
            envvar="WEB_CONCURRENCY",
            help="Worker processes (on Windows >1 may fail: decision 026)",
        ),
    ] = 1,
) -> None:
    """Run the scoring API (POST /predict, GET /health, GET /model)."""
    from fraudplatform.serving.server import run_api

    run_api(host, port, workers)


@serve_app.command("batch")
def serve_batch(
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD, default: production start")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD, exclusive")] = None,
    write_db: Annotated[bool, typer.Option(help="Also append to scoring.predictions")] = True,
) -> None:
    """Score a period with the frozen bundle: CSV + scoring.predictions."""
    from fraudplatform.config import load_serving_config
    from fraudplatform.db import connect
    from fraudplatform.serving.batch import run_batch
    from fraudplatform.serving.scorer import Bundle, Scorer

    first, last = _period(start, end)
    serving = load_serving_config()
    scorer = Scorer(Bundle.load(serving.bundle_dir), serving.risk_levels)
    name = f"batch_{first:%Y%m%d}_{last:%Y%m%d}"
    with connect() as conn:
        summary = run_batch(
            conn,
            scorer,
            first,
            last,
            PROJECT_ROOT / "data" / "processed" / "scoring" / f"{name}.csv",
            PROJECT_ROOT / "reports" / "serving" / f"{name}_summary.json",
            write_db=write_db,
        )
    typer.echo(summary)


@serve_app.command("bench")
def serve_bench(
    sequential: Annotated[int, typer.Option(help="Requests sent one at a time")] = 2000,
    concurrent: Annotated[
        int, typer.Option(help="Requests from several threads, per process count")
    ] = 2000,
    threads: Annotated[int, typer.Option(help="Client threads for the concurrent runs")] = 8,
    processes: Annotated[
        list[int] | None, typer.Option(help="API process counts to compare (repeatable)")
    ] = None,
    port: Annotated[int, typer.Option()] = 8765,
    start: Annotated[str | None, typer.Option(help="Sample period start")] = None,
    end: Annotated[str | None, typer.Option(help="Sample period end (exclusive)")] = None,
    batch_csv: Annotated[Path | None, typer.Option(help="Compare with this batch output")] = None,
    keep: Annotated[bool, typer.Option(help="Keep the benchmark rows in the log")] = False,
    repeats: Annotated[int, typer.Option(help="Repeat the measurement; report the median")] = 3,
    url: Annotated[
        str | None, typer.Option(help="Measure this running API (e.g. the container) instead")
    ] = None,
    output: Annotated[Path, typer.Option(help="Result file")] = PROJECT_ROOT
    / "reports"
    / "serving"
    / "latency.json",
) -> None:
    """Latency and throughput of the real API on replayed production payments."""
    from fraudplatform.db import connect
    from fraudplatform.serving.bench import run_bench

    first, last = _period(start, end)
    with connect() as conn:
        result = run_bench(
            conn,
            first,
            last,
            sequential,
            concurrent,
            threads,
            processes or [1, 4],
            port,
            output,
            batch_csv=batch_csv,
            keep=keep,
            repeats=repeats,
            url=url,
        )
    typer.echo(result["median_over_repeats"])
    typer.echo(result.get("agreement_with_batch"))


@serve_app.command("profile")
def serve_profile(
    payments: Annotated[int, typer.Option(help="Payments scored one at a time")] = 500,
    start: Annotated[str | None, typer.Option(help="Sample period start")] = None,
    end: Annotated[str | None, typer.Option(help="Sample period end (exclusive)")] = None,
) -> None:
    """Milliseconds per step of one online prediction (features, model, log...)."""
    import psycopg

    from fraudplatform.db import database_url
    from fraudplatform.serving.bench import run_profile

    first, last = _period(start, end)
    with psycopg.connect(database_url(), autocommit=True) as conn:
        result = run_profile(
            conn,
            first,
            last,
            payments,
            PROJECT_ROOT / "reports" / "serving" / "latency_breakdown.json",
        )
    typer.echo(result)


def _bundle_dir(bundle_dir: Path | None) -> Path:
    from fraudplatform.config import load_serving_config

    return bundle_dir or load_serving_config().bundle_dir


@serve_app.command("demo-bundle")
def serve_demo_bundle(
    out: Annotated[Path, typer.Option(help="Bundle folder")] = PROJECT_ROOT
    / "data"
    / "processed"
    / "demo_bundle",
) -> None:
    """Train a small stand-in bundle on the loaded data (NOT the Phase 8 model): smoke tests."""
    from fraudplatform.db import connect
    from fraudplatform.serving.demo import build_demo_bundle

    with connect() as conn:
        typer.echo(build_demo_bundle(conn, out))


@serve_app.command("check")
def serve_check(
    url: Annotated[str, typer.Option(help="Running API, e.g. http://api:8000")],
    payments: Annotated[int, typer.Option(help="Stored payments to send")] = 50,
    bundle_dir: Annotated[
        Path | None, typer.Option(help="The bundle the API serves (default: serving.yaml)")
    ] = None,
) -> None:
    """A running API must give exactly the batch probabilities and decisions. Exit 1 if not."""
    from fraudplatform.config import load_serving_config
    from fraudplatform.db import connect
    from fraudplatform.serving.checks import check_api, check_passed
    from fraudplatform.serving.scorer import Bundle, Scorer

    scorer = Scorer(Bundle.load(_bundle_dir(bundle_dir)), load_serving_config().risk_levels)
    with connect() as conn:
        result = check_api(conn, url, scorer, payments)
    typer.echo(result)
    if not check_passed(result):
        raise typer.Exit(code=1)


@serve_app.command("verify-bundle")
def serve_verify_bundle(
    bundle_dir: Annotated[Path | None, typer.Option(help="Default: serving.yaml")] = None,
) -> None:
    """This environment must reproduce the bundle's saved validation scores. Exit 1 if not."""
    from fraudplatform.config import load_serving_config
    from fraudplatform.db import connect
    from fraudplatform.serving.checks import verify_bundle
    from fraudplatform.serving.scorer import Bundle, Scorer

    folder = _bundle_dir(bundle_dir)
    scorer = Scorer(Bundle.load(folder), load_serving_config().risk_levels)
    with connect() as conn:
        result = verify_bundle(conn, scorer, folder)
    typer.echo(result)
    if not result["passed"]:
        raise typer.Exit(code=1)


def _monitoring_data(conn):
    from fraudplatform.config import (
        load_monitoring_config,
        load_serving_config,
        load_splits,
    )
    from fraudplatform.monitoring.run import load_monitoring_data
    from fraudplatform.serving.scorer import Bundle, Scorer

    cfg = load_monitoring_config()
    serving = load_serving_config()
    scorer = Scorer(Bundle.load(serving.bundle_dir), serving.risk_levels)
    return cfg, load_monitoring_data(conn, cfg, scorer, load_splits().production)


@monitor_app.command("run")
def monitor_run(
    as_of: Annotated[str, typer.Option(help="Monitoring date YYYY-MM-DD (data before it)")],
) -> None:
    """One monitoring run, as it would run on that date. Stored in the monitoring schema."""
    from datetime import date

    from fraudplatform.db import connect
    from fraudplatform.monitoring.run import monitor_as_of, write_tables

    with connect() as conn:
        cfg, data = _monitoring_data(conn)
        result = monitor_as_of(data, cfg, date.fromisoformat(as_of))
        day = result.as_of
        tables = {
            name: getattr(result, name).assign(as_of=day)
            for name in ("feature_drift", "metrics", "label_cohorts", "alerts")
        }
        write_tables(conn, tables)
    typer.echo(result.alerts.to_string(index=False) if len(result.alerts) else "no alerts")


@monitor_app.command("replay")
def monitor_replay() -> None:
    """Every weekly run since deployment (configs/monitoring.yaml), into the monitoring schema."""
    from fraudplatform.db import connect
    from fraudplatform.monitoring.run import replay, write_tables

    with connect() as conn:
        cfg, data = _monitoring_data(conn)
        tables = replay(data, cfg)
        write_tables(conn, tables)
    alerts = tables["alerts"]
    runs = len(cfg.replay.as_of_dates())
    typer.echo(f"{len(alerts)} alerts in {alerts['as_of'].nunique()} of {runs} runs")


@monitor_app.command("report")
def monitor_report() -> None:
    """Judge the replayed alerts against the simulator's truth; HTML report + tables."""
    import json

    from fraudplatform.config import load_simulation_config
    from fraudplatform.db import connect
    from fraudplatform.eda.data import load_features
    from fraudplatform.models.pipelines import prepare
    from fraudplatform.monitoring.evaluation import evaluate, weekly_truth
    from fraudplatform.monitoring.report import build_report
    from fraudplatform.monitoring.run import read_tables

    out = PROJECT_ROOT / "reports" / "monitoring"
    out.mkdir(parents=True, exist_ok=True)
    drift_start = load_simulation_config().drift.start  # the truth: evaluation only
    with connect() as conn:
        cfg, data = _monitoring_data(conn)
        tables = read_tables(conn, cfg.replay.as_of_dates())
        free = cfg.evaluation.drift_free
        drift_free = prepare(load_features(conn, free.start, free.end, with_label=False))
        truth = weekly_truth(conn, data)
    result = evaluate(
        tables,
        data,
        drift_free,
        drift_start,
        truth,
        cfg.replay.drift_window_days,
        cfg.labels.maturity_share,
        cfg.labels.recall_drop_alert,
    )
    page = build_report(
        tables,
        result,
        data.reference_metrics,
        drift_start,
        data.production_start.date(),
        cfg.replay.last_as_of,
    )
    (out / "monitoring_report.html").write_text(page, encoding="utf-8", newline="\n")
    result["timeline"].to_csv(out / "alert_timeline.csv", index=False, lineterminator="\n")
    result["posthoc"].to_csv(out / "posthoc_rules.csv", index=False, lineterminator="\n")
    truth.to_csv(out / "weekly_truth.csv", index=False, lineterminator="\n")
    summary = result["summary"] | {
        "reference": data.reference_metrics,
        "noisy_features": result["noisy_features"],
        "maturity_days": result["maturity_days"],
        "drift_start": str(drift_start),
    }
    with open(out / "summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, default=str)
        fh.write("\n")
    typer.echo(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
