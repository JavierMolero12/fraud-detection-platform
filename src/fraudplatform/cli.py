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
app.add_typer(db_app, name="db")
app.add_typer(dq_app, name="dq")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(policy_app, name="policy")
app.add_typer(analyse_app, name="analyse")
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


if __name__ == "__main__":
    app()
