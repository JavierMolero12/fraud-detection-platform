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
app.add_typer(db_app, name="db")
app.add_typer(dq_app, name="dq")
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


if __name__ == "__main__":
    app()
