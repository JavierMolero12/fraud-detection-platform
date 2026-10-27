"""MLflow tracking: local SQLite store in mlruns/ (git-ignored) plus lineage tags.

Every run records which data it saw (dataset hash, data quality run, feature build) and which
code produced it (git commit, dirty flag), so any number in a report can be traced back.
"""

from __future__ import annotations

import os
import subprocess

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import psycopg  # noqa: E402

from fraudplatform.config import PROJECT_ROOT  # noqa: E402

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def tracking_uri() -> str:
    default = f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').as_posix()}"
    return os.environ.get("MLFLOW_TRACKING_URI", default)


def experiment_id(name: str) -> str:
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifacts = (MLRUNS_DIR / "artifacts" / name).resolve()
    return mlflow.create_experiment(name, artifact_location=artifacts.as_uri())


UNKNOWN = "unknown"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version() -> tuple[str, str]:
    """(commit, dirty) of the code that is running.

    From git when there is a repository. In a Docker image there is none: the commit is baked
    in at build time (FRAUD_GIT_COMMIT, from the GIT_COMMIT build argument), and whether that
    working tree had uncommitted changes cannot be known, so it is "unknown", never a guess.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is not None:
        return commit, str(bool(_git("status", "--porcelain")))
    return os.environ.get("FRAUD_GIT_COMMIT") or UNKNOWN, UNKNOWN


def lineage_tags(conn: psycopg.Connection) -> dict[str, str]:
    dataset_hash, simulator_version = conn.execute(
        "SELECT dataset_hash, simulator_version FROM raw.load_log ORDER BY load_id DESC LIMIT 1"
    ).fetchone()
    build_id, feature_version, dq_run_id = conn.execute(
        "SELECT build_id, feature_set_version, dq_run_id FROM features.builds "
        "ORDER BY build_id DESC LIMIT 1"
    ).fetchone()
    commit, dirty = code_version()
    return {
        "dataset_hash": dataset_hash,
        "simulator_version": simulator_version,
        "dq_run_id": str(dq_run_id),
        "feature_build_id": str(build_id),
        "feature_set_version": feature_version,
        "git_commit": commit,
        "git_dirty": dirty,
    }
