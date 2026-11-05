"""Logging set-up shared by the CLI, batch jobs and (later) the API."""

from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("FRAUD_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
