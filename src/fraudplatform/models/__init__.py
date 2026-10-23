"""Baselines, model pipelines, evaluation metrics and experiment tracking."""

import os

# MLflow prints an LLM-tracing hint on import; it is irrelevant for this project.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
