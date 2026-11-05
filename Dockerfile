# syntax=docker/dockerfile:1
#
# Two images from one file (decision 027):
#   app    the scoring API and the `fraud serve` commands. Base dependencies only.
#   train  every `fraud` command, phases 2-10 (adds MLflow, SHAP, matplotlib). For the
#          `jobs` service in docker-compose.yml.
# Dependencies come from lock files pinned to the environment the model was trained in
# (docker/lock.sh). Data and the model bundle are never baked in: they are mounted.

# python:3.13-slim (Debian), pinned by digest
ARG PYTHON_IMAGE=python:3.13-slim@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp
RUN python -m venv /opt/venv

# --- dependencies: their own layers, rebuilt only when a lock file changes ----------------
FROM base AS deps-app
COPY docker/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

FROM base AS deps-train
COPY docker/requirements-train.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

# --- app: the API ----------------------------------------------------------------------------
FROM deps-app AS app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src build
COPY configs ./configs
COPY sql ./sql
# FRAUD_PROJECT_ROOT: configs/ and sql/ live here, not next to the installed package.
# One OpenMP thread per process: one-row predictions gain nothing from more (decision 026);
# throughput comes from worker processes (WEB_CONCURRENCY). `fraud serve api`, not plain
# `uvicorn --workers`: see serving/server.py (TCP_NODELAY, +40 ms per request otherwise).
ENV FRAUD_PROJECT_ROOT=/app \
    FRAUD_BUNDLE_DIR=/app/bundle \
    OMP_NUM_THREADS=1 \
    WEB_CONCURRENCY=2
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["fraud", "serve", "api", "--host", "0.0.0.0", "--port", "8000"]

# --- train: the whole pipeline ---------------------------------------------------------------
FROM deps-train AS train
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src build
COPY configs ./configs
COPY sql ./sql
ENV FRAUD_PROJECT_ROOT=/app \
    MPLBACKEND=Agg
USER 10001
ENTRYPOINT ["fraud"]
CMD ["--help"]
