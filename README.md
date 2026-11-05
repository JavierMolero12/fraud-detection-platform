# Transaction Fraud Detection & Risk Scoring Platform

End-to-end fraud risk scoring system built around the problems that make fraud detection hard in
practice: temporal leakage, delayed labels, extreme class imbalance, cost-sensitive decisions,
calibration and drift.

> **Status:** work in progress. Phase 2 of 14 (synthetic data + PostgreSQL).
> No model results exist yet; this README will only report numbers produced by real runs.

## Design

- Technical design (Phase 1): [`docs/phase1_design.md`](docs/phase1_design.md)
- Decision log: [`docs/decisions.md`](docs/decisions.md)

The design documents are currently in Spanish; they will be translated in the documentation phase.

## Quickstart

Requirements: Python 3.11+, Docker.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

fraud simulate --scale small       # writes data/raw/v<version>-small/
docker compose up -d db            # PostgreSQL 16 on localhost:5433
fraud db init                      # creates schemas and tables
fraud db load data/raw/v1.0.0-small
```

Scales: `tiny` (tests), `small` (development, ~48k transactions), `full` (~2.9M transactions).
Report of the full dataset: [`reports/simulation/v1.0.0-full.md`](reports/simulation/v1.0.0-full.md).

## Tests

```bash
pytest                    # unit tests
pytest -m integration     # needs `docker compose up -d db`
ruff check . && ruff format --check .
```
