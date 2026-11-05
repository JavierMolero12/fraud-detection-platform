# Transaction Fraud Detection & Risk Scoring Platform

End-to-end fraud risk scoring system built around the problems that make fraud detection hard in
practice: temporal leakage, delayed labels, extreme class imbalance, cost-sensitive decisions,
calibration and drift.

> **Status:** work in progress. Phase 7 of 14 (temporal validation + tuning).
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
pip install -e ".[dev]"              # add ,analysis for notebooks

fraud simulate --scale small       # writes data/raw/v<version>-small/
docker compose up -d db            # PostgreSQL 16 on localhost:5433
fraud db init                      # creates schemas and tables
fraud db load data/raw/v1.0.1-small
fraud dq run                       # rules, quarantine, raw -> core, report
fraud features build               # 51 point-in-time features (+ leaky variants)
fraud models compare               # baselines + models, train -> validation, MLflow
fraud models tune                  # temporal CV, tuning, leakage experiments (~35 min)
```

Scales: `tiny` (tests), `small` (development, ~48k transactions), `full` (~2.9M transactions).
Reports of the full dataset: [simulation](reports/simulation/v1.0.1-full.md), [data quality](reports/data_quality/dq_report.md).

## Pipeline so far

| Step | Command | Where |
|---|---|---|
| Synthetic data | `fraud simulate` | `src/fraudplatform/simulation/`, `configs/simulation.yaml` |
| Load raw extract | `fraud db load` | `sql/schema/` |
| Data quality (47 rules) | `fraud dq run` | `sql/data_quality/`, `dq.rules`, `dq.violations` |
| Analytical SQL | `psql -f` | `sql/analysis/` |
| EDA + statistics (train period only) | `jupytext --to ipynb --execute notebooks/02_eda.py` | [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb), [findings](reports/eda/eda_findings.md) |
| Point-in-time features (51) | `fraud features build` | `sql/features/`, [catalog](docs/feature_catalog.md), [findings](reports/features/feature_findings.md) |
| Baselines vs models | `fraud models compare` | `src/fraudplatform/models/`, [comparison](reports/models/phase6_comparison.md), [findings](reports/models/phase6_findings.md) |
| Temporal CV, tuning, H7/H8 | `fraud models tune` | `models/validation.py`, `models/tuning.py`, [findings](reports/models/phase7_findings.md) |

## Tests

```bash
pytest                    # unit tests
pytest -m integration     # needs `docker compose up -d db`
ruff check . && ruff format --check .
```
