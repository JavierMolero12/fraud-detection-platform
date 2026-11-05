# Transaction Fraud Detection & Risk Scoring Platform

End-to-end fraud risk scoring system built around the problems that make fraud detection hard in
practice: temporal leakage, delayed labels, extreme class imbalance, cost-sensitive decisions,
calibration and drift.

> **Status:** work in progress. Phase 12 of 14 (monitoring + drift).
> Every number in the reports comes from a recorded run.

## Design

- Technical design (Phase 1): [`docs/phase1_design.md`](docs/phase1_design.md)
- Decision log: [`docs/decisions.md`](docs/decisions.md)

The design documents are currently in Spanish; they will be translated in the documentation phase.

## Quickstart

Requirements: Python 3.11+, Docker. Everything below also runs in containers (next section).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[train,dev]"        # train: MLflow, SHAP, matplotlib; add ,analysis for notebooks

fraud simulate --scale small       # writes data/raw/v<version>-small/
docker compose up -d db            # PostgreSQL 16 on localhost:5433
fraud db init                      # creates schemas and tables
fraud db load data/raw/v1.0.1-small
fraud dq run                       # rules, quarantine, raw -> core, report
fraud features build               # 50 point-in-time features (+ leaky variants)
fraud models compare               # baselines + models, train -> validation, MLflow
fraud models tune                  # temporal CV, tuning, leakage experiments (~35 min)
fraud policy optimise              # calibration + cost-based policy on validation
fraud policy evaluate-test         # the ONE evaluation on the test period
fraud analyse validation           # explanations, error and segment analysis (~17 min)
fraud serve batch                  # score the production period: CSV + scoring.predictions
fraud serve api                    # real-time API on http://127.0.0.1:8000 (docs at /docs)
fraud serve bench                  # latency and throughput of the real API
fraud serve verify-bundle          # this environment reproduces the frozen model's scores
fraud monitor replay               # weekly monitoring runs since deployment, as of each date
fraud monitor report               # alerts vs the injected drift: HTML report + tables
```

The API and the batch job load the frozen bundle written by `fraud policy optimise`
(`data/processed/phase8/`, not in git; override with `FRAUD_BUNDLE_DIR`).

Scales: `tiny` (tests), `small` (development, ~48k transactions), `full` (~2.9M transactions).
Reports of the full dataset: [simulation](reports/simulation/v1.0.1-full.md), [data quality](reports/data_quality/dq_report.md).

## Docker

Two images from one [`Dockerfile`](Dockerfile) (decision 027). Both pin the libraries to the
versions the model was trained with ([`docker/`](docker/)).
- `app`: the API and the `fraud serve` commands.
- `train`: every `fraud` command.

The model bundle is mounted read-only, never copied into an image.

```bash
docker compose up -d api                              # db + API on http://localhost:8000
docker compose run --rm jobs features build           # any `fraud` command, in a container
docker compose run --rm --entrypoint fraud api serve verify-bundle   # the served env vs Phase 8
```

- **Without the Phase 8 bundle** (it is not in git), train a stand-in on the loaded data:
  `docker compose run --rm jobs serve demo-bundle`, then start the API with
  `FRAUD_BUNDLE_HOST_DIR=./data/processed/demo_bundle`. CI does exactly this on the tiny dataset
  ([`ci.yml`](.github/workflows/ci.yml)).
- **Settings:** `API_WORKERS` (default 2) and `API_PORT`.
- **On Linux**, run `jobs` as your user so the files it writes stay yours:
  `FRAUD_UID=$(id -u) FRAUD_GID=$(id -g)`.

## Real-time API

`fraud serve api` starts the service. `POST /predict` takes a payment as the authorisation
system sends it (before it reaches the database) and returns a calibrated probability, a risk
level and the decision with the lowest expected cost for that amount:

```bash
curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "transaction_id": "demo-0001", "customer_id": "C0024210", "card_id": "K00038141",
  "merchant_id": "M0001305", "timestamp": "2026-06-30T07:43:24+00:00",
  "amount": 8598.49, "currency": "EUR", "country": "IT",
  "payment_method": "ecommerce", "device_id": "D8949a4286eec"}'
```

```json
{"transaction_id": "demo-0001", "fraud_probability": 0.0020404363442641077,
 "risk_level": "low", "decision": "review", "model_version": "fcf88c67457e",
 "policy_version": "65a2111fa8c2", "scoring_timestamp": "2026-10-26T12:15:00.187080Z"}
```

A 0.2% risk is "low", but on €8,598 a review is cheaper than the expected loss: the risk level
is for display, the decision comes from the cost policy. The ids exist in the full simulated
dataset. Invalid or unknown inputs get a 422; a retried `transaction_id` returns the decision
already made. `GET /health` and `GET /model` (versions, policy, training period, lineage) are
also available, and the interactive docs are at `/docs`.

Measured on a laptop: ~49 ms per request one at a time (p50), ~100 requests/s with several
processes; online features are identical, bit for bit, to the training features. Details:
[serving findings](reports/serving/phase10_findings.md).

## Pipeline so far

| Step | Command | Where |
|---|---|---|
| Synthetic data | `fraud simulate` | `src/fraudplatform/simulation/`, `configs/simulation.yaml` |
| Load raw extract | `fraud db load` | `sql/schema/` |
| Data quality (47 rules) | `fraud dq run` | `sql/data_quality/`, `dq.rules`, `dq.violations` |
| Analytical SQL | `psql -f` | `sql/analysis/` |
| EDA + statistics (train period only) | `jupytext --to ipynb --execute notebooks/02_eda.py` | [`notebooks/02_eda.ipynb`](notebooks/02_eda.ipynb), [findings](reports/eda/eda_findings.md) |
| Point-in-time features (50) | `fraud features build` | `sql/features/`, [catalog](docs/feature_catalog.md), [findings](reports/features/feature_findings.md) |
| Baselines vs models | `fraud models compare` | `src/fraudplatform/models/`, [comparison](reports/models/phase6_comparison.md), [findings](reports/models/phase6_findings.md) |
| Temporal CV, tuning, H7/H8 | `fraud models tune` | `models/validation.py`, `models/tuning.py`, [findings](reports/models/phase7_findings.md) |
| Calibration, cost model, policy, test | `fraud policy optimise`, `fraud policy evaluate-test` | `src/fraudplatform/decision/`, [policy](reports/policy/policy_v1.json), [findings](reports/policy/phase8_findings.md) |
| Error analysis, explanations, model card | `fraud analyse validation` | `src/fraudplatform/analysis/`, [findings](reports/analysis/phase9_findings.md), [model card](reports/model_card.md) |
| Real-time API, batch scoring | `fraud serve api`, `fraud serve batch` | `src/fraudplatform/serving/`, `sql/features/online/`, [findings](reports/serving/phase10_findings.md) |
| Docker images, CI end to end | `docker compose`, GitHub Actions | `Dockerfile`, `docker/`, `.github/workflows/ci.yml`, [findings](reports/deployment/phase11_findings.md) |
| Monitoring: drift, delayed labels, alerts | `fraud monitor replay`, `fraud monitor report` | `src/fraudplatform/monitoring/`, `configs/monitoring.yaml`, [findings](reports/monitoring/phase12_findings.md), [report](reports/monitoring/monitoring_report.html) |

## Tests

```bash
pytest                    # unit tests
pytest -m integration     # needs `docker compose up -d db`
pytest -m "integration or not integration" --cov=fraudplatform   # both, with coverage
ruff check . && ruff format --check .
```
