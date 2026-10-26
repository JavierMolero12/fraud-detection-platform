# PR 10 — Real-time API and batch scoring

Branch: `feature/api-and-batch` → `main` (stacked on `feature/error-analysis`)

## Objective

Serve the frozen Phase 8 bundle (model + calibration + cost policy) in two ways, with the same
answers: a real-time API that scores one payment at authorisation time, and a batch job that
scores a whole period. Prove that the online features are the training features, and measure
latency.

## Changes

- **Online features**: `sql/features/online/single_transaction.sql` computes the 50 features of
  one new payment with lookups (strictly earlier history, labels reported before `ts`).
  `serving/online_features.py` converts the amount with the latest exchange rate up to 7 days
  old, and reads the features through the same COPY → CSV → pandas function as training
  (`eda.data.query_features`).
- **Scorer** (`serving/scorer.py`): loads `model.skops` + `policy.json`. Model and policy
  versions are the first 12 hex digits of their SHA-256. Calibration, risk levels
  (`configs/serving.yaml`), expected-cost decision.
- **API** (`serving/api.py`, `serving/online.py`): `POST /predict`, `GET /health`, `GET /model`.
  - Pydantic input contract: 422 on invalid or unknown input.
  - Idempotent retries.
  - Daily review capacity enforced under a PostgreSQL advisory lock.
  - Every decision logged to `scoring.predictions` (`sql/schema/06_scoring.sql`) with versions,
    time to decision and the 50 model inputs.
- **Batch** (`serving/batch.py`): offline features, no labels. Writes a CSV
  (`transaction_id, fraud_probability, risk_level, decision, model_version, scoring_timestamp`)
  and `scoring.predictions`; a re-run replaces earlier scores of the same model and period.
- **Measurement** (`serving/bench.py`):
  - `fraud serve bench`: real uvicorn processes, production payments replayed over HTTP, repeated
    measurements, agreement with the batch output.
  - `fraud serve profile`: milliseconds per step of the real code path.
- **CLI**: `fraud serve api | batch | bench | profile`.
- **Dependencies**: `fastapi`, `uvicorn`, `psycopg-pool`, `httpx`, `threadpoolctl`.
- **Docs**: [`reports/serving/phase10_findings.md`](../../reports/serving/phase10_findings.md),
  decisions 024–026, README (status, quickstart, API example), model card (serving row).

The Phase 1 design placed the service in `api/`. It lives in `serving/` instead, because the
scorer is shared by the API and the batch job.

## Validation

| Check | Result |
|---|---|
| Unit tests (24 new: input contract, risk levels, online decisions = batch decisions, capacity fallback, logged features, config, bundle loading) | 116 passed |
| Integration tests (13 new: exact parity, exchange rates, unknown entities, API = batch, logging, retries, unseen device, stale rates, endpoints, step timings, review capacity under 10 concurrent requests, batch output and re-run) | 50 passed |
| Exact parity on the full database: 1,539 production payments × 50 features | 0 differences in 76,950 values |
| API vs batch, 15,000 replayed production payments | max probability difference 0.0, 15,000 identical decisions |
| Latency, one at a time (median of 3 repetitions) | client p50 48.6 ms, server p50 43.8 ms |
| Throughput, 8 client threads, 1 / 2 / 4 / 8 processes | 30.9 / 57.7 / 92.3 / 99.7 requests/s |
| Batch, production period | 760,511 payments in 47.3 s |
| `ruff check`, `ruff format --check` | clean |

## Findings

- **Bit-for-bit parity was necessary (decision 024).** Parity "within 1e-9" passed, yet API and
  batch disagreed on 1 of 150 probabilities (0.000732 vs 0.000878). Causes:
  - pandas' default CSV float parser is 1 ulp off;
  - the online SQL used other types in a few expressions;
  - tree thresholds sit exactly on repeated training values, so one ulp changes the branch.
- **The feature query is 63% of the time and the model 24%** (profile: 41 ms total). Of the
  16.9 ms query, 13.1 ms is one subquery: the merchant's fraud reports in 90 days. Labels do
  not carry the merchant, so it reads the merchant's whole history. That subquery is the first
  thing to precompute.
- **Throughput comes from processes, not threads (decision 026).** One process with 8 requests
  in flight serves 31 requests/s, about 1.5× one at a time. It reaches ~100 requests/s with 8
  processes. The database is not the limit: the feature query alone does 333 queries/s at 8
  connections.
- **Production decision rates are rising.** The review rate went from 0.37% in February to 0.65%
  in June 2026, and the busiest day from 28 to 41 reviews of 60. If capacity starts to bind, the
  API (first come, first served) and the batch job (best reviews of the day) will diverge
  (decision 025). Phase 12 checks the cause with labels.

## Problems found during the phase

- The tolerance-based parity test hid a real training/serving difference. Fixed at the root, and
  the tests now compare exactly.
- The simulated exchange rates end on 2026-06-30, so a same-day lookup refused every later
  payment, even in EUR. Now EUR needs no rate, and other currencies use the latest rate up to
  7 days old.
- A batch re-run would have duplicated 760k log rows; it now replaces them.
- `uvicorn --workers N` failed to start some workers on Windows (`WinError 10022`). The
  benchmark starts separate processes and balances requests itself.
- On this shared laptop, tail latencies moved by a factor of 3.6 between runs:
  - one run overlapped a 244 s PostgreSQL checkpoint;
  - another ran while other programs used ~2 cores.

  The benchmark now repeats the measurement and reports medians and ranges.
- The unit and integration test files had the same name, which breaks pytest collection
  (renamed).
- The API's OpenMP thread limit was justified by a claim I had not measured. Measured: no effect
  on one-row predictions; the comment now says so.

## Risks

- **Ingestion lag.** Velocity features need earlier payments in `core.transactions`
  immediately. A lag would make the online counts lower than in training, and the parity test,
  which replays stored data, cannot see it.
- **No authentication or rate limiting.** The service is meant to sit on a private network
  behind the authorisation system.
- **Latency figures come from a shared laptop.** They need re-measuring in the Docker deployment
  (Phase 11).
- **The frozen bundle is not in git.** The API and batch job need `fraud policy optimise` to
  have been run, or `FRAUD_BUNDLE_DIR` pointing at a bundle.
