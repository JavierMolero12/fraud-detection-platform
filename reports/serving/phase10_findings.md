# Phase 10 findings: real-time API and batch scoring

What was built: `POST /predict` (FastAPI) scores one payment at authorisation time, and
`fraud serve batch` scores a whole period. Both use the frozen Phase 8 bundle unchanged: model
version `fcf88c67457e`, policy version `65a2111fa8c2` (the first 12 hex digits of their SHA-256,
the same hashes committed in [`frozen_bundle.sha256`](../policy/frozen_bundle.sha256)). Sources:
[`latency.json`](latency.json) (`fraud serve bench`), [`latency_breakdown.json`](latency_breakdown.json)
(`fraud serve profile`), [`batch_20260101_20260701_summary.json`](batch_20260101_20260701_summary.json)
(`fraud serve batch`), and the tests in `tests/integration/test_feature_parity.py` and
`tests/integration/test_api_and_batch.py`.

Machine: a Windows laptop (20 logical cores) running PostgreSQL in Docker Desktop, plus other
programs (a browser, a local Kubernetes cluster). Latencies are indicative; section 3 shows how
much they moved between repetitions.

## 1. Online features equal offline features, bit for bit

The API computes the 50 features of one new payment with lookups
([`single_transaction.sql`](../../sql/features/online/single_transaction.sql)); training and
batch use the window-function build (decision 017).

- **The first parity test passed and was not enough.** It compared values with a 1e-9 relative
  tolerance. Then the API and the batch job gave a different probability for 1 of 150
  transactions (0.000732 vs 0.000878 with the test bundle).
- **Two last-digit causes.** pandas' default CSV float parser, which built every training and
  batch frame, is not correctly rounded (`0.002707006369426752` is read 1 ulp low). And the
  online SQL computed some values with other types (`ln()` on float8 instead of numeric;
  untyped parameters that PostgreSQL treats as numeric instead of float8).
- **Why one ulp matters.** Tree split thresholds sit exactly on repeated training values (a
  merchant's smoothed fraud rate is shared by all its transactions in a window), so a value one
  ulp higher takes the other branch.
- **Fix (decision 024).** The API reads its features through the same function as training
  and batch (COPY → CSV → pandas), and the online SQL casts every parameter and intermediate to
  the offline type. The parity test now requires exact equality of the model input.

| Check | Result |
|---|---|
| Parity on the full database: 1,539 production payments × 50 features | 76,950 values, **0 differences** |
| Parity in CI (small dataset, stratified by payment method + fraud) | 499 payments, 0 differences |
| API vs batch, benchmark replay of production payments | 15,000 payments: max probability difference **0.0**, 15,000 identical decisions |

## 2. Where the time goes (one payment at a time)

`fraud serve profile` instruments the real code path (500 production payments, no HTTP):

| Step | p50 ms | p95 ms | Share of mean |
|---|---|---|---|
| Feature SQL (COPY + parse) | 25.80 | 33.21 | 63.4% |
| Model + calibration | 9.95 | 11.61 | 24.0% |
| Write the log row | 2.03 | 3.14 | 5.1% |
| Prepare the frame | 1.36 | 1.71 | 3.3% |
| Decide (cost policy, capacity) | 0.86 | 1.33 | 2.2% |
| Idempotency check | 0.66 | 1.18 | 1.7% |
| Exchange rate | 0.01 | 0.56 | 0.3% |
| **Total** | **41.12** | **49.49** | |

- **The feature query dominates, and one subquery dominates it.** Timed separately on 200
  production payments, the merchant's fraud reports in the last 90 days take 13.1 ms (median)
  of the 16.9 ms query. The card's fraud history takes 3.1 ms; every other part is below 1 ms.
  Labels do not carry the merchant, so to find which of the ~3,100 reports in the window belong
  to the merchant, the query reads the merchant's entire transaction history (one example:
  3,648 rows on 3,508 different pages). The cost grows with that history.
  In production this would be a counter per merchant, updated when a chargeback arrives. The
  offline build already uses a denormalised table for it, but that is a snapshot taken at build
  time, and the API must see reports received after it.
- **The model costs ~10 ms** for one row: 400 trees evaluated one call at a time.
- **On this laptop the absolute numbers depend on the CPU power state.** A first version of this
  measurement gave 55 ms in total while the CPU was otherwise idle, and 46 ms with another core
  kept busy. The loop mostly waits on the database, so the clock drops between steps. The shares
  are stable; use the benchmark for end-to-end latency.

## 3. Latency and throughput of the real API

`fraud serve bench`: uvicorn processes started for the run, production-period payments replayed
over HTTP as new requests, every row deleted afterwards. 3 repetitions; each has 1,000 sequential
requests plus 1,000 concurrent requests (8 client threads) per process count. Median over the
repetitions, with the range in brackets:

| One payment at a time (1 process) | p50 | p95 | p99 |
|---|---|---|---|
| Client (HTTP round trip) | 48.6 ms [46.8–50.6] | 154.2 ms [65.5–235.2] | 266.3 ms [83.7–1,075.5] |
| Server (request → decision) | 43.8 ms [42.5–44.9] | 80.2 ms [60.2–84.9] | 173.1 ms [74.7–766.5] |

| 8 client threads | 1 process | 2 processes | 4 processes | 8 processes |
|---|---|---|---|---|
| Requests/s | 30.9 [28.0–31.5] | 57.7 [51.9–60.0] | 92.3 [66.5–99.7] | 99.7 [90.6–108.7] |
| Server p50 | 230.7 ms | 117.4 ms | 66.8 ms | 60.7 ms |
| Server p95 | 278.8 ms | 149.0 ms | 101.9 ms | 83.4 ms |

- **The median is stable; the tail is not.** p50 moved by 4 ms between repetitions, p95 by a
  factor of 3.6. A single run is not a fair number to publish on a shared laptop (section 6).
- **Threads barely help; processes do (decision 026).** With 8 requests in flight, one process
  serves 31 requests/s, only ~1.5× one at a time. Requests wait for the GIL inside the
  process, so the server time rises from 44 to 231 ms. Throughput roughly doubles with 2
  processes and triples with 4.
- **It levels off at ~100 requests/s, and it is not the database.** The feature query alone
  sustains 333 queries/s at 8 concurrent connections (468 at 16). The likely limits are the
  benchmark client (8 threads in one Python process) and the CPU shared with the other programs.
  I did not isolate which.
- On Windows, `uvicorn --workers N` failed to start some workers (`WinError 10022` on the
  shared socket). The benchmark starts independent processes on consecutive ports and spreads
  requests over them, like a load balancer.

## 4. Batch scoring of the production period

`fraud serve batch` scored **760,511** payments (2026-01-01 → 2026-07-01) from the offline
features, without reading any label:

- **Speed:** 47.3 s on the first run (16,070 per second). A re-run took 68.9 s because it first
  deletes the earlier scores of the same model and period (no duplicates: 760,511 rows, 760,511
  distinct ids).
- **Decisions:** 749,932 approve, 3,831 review, 6,748 block. Risk levels: 745,410 low, 7,980
  medium, 7,121 high. Reviews per day: mean 21.2, max 41 of 60.
- **Output:** `transaction_id, fraud_probability, risk_level, decision, model_version,
  scoring_timestamp` (CSV, `data/processed/scoring/`, not in git), plus `scoring.predictions`.

**Decision rates are rising.** No labels are used here; Phase 12 checks the cause.

| Month | Payments | Block % | Review % | Max reviews in a day |
|---|---|---|---|---|
| Test period (Oct–Dec 2025) | 387,978 | 0.71 | 0.37 | 30 |
| 2026-01 | 128,286 | 0.91 | 0.42 | 28 |
| 2026-02 | 115,842 | 0.69 | 0.37 | 27 |
| 2026-03 | 128,544 | 0.73 | 0.44 | 28 |
| 2026-04 | 126,667 | 1.04 | 0.50 | 40 |
| 2026-05 | 132,724 | 1.02 | 0.62 | 41 |
| 2026-06 | 128,448 | 0.92 | 0.65 | 41 |

The review rate has risen every month since February, and the busiest day went from 28 to 41
reviews. If the trend continues, the daily capacity of 60 will start to bind. That is the case
where the API (first come, first served) and the batch job (best reviews of the day) stop
deciding the same (decision 025). It coincides with the drift the simulator starts in March
2026: payments moving online and to wallets, plus a new wallet fraud pattern that ramps up.
Without labels, that cause is not established here.

## 5. Online decisions

- **Same policy as batch.** Calibrated probability, then the action with the lowest expected
  cost for that amount. The README example shows why the risk level is only for display: an
  €8,598 online payment at 0.20% ("low") is sent to review.
- **Daily review capacity online.** Reviews go first come, first served: a payment that would be
  reviewed when no slot is left gets its best alternative. Count and insert run under a
  PostgreSQL advisory lock per UTC day, taken only by payments whose best action is a review.
  Test: 10 simultaneous review candidates with 3 slots → exactly 3 reviews, 7 blocks.
- **Idempotency.** A retried `transaction_id` returns the decision already logged, so it is not
  scored twice or counted twice against the capacity.
- **Input contract (422 otherwise).** Time-zone-aware timestamp, not in the future. Positive
  amount. ISO currency and country codes. A card-present payment has no device; a wallet payment
  has one. No extra fields. Unknown customer, card of that customer, merchant or country, or an
  exchange rate more than 7 days old, also gives 422: the payment cannot be scored, and the
  authorisation system applies its fallback rule.
- **Logging.** Every API decision goes to `scoring.predictions` with model and policy versions,
  the time to decision and the 50 model inputs (JSON), ready for drift monitoring in Phase 12.

## 6. Problems found during the phase

- **Parity within 1e-9 was not parity** (section 1). Found because a test compared API and
  batch probabilities, not just features.
- **The simulated exchange-rate table ends on 2026-06-30.** A same-day lookup refused every
  later payment, even in EUR. Now EUR needs no rate, and other currencies use the latest rate
  up to 7 days old.
- **A retried batch run would have duplicated 760k rows** in the log. It now replaces them.
- **The OpenMP thread limit was set in the start-up thread**, and I assumed it covered the
  request threads. Measuring showed it made no difference for one-row predictions (8.8 vs
  8.9 ms). Kept for multi-process deployments, with an accurate comment.
- **Benchmark noise.** One run overlapped a PostgreSQL checkpoint that spent 244 s writing the
  pages left by the batch re-run (p99 627 ms, max 4.3 s). Another, with no checkpoint, still
  had a p99 of 732 ms; right after it the browser and other programs were using about 2 cores.
  Hence the repetitions and ranges in section 3.
- **Two test files had the same name** (`tests/unit` and `tests/integration`), which breaks
  pytest's collection. Renamed.

## 7. Not done, and risks

- **No explanations in `/predict`** (decision 026). The logged inputs allow explaining any
  decision later.
- **No authentication or rate limiting.** The service is meant to sit behind the authorisation
  system on a private network; it is not hardened for exposure.
- **Ingestion lag would create training/serving skew.** Velocity features assume earlier
  payments are already in `core.transactions` when the next one arrives. With an ingestion delay,
  the 5-minute and 1-hour counts would be lower online than offline, so card-testing bursts would
  be missed. The parity test cannot see this, because it replays stored data. In production each
  authorised payment must reach the history immediately (written by the same service, or a
  streaming feature store).
- **The review capacity counts API reviews only.** Batch scores are for monitoring and reporting,
  not for sending reviews.
