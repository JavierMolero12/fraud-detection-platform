# Phase 11 findings: Docker images, tests and CI

What was built:
- Two images from one [`Dockerfile`](../../Dockerfile) (decision 027).
- A compose stack: `db`, `api`, and a `jobs` service for any `fraud` command.
- A CI that also builds the images and tests them end to end (decision 028).

Sources: [`cross_platform_scores.json`](cross_platform_scores.json)
(`docker/compare_environments.py`), [`latency_docker.json`](latency_docker.json)
(`fraud serve bench --url`), `fraud serve verify-bundle`, and `coverage` on the test suite.
Machine: the same Windows laptop as Phase 10. The containers run in Docker Desktop's Linux VM
(20 CPUs, 8 GB); the other projects' containers were stopped for the measurements.

## 1. The images

| Image | Size | Contents |
|---|---|---|
| `app` | 682 MB | API and `fraud serve` commands; base dependencies only (42 packages) |
| `train` | 944 MB | every `fraud` command; adds MLflow, SHAP, matplotlib (101 packages) |

- **Dependencies are locked inside the base image** (`python:3.13-slim`, pinned by digest),
  constrained to the training environment's versions: scikit-learn 1.9.1, numpy 2.5.4,
  pandas 3.0.6, scipy 1.18.1, skops 0.16.0, pyarrow 25.0.1 (`docker/lock.sh`).
- **What makes `app` big is the scientific stack**, not the project: pyarrow 156 MB, scipy
  139 MB, pandas 72 MB, numpy 70 MB, scikit-learn 48 MB. pyarrow could go and save 156 MB, but it
  decides how pandas stores the category strings in the training environment. It stays until a
  retraining can change both sides together.
- **The API container** runs as a non-root user with a read-only root filesystem and a
  healthcheck. It is healthy 6 s after start. The model bundle is mounted read-only, never
  copied into the image.
- **Before**, the API imported MLflow indirectly (`prepare()` lived in the module that tracks
  experiments). It moved to `models/pipelines.py`; the serving code now loads only pandas,
  pyarrow, scipy, scikit-learn and skops.
- **Build time**, both images: 2 minutes without cache on this machine.

## 2. Does the container serve the same model? Almost bit for bit, and the same decisions

`fraud serve verify-bundle` re-scores the validation period (359,220 payments) in the
container and compares with the scores Phase 8 saved on Windows:

| Compared row by row, Windows venv vs Linux `app` image | Rows differing | Max difference |
|---|---|---|
| Tree output (log-odds, before the sigmoid) | 0 | 0 |
| Probability | 1,446 (0.40%) | 1.1e-16 (2 ulps) |
| Calibrated probability | 1,423 | 1.1e-16 (2 ulps) |
| Decision (approve / review / block) | 0 | – |

- **Same library versions, still not bit for bit.** The trees add the same leaves in the same
  order, but the sigmoid calls the operating system's `exp()` (MSVC on Windows, glibc on Linux),
  and they round differently in the last bit.
- **No decision changes.** The live check agrees: the Linux API container against the batch
  scores computed on Windows, 6,000 production payments, max difference 4.3e-19, 6,000
  identical decisions.
- **`verify-bundle` reports the exact result** (`identical: false`, 1,446 differing, 2 ulps).
  It passes up to 4 ulps; anything larger means another model or other inputs. A test checks
  both sides: one ulp passes, a 1e-9 change fails.

## 3. `uvicorn --workers` added ~40 ms to every request

The first Docker benchmark showed the server deciding in 42 ms while the client waited 88 ms.
Narrowed down step by step:

| From the `jobs` container | `GET /health` | `GET /model` | `POST /predict` |
|---|---|---|---|
| 4 workers, `uvicorn --workers` | 44.0 ms | 44.0 ms | 87.8 ms |
| 1 worker | 1.6 ms | 1.3 ms | 45.6 ms |
| 4 workers, `fraud serve api` (fixed) | 1.2 ms | 0.9 ms | 43.1 ms |

- **First hypothesis, discarded.** Turning off Nagle's algorithm on the client
  (`TCP_NODELAY`) changed nothing (88.2 vs 88.1 ms).
- **Cause.** With several workers, uvicorn creates the listening socket as
  `socket.socket(family)`, with protocol 0. asyncio only sets `TCP_NODELAY` on connections whose
  socket says `IPPROTO_TCP`, so every worker connection kept Nagle's algorithm. The response
  body then waited for the client's delayed ACK, about 40 ms on Linux. With one worker, asyncio
  creates the socket itself, with the right protocol.
- **Fix.** `fraud serve api` (`serving/server.py`) runs uvicorn with the same socket declared
  as TCP; the image uses it.
- **Guards.** A unit test checks the socket's protocol (on Linux). A CI step fails if a request
  that does no work takes over 20 ms in the container.
- **Phase 10 did not see this** because on Windows the benchmark ran separate single-worker
  processes.

## 4. Latency in Docker

`fraud serve bench --url http://api:8000` from the `jobs` container (same Docker network). Real
Phase 8 bundle, 4 workers, 3 repetitions of 1,000 sequential + 1,000 concurrent requests
(8 client threads). Median with the range in brackets:

| | Docker, 4 workers | Phase 10 host, Windows |
|---|---|---|
| Client p50, one at a time | 44.6 ms [44.4–47.3] | 48.6 ms [46.8–50.6] |
| Client p95, one at a time | 59.0 ms [57.0–110.6] | 154.2 ms [65.5–235.2] |
| Server p50, one at a time | 40.5 ms | 43.8 ms |
| Requests/s, 8 client threads | 74.1 [60.9–80.4] | 92.3 [66.5–99.7] (4 processes) |

- **One at a time, the container is as fast as the host.**
- **Concurrent throughput is lower**: the client, the four workers and PostgreSQL share the
  same VM.
- Before the fix, the same measurement gave 88.2 ms and 67.9 requests/s.

## 5. CI and tests

The first job is lint, then unit and integration tests (PostgreSQL service), then coverage.
The second job runs only if the first passes:
- builds the `app` and `train` images, with layer caching;
- runs the whole data pipeline in containers on the tiny dataset (simulate, load, data
  quality, features) and trains the demo bundle;
- starts the API container and runs `fraud serve check`: 50 stored payments, the same
  probability and decision as batch scoring with the same bundle, or the job fails;
- runs the Nagle guard.

Locally, with an isolated copy of the stack (`fraud-ci` project, port 5434):
- each pipeline step took 2–3 s;
- the check passed: 50 payments, difference 0.0, 50 identical decisions, served model version
  equal to the local one.

**Tests:** 117 unit (1 runs on Linux only) and 53 integration, all passing.
**Coverage** (unit + integration): **73%** of 3,339 statements.
- 92–100% for the logic: simulation, features, costs, calibration, metrics, serving.
- 0–50% for job orchestration that only real runs exercise: the CLI, the Phase 6–9 jobs,
  plots, the benchmark.

No minimum threshold is enforced yet.

## 6. Problems found during the phase

- **Bit-for-bit across operating systems is not achievable** for the sigmoid (section 2). The
  first version of `verify-bundle` required it and failed on correct behaviour.
- **The uvicorn multi-worker Nagle delay** (section 3).
- **Docker Desktop stopped responding twice** (backend waiting for WSL). Fixed by shutting
  down WSL and restarting Docker Desktop. The data survived.
- **Git Bash on Windows** rewrites `/repo`-style paths in `docker run -v`: `docker/lock.sh`
  disables that conversion.
- **`docker compose run` has no `--no-build` flag** (only `up` does); the CI script used it.
- **The socket test fails on Windows**: Python there reads the protocol from the descriptor. It
  runs on Linux, where the API runs with several workers.

## 7. Not done, and risks

- **The CI has not run yet** at the time of writing; the PR records its result.
- **Images are not published to a registry** and are not scanned for vulnerabilities.
- **The CI checks the serving machinery with a demo bundle on tiny data**, not the real model.
  The demo bundle approved all 50 payments locally, so the decision comparison mostly exercises
  approvals. The real model is checked locally with `verify-bundle`.
- **Upgrading any locked library means retraining**, regenerating the locks and re-running
  `verify-bundle`.
