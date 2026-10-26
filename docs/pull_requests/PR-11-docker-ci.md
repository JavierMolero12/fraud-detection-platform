# PR 11 — Docker images, tests and CI end to end

Branch: `feature/docker-ci` → `main` (stacked on `feature/api-and-batch`)

## Objective

Package the platform in Docker images that serve the frozen model exactly as it was trained,
run the whole stack with compose, and make the CI build those images and test them end to end.

## Changes

- **`Dockerfile`** with two targets (decision 027):
  - `app`: the API and the `fraud serve` commands. Non-root, read-only filesystem, healthcheck.
  - `train`: every `fraud` command.

  The Python base image is pinned by digest. Dependencies come from lock files generated inside
  it, constrained to the training environment's versions (`docker/lock.sh`,
  `docker/training-env-constraints.txt`). `.dockerignore` is an allow-list.
- **`docker-compose.yml`**:
  - `db`, plus `api` with the bundle mounted read-only and the workers configurable;
  - `jobs` (profile `jobs`) with `data/`, `reports/`, `mlruns/` mounted.

  A second isolated stack can run next to the development one.
- **Dependencies**: `pyproject.toml` split into base plus a `train` extra (MLflow, SHAP,
  matplotlib). `skops` is declared (it was only a transitive dependency of MLflow). `prepare()`
  moved to `models/pipelines.py`, so serving no longer imports MLflow.
- **New commands**:
  - `fraud serve demo-bundle`: a stand-in bundle on whatever data is loaded;
  - `fraud serve check`: a running API must give the batch probabilities and decisions;
  - `fraud serve verify-bundle`: this environment must reproduce the bundle's saved
    validation scores;
  - `fraud serve bench --url`: measures an already running API.
- **`fraud serve api`** now goes through `serving/server.py`, which fixes the missing
  `TCP_NODELAY` of `uvicorn --workers`.
- **CI**:
  - a concurrency group;
  - the `train` extra installed;
  - coverage in the job summary and as an artifact;
  - a new `docker` job: builds both images (layer cache), runs the pipeline in containers on the
    tiny dataset, starts the API container, runs `fraud serve check` and a 20 ms guard against
    the Nagle delay.
- **Other**: `.github/pull_request_template.md`, `docker/compare_environments.py`.
- **Docs**: [`reports/deployment/phase11_findings.md`](../../reports/deployment/phase11_findings.md),
  decisions 027–028, README (Docker section, install with `.[train,dev]`).

## Validation

| Check | Result |
|---|---|
| Unit tests (1 new: listening socket declared TCP, Linux only) | 116 passed, 1 skipped on Windows; 25/25 serving tests on Linux |
| Integration tests (3 new: `check_api` passes and cleans up, fails on another model; `verify_bundle` tolerates 1 ulp, rejects 1e-9) | 53 passed |
| Coverage, unit + integration | 73% of 3,339 statements |
| Image build, both targets, no cache | 2 min; `app` 682 MB, `train` 944 MB |
| Pipeline in containers, tiny dataset (isolated stack) | 6 steps, 2–3 s each; API healthy in 6 s |
| `fraud serve check` against the API container | 50 payments, difference 0.0, 50 identical decisions |
| `verify-bundle`, real Phase 8 model, Linux `app` image | tree outputs identical; 1,446 of 359,220 probabilities 1–2 ulps away; passed |
| Linux API vs Windows batch, 6,000 production payments | max difference 4.3e-19, 6,000 identical decisions |
| Docker latency, 4 workers, one at a time | client p50 44.6 ms (Phase 10 host: 48.6 ms) |
| Docker throughput, 8 client threads | 74.1 requests/s |
| `ruff check`, `ruff format --check` | clean |

## Findings

- **Same library versions are not enough for bit-for-bit results across operating systems.**
  - The tree outputs are identical.
  - 0.4% of probabilities differ by 1–2 ulps, from the platform's `exp()` in the sigmoid.
  - No decision changes.

  `verify-bundle` reports the exact result and passes up to 4 ulps.
- **`uvicorn --workers` added ~40 ms to every request on Linux.**
  - Cause: its listening socket has protocol 0, so asyncio never sets `TCP_NODELAY`, and
    responses wait for the client's delayed ACK.
  - Measured: `GET /model` took 44 ms with 4 workers vs 1.3 ms with 1. Fixed: 0.9 ms; a full
    prediction 44.6 ms instead of 88.2 ms.
- **Container latency equals the host one at a time; throughput is lower**: 74 vs 92
  requests/s, because client, workers and database share Docker Desktop's VM.
- **The images are big because of the scientific stack** (pyarrow 156 MB, scipy 139 MB), not
  the project. pyarrow stays: it defines how pandas stores categories in training.

## Problems found during the phase

- The first `verify-bundle` demanded bit-for-bit equality and failed on correct behaviour.
- The Nagle delay above. The first hypothesis (client side) was tested and discarded.
- Docker Desktop hung twice waiting for WSL; restarting WSL fixed it, with no data loss.
- Git Bash rewrote `docker -v` paths. `docker compose run` has no `--no-build` flag. The socket
  test cannot run on Windows (Python reads the protocol from the descriptor there).

## Risks

- **The CI tests the serving machinery with a demo bundle on tiny data**, which approved all 50
  payments locally. The real model is only checked locally (`verify-bundle`).
- **Images are not published or scanned for vulnerabilities.**
- **Upgrading a locked library requires retraining**, new locks and `verify-bundle`.
- **Latency figures still come from a laptop** (Docker Desktop VM shared with the client).
