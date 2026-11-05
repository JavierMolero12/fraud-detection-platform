# Reproducing the project from scratch

This guide rebuilds every result from an empty machine: the dataset, the database, the
features, the models, the frozen bundle, the batch scores and the monitoring replay. It also
records what happened when it was followed for real, in a clean clone with an empty database
(October 2026). That run found four problems, all fixed before this guide was written. They are
listed in [section 6](#6-what-the-reproduction-found).

For a five-minute check that everything wires together, see the README's
[quick check](../README.md#quick-check-about-5-minutes-docker-only). It uses the `tiny` dataset
and a stand-in model. The numbers in the reports need the full dataset, which is what this
guide covers.

## 1. What you need

| | Measured in the reproduction |
|---|---|
| Docker | Docker Desktop 4.49 on Windows 10, WSL 2 backend. Any recent Docker Engine with Compose v2 should do |
| Memory for Docker | **8 GB** (the VM had 8.1 GB and 20 CPUs, on a 16 GB laptop). Run nothing else in it |
| Disk | about 9 GB: images 2.1 GB (`app` 683 MB, `train` 944 MB, `postgres:16.4` 432 MB), database volume 5.8 GB (3.7 GB of data, the rest write-ahead log), raw CSVs 378 MB |
| Time | **77 minutes** of computing, plus the first image build (2 minutes without cache) |
| Python on the host | not needed |

**One PostgreSQL at a time.** The feature build sorts 2.85M rows several times with 1 GB of
shared buffers and 64 MB per sort. With a second database running in the same 8 GB VM, it
stalled for 13 hours ([section 6](#6-what-the-reproduction-found)). Alone, it takes 10
minutes.

## 2. The commands

```bash
git clone https://github.com/JavierMolero12/fraud-detection-platform.git
cd fraud-detection-platform
export GIT_COMMIT=$(git rev-parse HEAD)    # baked into the images: lineage of every run
docker compose --profile jobs build        # both images
docker compose up -d --wait db

docker compose run --rm jobs simulate --scale full
docker compose run --rm jobs db init
docker compose run --rm jobs db load data/raw/v1.0.1-full
docker compose run --rm jobs dq run
docker compose run --rm jobs features build
docker compose run --rm jobs models compare
docker compose run --rm jobs models tune --n-jobs 1
docker compose run --rm jobs policy optimise
docker compose run --rm jobs analyse validation
docker compose run --rm jobs serve batch
docker compose run --rm jobs monitor replay
docker compose run --rm jobs monitor report
```

- **Linux:** prefix the `run` commands with `FRAUD_UID=$(id -u) FRAUD_GID=$(id -g)`, so the
  files the containers write are yours.
- **Git Bash on Windows:** `export MSYS_NO_PATHCONV=1` first, or Git Bash rewrites the
  container paths.
- **Next to an existing stack:** `COMPOSE_PROJECT_NAME`, `DB_CONTAINER_NAME`, `POSTGRES_PORT`
  and `API_PORT` give an isolated one. The reproduction used `fraud-repro`, port 5435.
- **`--n-jobs 1`:** the logistic regression grid fits in parallel and each worker copies the
  training frame. Four workers (the default, as on the 16 GB development machine) were
  OOM-killed in the 8 GB VM. The results do not depend on it.

Then serve the bundle you just trained: `docker compose up -d api`, and
`curl http://localhost:8000/health`.

## 3. Each step

Times are from the reproduction, Docker on a Windows laptop.

| Step | Time | What it does | Where the output goes |
|---|---|---|---|
| `simulate --scale full` | 1.5 min | 2,866,727 payments, seed 42, simulator 1.0.1 | `data/raw/v1.0.1-full/` |
| `db init` | 4 s | schemas `raw`, `dq`, `core`, `features`, `scoring`, `monitoring`, `sim` | database |
| `db load` | 2 min | the CSVs into `raw`, the simulator's truth into `sim` | database |
| `dq run` | 6 min | 47 rules; quarantine; `raw` → `core` | [`reports/data_quality/dq_report.md`](../reports/data_quality/dq_report.md) |
| `features build` | 10 min | 50 point-in-time features for 2,852,459 payments, plus the leaky set for Phase 7 | database |
| `models compare` | 5.5 min | two baselines and three model families (seven rows) on validation (Phase 6) | [`reports/models/phase6_comparison.md`](../reports/models/phase6_comparison.md) |
| `models tune --n-jobs 1` | 32 min | temporal CV, H7, tuning, H8 (Phase 7) | `data/processed/phase7/` |
| `policy optimise` | 1 min | calibration and the cost policy on validation; **the frozen bundle** (Phase 8) | `data/processed/phase8/` |
| `analyse validation` | 15 min | error analysis and SHAP on validation (Phase 9) | `data/processed/phase9/` |
| `serve batch` | 1.5 min | the bundle scores Jan–Jun 2026 into `scoring.predictions` (Phase 10) | [`reports/serving/`](../reports/serving) |
| `monitor replay` | 1.5 min | 42 weekly monitoring runs over production (Phase 12) | [`reports/monitoring/`](../reports/monitoring) |
| `monitor report` | 45 s | the HTML report | [`monitoring_report.html`](../reports/monitoring/monitoring_report.html) |

The notebooks (`notebooks/*.py`) read these outputs. To re-execute them you need the `train`
extra on the host (`pip install -e ".[train,dev]"`), as in section 7.

## 4. What is not re-run: the test evaluation

The test period (Oct–Dec 2025) was evaluated **once**, at the end of Phase 8
([decision 022](decisions.md)). Its record,
[`reports/policy/test_evaluation.json`](../reports/policy/test_evaluation.json), is committed,
and `fraud policy evaluate-test` refuses to run while it exists. `--force` exists, but running
it again would turn the test set into one more validation set. The reproduction skipped it.

What can be checked instead is that the bundle you trained behaves like the one that was
evaluated. Section 5 says how.

## 5. Checking your result

After the last step, `git status` lists the committed reports your run rewrote, and
`git diff` shows how. These differences are expected:

- **Run dates and durations** in the data quality report, the Phase 6 table and the batch summary.
- **The code line of the Phase 6 table:** the commit you built from instead of the original.
- **The model and policy versions** in the batch summary (see below).
- **PNG figures:** the same data, rendered by a different matplotlib build.
- **On another operating system:** the last digit of a few sums of probabilities, such as
  `mean_predicted` in `reports/monitoring/weekly_truth.csv`.

Everything else should match.

**The bundle cannot be checked by its hash.** `model_version` and `policy_version` are the
first 12 hex digits of the files' SHA-256. skops names the arrays inside `model.skops` after
Python object ids, so the same model saved twice gives different bytes. `policy.json` records the
commit it was trained from. What identifies the bundle is what it decides:
[`reports/serving/batch_20260101_20260701_summary.json`](../reports/serving/batch_20260101_20260701_summary.json)
holds its decisions, risk levels and probability quantiles on the 760,511 payments of January
to June 2026. If your run leaves that file unchanged apart from the two versions and the
timing, your bundle decides like the evaluated one.

`docker compose run --rm jobs serve verify-bundle` checks something else: that the environment
you serve from reproduces the scores saved with the bundle.

### Result of the reproduction

A clean clone of `feature/documentation` at `bc8c60f`, Docker on Windows (Linux containers), an
empty database. Compared with the committed reports and with the development environment
(Windows, native Python), where the results were first produced:

| What | Compared with | Result |
|---|---|---|
| Dataset | the simulation report | the same hash, `350d74bd…`; 2,866,727 raw payments |
| `core.transactions`, `core.fraud_labels`, both feature tables | the development database, MD5 over every row | identical: 2,852,459 payments, 26,004 labels |
| Data quality report | committed | identical, apart from the run line |
| Phase 6, run twice | committed | every metric identical, both times |
| Phase 7 | development re-run with the fix | H7, H8 and every validation metric identical to 4 decimals; the same winner. 3 of 16 candidates differ by 0.0002–0.0009 in CV |
| The model | development bundle | all 400 trees identical bit for bit; 1,446 of 359,220 validation probabilities 1–2 ulps apart (the operating system's `exp()`, as in Phase 11) |
| The policy | development `policy.json` | identical, apart from the lineage and the calibration intercept (18 ulps) |
| Phase 9 | development outputs | identical, SHAP values included, apart from probabilities up to 4 ulps apart |
| Batch scoring, Jan–Jun 2026 | committed summary | identical decisions (749,932 approved, 3,831 reviewed, 6,748 blocked), risk levels, reviews per day and quantiles |
| Monitoring | committed reports | alert timeline, post-hoc rules, summary and HTML report identical; `weekly_truth.csv` differs in the last digit of a mean in 3 of 26 weeks |
| API | `serve check`, and the README request | 50 of 50 decisions equal to batch, largest difference 0.0; the README payment gets exactly the README's probability |

The test evaluation was not re-run (section 4).

## 6. What the reproduction found

Four problems, in the order they appeared. All four were fixed before this guide was written,
and the commands above already include the fixes.

**1. The feature build stalled for 13 hours.** `20_windows.sql` ran for 12 h 55 min; the step
then failed when Docker stopped answering.
- **Cause:** memory. The reproduction's PostgreSQL ran next to the development one in the same
  8 GB Docker VM, and the laptop had 1.1 GB of its 16 GB free. The laptop also went to sleep
  that night, so how much of the 13 hours each cause took is not known.
- **Fix:** with the development database stopped and Docker restarted, the same query took
  132 s and the whole build 10 minutes. Section 1 now asks for one PostgreSQL at a time.

**2. Runs in containers had no lineage.** MLflow tagged them `git_commit = unknown` and
`git_dirty = True`. There is no git in the image, and `str(bool("unknown"))` is `"True"`.
- **Fix** (`bb91ceb`): the images record the commit they were built from (build argument
  `GIT_COMMIT`; CI passes `github.sha`). Whether that tree had uncommitted changes cannot be
  known inside the image, so it is recorded as `unknown`.

**3. `models tune` was killed for lack of memory** after 290 s. The VM's kernel log shows a
global OOM kill of the `fraud` process.
- **Cause:** the logistic regression grid runs 4 fits in parallel, and each worker holds its own
  copy of the training frame. That fits in 16 GB natively, not in an 8 GB VM that also runs
  PostgreSQL.
- **Fix** (`bc8c60f`): `fraud models tune --n-jobs`. The default stays at 4; the reproduction
  used 1.

**4. Phase 7 did not give the same numbers twice.** Before it was killed, the reproduction
logged H7 as 0.7960 / 0.8144 / 0.7881 (temporal CV / shuffled CV / validation). The committed
values were 0.7970 / 0.8155 / 0.7904.
- **Cause:** `load_features` had no `ORDER BY`.
  - Without the leaky columns, PostgreSQL scans the time index, and rows already came in the
    order the table is written in, `(ts, transaction_id)`. The digest of that order was the
    same in both databases.
  - With them (Phase 7 only), it runs a parallel hash join, and rows came back in a new order
    on every load: 8 loads, 8 different orders, on either database.
  - Row order changes the model. Gradient boosting places its bin edges from a 200,000-row
    subsample, and shuffled K-fold assigns rows to folds by position.
- **Fix** (`bc8c60f`): `ORDER BY f.ts, f.transaction_id`.
  - The frames of every other phase are unchanged: same digest before and after.
  - An integration test moves half the rows to the end of the table and checks the order; it
    fails without the fix.
- **Consequence:** Phase 7 was re-run in the development environment with the fix, and its
  report now shows the new numbers. The original ones are kept in a note
  ([Phase 7 findings](../reports/models/phase7_findings.md)). The conclusions do not change.
  - H7 out-of-time validation is now 0.7901, the same model and rows as Phase 6's 0.7901.
  - Row order alone had moved these numbers by up to 0.003. That is as large as the tuning gain
    on validation (+0.002), which was already called noise.
  - The model in the bundle is not affected: Phase 8 trains from `configs/model.yaml` on frames
    that were already in a fixed order.

## 7. Without Docker

The development itself ran natively on Windows: Python 3.13, the `fraud` CLI installed in a
virtual environment, PostgreSQL in Docker.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -c docker/training-env-constraints.txt -e ".[train,dev]"
docker compose up -d db                 # or your own PostgreSQL 16 and DATABASE_URL
fraud simulate --scale full             # DATABASE_URL defaults to localhost:5433 (compose)
# ... the same commands as in section 2, without "docker compose run --rm jobs"
```

[`docker/training-env-constraints.txt`](../docker/training-env-constraints.txt) is the
`pip freeze` of the environment the model was trained in. Without `-c`, pip installs the newest
compatible libraries, and a different scikit-learn or numpy can move the numbers.
