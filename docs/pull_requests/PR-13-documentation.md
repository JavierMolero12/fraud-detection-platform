# PR 13 — Documentation, reproduction from scratch, and the fixes it found

Branch: `feature/documentation` → `main` (stacked on `feature/monitoring`)

## Objective

The Phase 1 exit criterion: someone outside the project can reproduce it from scratch. Every
document a reader needs is in English and up to date. The whole pipeline was then run in a clean
clone, in Docker, on an empty database, and compared with the committed results.

## Changes

- **Reproduction:**
  - [`docs/reproducibility.md`](../reproducibility.md): requirements, commands, the time of each
    step, how to check the result, what is not re-run (the test evaluation), and what the
    reproduction found.
- **Translations:**
  - [`docs/phase1_design.md`](../phase1_design.md) and [`docs/decisions.md`](../decisions.md)
    are now in English (they were in Spanish). The decision index covers all 30 decisions.
  - Decision 016 gets a dated correction: the warm-up is 92 days in the config, not 90.
- **New documents:**
  - [`docs/architecture.md`](../architecture.md): the system as built, with diagrams and the
    deviations from the Phase 1 design.
  - [`reports/final_report.md`](../../reports/final_report.md): the whole project in one
    report, with every hypothesis and its verdict.
- **README:** rewritten.
  - Results first, then a 5-minute check with Docker only, the full reproduction and the API.
  - A map of code and results, and the history of the phases.
  - CI and license badges.
- **Model card:** marked final.
- **License:** MIT (`LICENSE`, `pyproject.toml`).
- **GitHub:** repository description and topics. The repository stays private.
- **Fixes found by the reproduction** (section 6 of the reproduction guide):
  - Lineage in containers: the images record the commit they were built from.
  - Row order of the training frames: `ORDER BY` in `load_features`.
  - `fraud models tune --n-jobs`.
- **Phase 7 re-run** with the fixed row order. Its findings, CV tables and notebook now show
  the new numbers, and the original ones are kept in a note.
- **Tests:**
  - Every relative link in the Markdown documents resolves.
  - Lineage from git and from the image.
  - Row order of the frames, with and without the leaky columns.

## Validation

| Check | Result |
|---|---|
| Unit tests (3 new: every relative Markdown link resolves; lineage from git and from the image) | 134 passed, 1 skipped on Windows |
| Integration tests (3 new: frame row order with and without leaky columns, on a heap made unordered on purpose) | 58 passed |
| The row-order test without the fix | fails, as it should |
| Reproduction from scratch: clean clone, Docker, empty database | 77 minutes of computing; results in the table below |
| Phase 7 re-run in the development environment with the fix | 36 minutes; the same configuration selected |
| `ruff check`, `ruff format --check` | clean |

## Findings

The reproduction reproduced the project. Compared with the committed reports and the
development environment:

| What | Result |
|---|---|
| Dataset, `core` and feature tables | identical (same hash; MD5 over every row) |
| Data quality, Phase 6 | identical, apart from run dates and fit times |
| Phase 7, after the fix | identical to 4 decimals; the same winner |
| The model in the bundle | 400 trees identical bit for bit; 1,446 of 359,220 probabilities 1–2 ulps apart (Linux vs Windows `exp()`) |
| Batch scoring, Jan–Jun 2026 | identical decisions, risk levels and quantiles on 760,511 payments |
| Monitoring | identical alerts and HTML report |
| API | 50 of 50 decisions equal to batch; the README payment gets exactly the README's probability |

- **Phase 7 numbers moved by up to 0.003** with a fixed row order. H7 temporal CV is 0.799 (was
  0.797). H8 is 0.852 offline and 0.401 served (was 0.847 and 0.399). Tuned vs default on
  validation is +0.004 (CI −0.001 to +0.010; was +0.002). No conclusion changed. The original
  numbers are kept in section 5 of the Phase 7 findings.
- **The shipped model is not bit for bit the search's best.** `configs/model.yaml` stores the
  learning rate and L2 rounded (0.021, 0.68). The bundle scores 0.792 on validation against 0.794
  for the exact values: noise, but now documented (decision 020 note).
- **A bundle cannot be identified by its hash.** skops names arrays after Python object ids, so
  the same model saved twice gives a different `model_version`. The committed batch summary is
  the check instead.

## Problems found during the phase

- **The feature build stalled for 13 hours** in the reproduction. Two PostgreSQL instances were
  running in the 8 GB Docker VM with 1.1 GB free on the host, and the laptop slept overnight.
  Alone, the build takes 10 minutes. The guide now asks for one database at a time.
- **Containers recorded no lineage** (`git_commit = unknown`, `git_dirty = True`). Fixed: the
  images record the commit they were built from.
- **`models tune` was OOM-killed** in the Docker VM, by 4 parallel logistic regression fits.
  Fixed: `--n-jobs`.
- **Phase 7 was not reproducible:** a parallel join returned rows in a new order on every load.
  Fixed with `ORDER BY`. The frames of every other phase were already in that order (same
  digest before and after).
- **The README rewrite dropped a phrase a test checks** ("50 point-in-time features"). Restored.
- **Docker stopped answering** after the stall. It was restarted (`wsl --shutdown`), and the
  data survived.
