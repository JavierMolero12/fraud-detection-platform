# PR 14 — Tech-lead review

Branch: `feature/tech-lead-review` → `main` (stacked on `feature/documentation`)

## Objective

The Phase 1 exit criterion for the last phase: "scores and questions". Review the whole project
as a tech lead would before real traffic. Score each area with its evidence, rank the findings,
and fix what is small and safe. The interview questions were prepared in a separate document,
outside the repository.

## Changes

- **[`docs/tech_lead_review.md`](../tech_lead_review.md):**
  - scores for 13 areas, each with its evidence;
  - what is strong;
  - 14 findings ranked by severity, each marked fixed or open;
  - what to do next, in order.
- **A time budget for the API** (finding 1):
  - every connection of the API pool sets `statement_timeout`: `serving.yaml`, 500 ms;
  - a cancelled statement answers 503 "scoring exceeded its time budget", so the authorisation
    system applies its fallback instead of waiting;
  - batch scoring does not use this pool.
- **README:** links to the review and this PR.

## Validation

| Check | Result |
|---|---|
| Unit tests | 134 passed, 1 skipped on Windows |
| Integration tests (1 new: a 1 ms budget gives 503, not a wait) | 59 passed |
| `ruff check`, `ruff format --check` | clean |

## Findings

**Verdict: 4.0 out of 5.** The method is production-grade, the operations are not.

| Severity | Open findings (the full list is in the review) |
|---|---|
| High | Payment history freshness: velocity features assume the history is current. |
| High | The deployed model is six months older than its data; there is no refit-before-deploy protocol. |
| High | No authentication, rate limiting or TLS. |
| Medium | Unscored payments (unknown entities) have no defined, measured fallback. |
| Medium | Idempotency ignores the payload; a test codifies it. |
| Medium | `model_version` identifies a file, not a model. |
| Medium | The selected parameters are stored rounded. |
| Medium | Determinism is not checked in CI. |
| Medium | Every feature has two implementations. |

## Problems found during the phase

- **`/predict` had no timeout.** The native latency measurement had already recorded a
  910 ms request. Fixed in this PR.
- **The budget was chosen from warm-cache measurements.** A demo after a database restart got a
  503 on its first payment. Measured afterwards: the first lookup of an entity took 465–965 ms
  for 3 of 12 payments, and 23–32 ms once cached. The budget works as designed; how often it
  fires with a cold cache, and what to do about it, is in finding 1.
- **An existing test codifies a questionable contract:** a retry with a different amount
  silently gets the old decision. It is left as it is and listed as finding 6, because
  changing it needs a schema change (a request hash).
