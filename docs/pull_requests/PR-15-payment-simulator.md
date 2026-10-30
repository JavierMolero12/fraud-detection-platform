# PR 15 — Payment simulator

Branch: `feature/payment-simulator` → `main` (stacked on `feature/tech-lead-review`)

## Objective

Show the system working on one payment at a time, without reading code or using a terminal. The
simulator started as a local demo tool; this PR moves it into the package, with a command,
tests and documentation.

## Changes

- **`fraud serve simulator`** ([`serving/simulator.py`](../../src/fraudplatform/serving/simulator.py),
  [`simulator.html`](../../src/fraudplatform/serving/simulator.html)): a page on
  http://127.0.0.1:8001 in front of the running API.
  - At start it picks twelve batch-scored payments of the production period, one per condition:
    legitimate, each fraud scenario, a missed fraud, a false positive. A condition that matches
    nothing is skipped.
  - Each send goes to `POST /predict` under a new id `ui-…`. The page shows the request, the
    answer, 14 of the logged features, the expected cost of each action and, on request, the
    generator's truth.
  - The amount can be changed before sending.
  - The prediction the API logs is deleted right after the answer, so the period's predictions and
    the day's review count stay as they were.
  - The costs and risk levels come from the API's `GET /model`, so the page explains the bundle
    the API actually serves.
  - The page's text is in Spanish; the code is in English.
- **README:** a section on the simulator and a link to this PR.

## Validation

| Check | Result |
|---|---|
| Unit tests (1 new: every scenario has a text, every feature shown exists) | 135 passed, 1 skipped on Windows |
| Integration tests (6 new, real API and database) | 65 passed |
| `ruff check`, `ruff format --check` | clean |
| By hand, full dataset: each of the twelve payments sent 5 times | 60 × 200; from the 3rd send, 57–92 ms but one at 226 ms (the test suite was running); no `ui-` rows left |

The integration tests check that the page hides the truth until asked, that a send gives exactly
the batch probability and decision, that the costs are the cost model's, that nothing stays in
`scoring.predictions`, and that an unreachable API (502) or a cancelled query (503) reach the page
with a reason.

## Problems found during the phase

- **The response time on the page was about 200 ms too high.** The first version created an HTTP
  client for every send, and creating one costs about 200 ms on Windows. The page showed around
  400 ms for payments the API answered in 40–70 ms (measured with curl). One client per run now.
- **An unreachable API showed nothing.** The send failed inside the server, and the page stayed as
  it was. It now answers 502 with the API's address.
- **The guard on the simulator's truth caught the new reader.** A unit test allows only
  evaluation code to read `sim.ground_truth`. The simulator reads it to reveal the truth after the
  API's answer, so `serving/simulator.py` was added to the list with that reason. The API's own
  modules stay outside it.
- **A cold cache can give several 503s in a row.** One payment failed three times (731–913 ms)
  before it passed on the fourth send. This is finding 1 of the tech-lead review, seen from the
  user's side; the page explains the 503 and asks for another send.
