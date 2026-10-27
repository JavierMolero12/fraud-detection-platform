# Tech-lead review

A review of the whole project as a tech lead would do it before real traffic: what is solid,
what is not, and what to do next, in order. Written in Phase 14, after the reproduction from
scratch (Phase 13).

**Verdict: 4.0 out of 5.** The method is production-grade:
- leakage is controlled and measured;
- validation follows time;
- decisions come from cost;
- the test set was used once;
- the whole pipeline reproduces from an empty machine.

The operations are not production-grade: no authentication, no real-time ingestion, no
defined fallback for payments it cannot score. Before any real traffic, four things matter
most:
- the freshness of the payment history;
- a protocol to refit before deploying;
- the fallback for unscored payments;
- authentication.

The time budget the review also found missing is fixed in this PR.

## 1. Scores

1 = missing, 3 = works with known gaps, 5 = what a careful team would ship.

| Area | Score | Why |
|---|---|---|
| Problem framing and design | 5 | Eleven hypotheses stated before the data, costs and capacity as explicit assumptions, 30 decisions with alternatives ([design](phase1_design.md), [decisions](decisions.md)) |
| Simulated data | 4 | Seven behavioural scenarios, label delay and noise, injected defects, a drift; but one seed, one drift, and labels without a dispute lifecycle (finding 11) |
| Data quality and SQL | 5 | 47 rules with explicit actions, recall measured against every injected defect (100%), and a simulator bug found by the rules ([report](../reports/data_quality/dq_report.md)) |
| Features and leakage control | 5 | Point-in-time SQL, a time-travel test (0 of 50 change), and leakage measured, not described: 0.852 offline vs 0.401 served |
| Modelling and validation | 4 | Temporal CV with a gap, paired day-block bootstraps, honest "within noise" verdicts; but nondeterministic Phase 7 frames until Phase 13, rounded selected parameters, and no refit before deployment (findings 3, 8, 9) |
| Decision policy and calibration | 4 | Per-payment expected cost, capacity, sensitivity to every assumption, one test evaluation recorded by commit; the reviewer-recall assumption, the one that matters, is unverified |
| Error analysis and interpretability | 4 | Found that most measured errors are label noise (real false decline rate 0.036%, not 0.128%); caught SHAP's tree explainer being wrong for this model; no reason codes for reviewers yet |
| Serving | 3 | Bit-for-bit online/offline parity, idempotency, a capacity lock that holds under concurrency; but no time budget (fixed here), history freshness, an undefined fallback, and idempotency that ignores the payload (findings 1, 2, 5, 6) |
| Packaging and CI | 4 | Two pinned images, the model mounted read-only, CI that runs the pipeline in containers and checks the API against batch; images not scanned, signed or published |
| Monitoring | 4 | Every run sees only what was known that day; rules fixed before the replay; an honest negative result for PSI; but no scheduler, alert routing or runbook |
| Reproducibility and documentation | 4 | A reproduction from scratch matches row for row and bit for bit; it took a manual run to find four problems that CI could have caught (finding 9) |
| Code quality and tests | 4 | 194 tests, 74% coverage (92–100% on the logic), typed configuration, ruff; two implementations of every feature to keep in sync (finding 10) |
| Security and operations | 2 | No authentication, rate limiting or TLS; development credentials; no metrics endpoint, tracing or runbooks |

## 2. What is strong

- **Leakage handled as an engineering problem.**
  - Point-in-time SQL with an explicit `reported_at` rule.
  - A test that deletes the future and rebuilds.
  - A deliberately leaky model kept as a control.
- **Results reported against the hypothesis, not for it.** H5, H9 and H11 are written up as not
  supported, and the test PR-AUC drop (0.792 → 0.752) is reported with its interval.
- **The test set was used once**, after the policy was frozen in git (`32ce00c`, then `3b95790`).
  The command refuses to run again.
- **Exact parity instead of "close enough".** The 1e-9 tolerance hid a real disagreement (1 in
  150 probabilities); the fix was a shared parser, and the test now asks for equality.
- **Monitoring that respects time.** Delay-adjusted calibration caught the drift after 67 days
  with no false alarm; the conventional PSI threshold never did, and the report says so.
- **The reproduction from scratch was done, not claimed**, and the four problems it found are
  fixed with tests ([reproducibility](reproducibility.md)).

## 3. Findings, by severity

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | High | **No time budget in `/predict`.** A slow feature query held the authorisation for as long as it took (one native request took 910 ms). | **Fixed in this PR:** `statement_timeout` of 500 ms (`serving.yaml`); 503 "scoring exceeded its time budget" |
| 2 | High | **Payment history freshness.** Velocity features assume every earlier payment is already in `core.transactions`. With any ingestion delay, the 5-minute and 1-hour counts are lower online than in training, and card-testing bursts are missed. Documented in Phase 10; the parity test cannot see it. | Open: write each authorised payment to the history in the scoring transaction (or a streaming feature store), and test with a simulated lag |
| 3 | High | **The deployed model is six months older than its data.** It was trained on data up to 2025-07-01 and deployed for 2026-01-01; it never saw July–December 2025. Model ageing is one of the two candidate causes of the test drop. | Open: decide the protocol: evaluate the procedure once, refit on all mature labels, deploy behind a shadow period |
| 4 | High | **No authentication, rate limiting or TLS** on the API. | Open: mTLS or a service token, a rate limit, a private network |
| 5 | Medium | **Payments it cannot score have no defined fallback.** An unknown customer, card, merchant or stale exchange rate gets 422; the authorisation system "applies its fallback rule", which is not specified, tested or measured. 422 also reads as a client error for what is an expected condition. | Open: define the fallback (rules, or approve under a limit), measure how often it fires, answer 200 with a `not_scored` decision |
| 6 | Medium | **Idempotency ignores the payload.** A retry with the same `transaction_id` and a different amount gets the old decision; a test codifies this. | Open: store a hash of the request; 409 on a mismatch |
| 7 | Medium | **`model_version` identifies a file, not a model.** skops names arrays after object ids, so the same model saved twice gets another version (found by the reproduction). | Open: version by a hash of the model's content (tree arrays, parameters, columns) |
| 8 | Medium | **The selected parameters are stored rounded** (0.021 and 0.68), so the bundle is not the search's best model: 0.792 vs 0.794 on validation. Noise, but a silent difference. | Documented (decision 020 note). Open: store the exact values |
| 9 | Medium | **Nondeterminism was found by hand.** Phase 7 frames changed order on every load from Phase 7 to Phase 13; only the reproduction from scratch noticed. | Fixed (`ORDER BY`, with a test). Open: a CI job that runs a training step twice on the small dataset and compares the outputs |
| 10 | Medium | **Two implementations of every feature**: window functions offline, lookups online (decision 017). The exact parity test guards them, but every feature change costs double. | Open: generate both from one definition, or a feature store |
| 11 | Low | **Labels have a single state.** A real chargeback is raised, disputed and resolved; `core.fraud_labels` keeps one row with the final answer. | Open for real data: label history with validity dates, read as of a date |
| 12 | Low | **Monitoring runs on demand.** No scheduler, no alert routing, no runbook. The rolling-reference rule is a hypothesis from one drift scenario. | Open: schedule, route alerts, test the rule on a second scenario |
| 13 | Low | **Batch and online allocate reviews differently.** Batch ranks a whole day with hindsight; the API serves in arrival order. They agree while capacity does not bind (busiest day: 41 of 60). | Documented (decision 025) |
| 14 | Low | **Supply chain.** Images are not scanned or signed, there is no SBOM, and dependency updates are manual. | Open |

## 4. What to do next, in order

1. **Before any real traffic:**
   - history freshness (2);
   - authentication and rate limiting (4);
   - a defined, measured fallback (5);
   - the refit-and-deploy protocol (3). Each comes with a test that fails today.
2. **Before the first model change:**
   - content-based model versions (7);
   - exact selected parameters (8);
   - a determinism job in CI (9).
3. **Before scaling the team:**
   - one definition per feature (10);
   - a monitoring schedule with alert routing and runbooks (12);
   - idempotency on the payload (6);
   - image scanning (14).
4. **Before trusting the business case:**
   - verify with a fraud team how much fraud a review stops, the one assumption that moves the
     policy (+7.3% regret if it is half);
   - repeat the evaluation on a second simulated scenario, and on Sparkov.
