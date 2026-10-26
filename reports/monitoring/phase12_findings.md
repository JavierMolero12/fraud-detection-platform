# Phase 12 findings: monitoring, drift and delayed performance

**What was replayed:** 42 weekly monitoring runs over simulated production (payments
2026-01-01 → 2026-07-01, runs 2026-01-08 → 2026-10-22). Each run used only the payments and the
fraud labels that existed on its date.

**Drift the simulator injected on 2026-03-01** (unknown to the monitoring):
- legitimate behaviour moves online and to wallets;
- a new wallet fraud pattern ramps up.

**Rules:** fixed in [`configs/monitoring.yaml`](../../configs/monitoring.yaml) before the replay
(decision 029).

**Sources:**
- [HTML report](monitoring_report.html), [`alert_timeline.csv`](alert_timeline.csv),
  [`posthoc_rules.csv`](posthoc_rules.csv), [`weekly_truth.csv`](weekly_truth.csv),
  [`summary.json`](summary.json);
- the `monitoring` schema (`fraud monitor replay`, `fraud monitor report`).

## 1. What really happened (known only afterwards)

From the simulator's truth and every label known today, per payment week:

| Payments in | New-pattern frauds | Recall on reported frauds (weekly mean) |
|---|---|---|
| Jan–Feb | 0 | 0.857 |
| Mar | 195 | |
| Apr | 741 | |
| May–Jun | 2,080 | 0.778 |

- **The model is not blind to the new pattern:** it reviews or blocks 76% of it.
- **But it underestimates fraud.** From mid-March, observed fraud runs above the mean predicted
  probability (e.g. 1.51% observed vs 1.10% predicted in the week of 9 April).
- **Overall recall slides** from about 0.86 to about 0.78.
- **PR-AUC is a poor sentinel here:** a higher base rate pushes it up while recall falls.

## 2. Alert timeline with the rules fixed in advance

Reference = the validation period (Jul–Sep 2025): 0.36% of payments reviewed, 0.70% blocked,
recall 0.836.

| Rule | Runs before the drift | First alert after the drift |
|---|---|---|
| Input PSI ≥ 0.25 | 8 (all `device_age_days`) | none from the drift itself |
| Review rate ×1.25 (warning) | 2 | +18 days |
| Block rate ×1.25 (warning) | 2 | +32 days |
| Block rate ×1.5 (alert) | 1 | +46 days |
| Review rate ×1.5 (alert) | 0 | +60 days |
| **Labels: observed / expected reports ≥ 1.25, p < 0.01** | **0** | **+67 days (7 May)** |
| **Retraining recommended** (2 runs in a row) | 0 | **+74 days (14 May)** |
| Recall of the newest mature cohort, −5 points | 0 | +123 days (2 July) |

- **The label rules are the reliable ones**: silent for all 8 runs before the drift, then
  persistent.
- **Delay-adjusted calibration** sat at 0.98–1.10 before March. It climbed from April (1.10 on
  9 April, 1.29 on 7 May) to a peak of 1.77.
- **It answers two months earlier than mature-cohort recall**, because it does not have to wait
  for labels to mature: it compares what has been reported so far with what the model's own
  probabilities predicted for that delay.
- **The decision-rate rules reacted earlier, but they also alerted in January**, when there
  really was more fraud (a post-holiday peak, 0.93% vs 0.69% in February). A rate alert says
  "something changed", not "the model got worse".
- **Input PSI with the conventional thresholds failed.** `device_age_days` alerted every week
  from the first run (PSI 1.32–1.66): a device can only get older, so against a fixed reference
  it always drifts. No other input ever reached 0.10. The largest after the drift was 0.090, a
  label-derived merchant feature at the end of production.

## 3. The drift was visible in the inputs, below the threshold

- **The step.** `payment_method`, `is_card_present`, `device_type`, `operating_system` and the
  device-history inputs had a weekly PSI of 0.000–0.003 in January and February. They jumped
  to 0.06–0.075 in the first full week of March and stayed there. The score PSI went from
  ≤ 0.0012 to about 0.02.
- **The noise.** Weekly PSI inside the reference period itself (each week against the whole
  period) never exceeded 0.009 for any input except `device_age_days`.
- **So the 0.10 convention is about 10× the noise** of a weekly window on this system. It is
  too coarse to see a real change in the payment mix.

## 4. Post-hoc alternatives (hypotheses, not results)

Designed after seeing the replay. Each was checked on weeks with no injected drift that were
never used to design it: the first half of 2025, in the training period.

| Rule | Production runs alerting before the drift | Drift-free weeks alerting | Days to first alert after the drift |
|---|---|---|---|
| Input PSI vs reference ≥ 0.10 (fixed in advance) | 0 | 0 of 26 | never |
| Input PSI vs reference ≥ 0.05 | 0 | 3 of 26 | 11 |
| **Input PSI vs the previous 4 weeks ≥ 0.03** | **0** | **0 of 22** | **11** |
| Labels: observed > expected, p < 0.01, no ratio threshold | 0 | n/a | 39 |

All PSI rules exclude `device_age_days`, whose weekly PSI already exceeds 0.05 inside the
reference.

- **Fixed reference at 0.05:** catches the step but false-alarms on inputs that accumulate
  history. `card_prior_fraud_reports` reaches 0.07 and `account_age_days` 0.049 in drift-free
  weeks, because the simulated world gets older.
- **Rolling reference** (each week against the four before it) separates a step from slow
  ageing. The largest value in drift-free weeks was 0.013, against 0.049 in the drift week.
- **For the next version of the rules:** a fixed reference for slow drift, a rolling reference
  for steps, and the "drifts by construction" inputs left out of the fixed view. To validate it
  needs another drift scenario (another seed, another date). The rules in the repository stay as
  they were fixed, so the result above is not rewritten.

## 5. H9: "PSI detects the drift before the labels show the performance drop"

**Not supported as stated.**
- **With the rules fixed in advance**, input PSI never caught the drift. The labels did, after
  67 days.
- **With the post-hoc rolling rule**, inputs would have alerted after 11 days, 8 weeks before
  the labels. But what they detect is the shift in *legitimate* behaviour, not the new fraud.
  The new fraud is under 1% of payments and leaves almost no mark on input distributions.
- **What showed that the model was getting worse** was the label-based calibration: more fraud
  reported than the model expected.

## 6. Problems found during the phase

- **`device_age_days` (and, more slowly, history-accumulating inputs) drift by construction.**
  The model card already warned about `merchant_new_for_customer`; age-like inputs need their
  own treatment.
- **The guard test for the simulator's truth found two readers I had not listed:** the Phase 3
  data quality evaluation, which is legitimate (it measures rule recall against the injected
  defects). It is now on the explicit allow-list.
- **The first report draft** had odd axis ticks, a label over the March peak, a duplicated
  legend entry and a summary sentence that attributed a June PSI to "the step in March". Fixed
  after looking at the rendered page.
- **Dark mode** is defined in the report's CSS but could not be checked visually: the browser
  preview renders local files in light mode only.
- **The same test file name in `unit/` and `integration/` broke collection again**, as in
  Phase 10. Renamed, and a test now fails on any duplicate test file name.

## 7. Not done, and risks

- **No retraining was run.** The rule says *when* (14 May); whether a model retrained on
  data up to then recovers recall is the next experiment.
- **The replay monitors batch scores.** The API logs the same inputs (`scoring.predictions.features`),
  but no scheduler runs `fraud monitor run` every week.
- **One injected drift, one seed.** The post-hoc rules need a second scenario before they
  replace the ones in the configuration.
