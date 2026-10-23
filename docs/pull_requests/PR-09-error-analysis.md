# PR 09 — Error analysis, interpretability and model card

Branch: `feature/error-analysis` → `main` (stacked on `feature/cost-threshold-calibration`)

## Objective

Understand where and why the deployed model + policy go wrong, what drives its decisions, how
it performs by segment, and write the first model card. On validation only: the test period
has been used once.

## Changes

- `analysis/explain.py`: permutation importance (single and grouped by feature family) and
  SHAP values with `PermutationExplainer`, with an additivity check.
- `analysis/errors.py`: errors at the decision level, label-noise attribution with the
  simulator truth, over-representation profiles, segment metrics, per-scenario outcomes.
- `analysis/run.py` + `fraud analyse validation [--quick]`: the job, saving each result as
  soon as it is computed; `--quick` smoke-tests the whole job in ~10 s.
- `notebooks/07_error_analysis`, `reports/analysis/phase9_findings.md` + tables,
  **`reports/model_card.md`** (draft), decision 023.
- **Correction across the docs**: the feature catalog has **50** features, not 51 as written
  since Phase 5; a test now ties the README to the catalog.

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. 7 new: SHAP additivity on a categorical-split model, attribution sanity, grouped importance, outcomes and label noise, profiles, segments, doc-count regression) | 92 passed |
| Integration tests | 37 passed |
| SHAP additivity on 900 validation payments | max error 4.1e-14 |
| Full job | 993 s |

## Findings

- **Most measured errors are label noise.** 327 of 455 blocked "legit" payments are
  unreported fraud (all 198 blocked micro-charges are card testing); real false decline rate
  0.036% vs 0.128% measured. 153 of 429 approved "fraud" payments are legit disputes.
- **Real misses are stealth fraud** (221 of 276): 2.8% of stealth fraud is stopped vs 91–100%
  of every other scenario.
- **Drivers:** device age, cross-border, merchant category, device sharing and novelty.
  22 of 49 inputs have no measurable importance; the artificially strong operating system is
  rank 44.
- **Weak segments:** card present (PR-AUC 0.667), contactless (0.544), domestic (0.602).
- Real false positives are mostly cross-border purchases of €100–500 (travellers).

## Problems found during the phase

- `shap.TreeExplainer` returns silently wrong (non-additive) values for HistGradientBoosting
  with native categorical splits. Switched to `PermutationExplainer` with an additivity check
  (decision 023).
- `add_bands` overwrote the numeric model input `cross_border` with text; the first run
  failed. Fixed with a separate display column.
- A duplicated column made the first full run fail at the very end, losing 16 minutes of
  compute. Results are now saved step by step, and `--quick` smoke-tests the job first.
- My toy test for "correlated features hide each other" did not reproduce the effect with
  gradient boosting; rewritten. The effect does appear on the real model (card velocity).
- Two per-scenario percentages were copied wrong into the findings; corrected after
  re-computing them.
- The feature count (50, not 51) was wrong in four documents since Phase 5.

## Risks

- The simulator truth that explains the errors does not exist in production: there, a sample
  of blocked payments must be investigated to estimate the real false positive rate.
- Segment results for small countries rest on few frauds.
