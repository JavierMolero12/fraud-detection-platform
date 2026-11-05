# PR 04 — Exploratory data analysis and statistics

Branch: `feature/eda` → `main` (stacked on `feature/data-quality-sql`)

## Objective

Understand the train data before designing features: base rate, label delay, distributions,
outliers, which attributes relate to fraud and how strongly, and which attributes overlap.
Write the findings down, including the ones that did not match expectations.

## Changes

- `configs/splits.yaml` + `load_splits()`: the temporal splits from the design, validated as
  contiguous. First consumer: the EDA, which only reads the **train** period (decision 015).
- `src/fraudplatform/eda/`
  - `stats.py`: Wilson intervals, rate tables with lift, bias-corrected Cramér's V, chi-square
    association table with Benjamini-Hochberg adjustment, univariate AUC (Mann-Whitney),
    Tukey fences, Bayes precision from prevalence/recall/FPR.
  - `data.py`: train frame via `COPY ... TO STDOUT` (1.34M rows in ~11 s).
  - `plots.py`: matplotlib style and charts with a validated palette (CVD-checked).
- `sql/eda/transactions_frame.sql`: transaction-time attributes only (ages at `ts`, local
  hour of the payment country); no historical aggregates.
- `notebooks/02_eda.py` (jupytext source, linted) → `notebooks/02_eda.ipynb` (executed,
  outputs committed so GitHub renders them). 18 figures in `reports/figures/eda/`.
- `reports/eda/eda_findings.md`: written from the run's outputs.
- New dependencies: `scipy`, `matplotlib` (core); `analysis` extra for notebooks.

## Validation

| Check | Result |
|---|---|
| Unit tests (incl. 10 new for the statistics helpers, checked against hand-computed values) | 34 passed |
| Integration tests | 26 passed (unchanged) |
| Notebook executes top to bottom | yes, 0 errors, ~55 s |
| `ruff check` / `ruff format --check` | clean (`.ipynb` excluded: generated) |

## Main findings (train period, observed labels)

- Base rate **0.848%** (CI 0.833–0.864%). At 90% recall and 1% FPR, precision would be 43.5%.
- Only **42.9%** of fraud labels are known two weeks after the transaction; 94.4% after 60 days.
- Amount: skewness 59.8 → 0.10 on log10; U-shaped fraud rate (1.04% in the lowest decile,
  0.11% in the second, 3.0% in the top); log-scale outliers are 5.0% fraud.
- Cross-border is the strongest attribute (11.4% vs 0.34%). **The apparent country effect is a
  confounder:** domestic fraud is 0.14–0.48% in every country.
- Night hours (02–05 local) reach 5.7–9.4% fraud; weekday shows no signal.
- Interactions: < €5 payments are 2.86% fraud in e-commerce vs ≤ 0.04% card present.

## Findings that did not go as expected / limitations

- **Account age barely matters** on its own (AUC 0.507); hypothesis H11 (cold start) is not
  supported by static attributes.
- **Some strong signals are artificial**: operating system (linux 26.9% fraud) and night hours
  are strong because of simulator parameters. Documented in the findings so later feature
  importances are not over-interpreted.
- **Known data artefact** (decision 014): 93 transactions on the first/last second of the
  period due to timestamp clipping in the generator. Not fixed (negligible, would change
  numbers already published in PRs 02 and 03).
- Univariate AUC is misleading for non-monotonic effects (amount 0.703, local hour 0.399):
  noted in the report so it is not used as a feature-selection criterion on its own.
