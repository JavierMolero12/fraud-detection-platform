# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 · Where does the model go wrong, and why?
#
# The frozen bundle (Phase 8: gradient boosting + calibration + expected-cost policy) on
# **validation** (Jul–Sep 2025). The test period has been used once and is not analysed here:
# studying its errors to change the model would turn it into another validation set.
#
# Output of `fraud analyse validation` (`data/processed/phase9/`).
#
# Errors are defined at the decision the business experiences:
# - **false positive**: a payment labelled legit that was **blocked**;
# - **false negative**: a payment labelled fraud that was **approved**.
#
# The simulator's ground truth (true scenario of each payment) is used only here, to explain
# the errors. It is never a model input.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from fraudplatform.config import PROJECT_ROOT
from fraudplatform.eda import plots
from fraudplatform.features.catalog import FEATURES

plots.apply_style()
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
run_dir = PROJECT_ROOT / "data" / "processed" / "phase9"
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
decisions = pd.read_parquet(run_dir / "validation_decisions.parquet")
group_of = {f.name: f.group for f in FEATURES}
decisions["outcome"].value_counts().to_frame("transactions")

# %% [markdown]
# ## 1. How much of the "error" is the label, not the model?
#
# Labels are noisy on purpose (as in reality): some fraud is never reported, and some legit
# payments are disputed as fraud. Cross-tabulating the decision outcome with that noise:

# %%
pd.crosstab(decisions["outcome"], decisions["label_noise"], margins=True)

# %% [markdown]
# ## 2. What drives the model? Global importance
#
# **Permutation importance**: drop in PR-AUC when a column (or a whole family of columns,
# shuffled together) is randomly shuffled on validation. 5 repetitions; the error bar is the
# standard deviation.

# %%
grouped = pd.read_csv(run_dir / "permutation_grouped.csv")
display(grouped.style.format({"importance": "{:.4f}", "std": "{:.4f}"}))
single = pd.read_csv(run_dir / "permutation_single.csv", index_col=0)
single["group"] = single.index.map(group_of)
top = single.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
y = np.arange(len(top))
ax.errorbar(
    top["importance"],
    y,
    xerr=top["std"],
    fmt="o",
    color=plots.BLUE,
    markersize=7,
    elinewidth=2,
    capsize=0,
)
ax.set_yticks(y, top.index)
ax.axvline(0, color=plots.MUTED, linewidth=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("drop in PR-AUC when the feature is shuffled (validation)")
ax.set_title("Top 15 features by permutation importance")
fig.tight_layout()
plots.save(fig, "permutation_importance_top15", folder="analysis")
plt.show()
single.head(15).style.format({"importance": "{:.4f}", "std": "{:.4f}"})

# %% [markdown]
# **SHAP values** (exact, additivity-checked) on a sample of 900 validation payments covering
# every kind of decision. Mean |SHAP| measures how much a feature moves individual
# predictions, in log-odds. It is a different question from permutation importance (how much
# the *ranking quality* depends on the feature), so the two orders need not agree.

# %%
shap_sample = pd.read_parquet(run_dir / "shap_sample.parquet")
shap_cols = [c for c in shap_sample.columns if c.startswith("shap__")]
mean_abs = shap_sample[shap_cols].abs().mean().sort_values(ascending=False)
mean_abs.index = mean_abs.index.str.removeprefix("shap__")
print(f"SHAP additivity error (max over the sample): {summary['shap_additivity_error']:.1e}")
comparison = pd.DataFrame({"mean_abs_shap": mean_abs.head(15)})
comparison["permutation_rank"] = [single.index.get_loc(f) + 1 for f in comparison.index]
comparison.style.format({"mean_abs_shap": "{:.3f}"})

# %% [markdown]
# ## 3. Explaining single decisions
#
# Three real validation payments: a fraud the policy blocked with high confidence, a legit
# payment it blocked (false positive), and the largest fraud it approved (false negative).
# Each bar is how much that feature pushed the log-odds up (towards fraud) or down.


# %%
def explain_row(row: pd.Series, top_n: int = 8) -> pd.DataFrame:
    contrib = row[shap_cols].astype(float)
    contrib.index = contrib.index.str.removeprefix("shap__")
    order = contrib.abs().sort_values(ascending=False).index[:top_n]
    return pd.DataFrame({"value": [row[f] for f in order], "shap_log_odds": contrib[order]})


examples = {
    "fraud blocked (most confident)": shap_sample[shap_sample["outcome"] == "fraud blocked"]
    .sort_values("p")
    .iloc[-1],
    "legit blocked (false positive)": shap_sample[shap_sample["outcome"] == "legit blocked (FP)"]
    .sort_values("p")
    .iloc[-1],
    "fraud approved (largest false negative)": shap_sample[
        shap_sample["outcome"] == "fraud approved (FN)"
    ]
    .sort_values("amount_eur")
    .iloc[-1],
}
fig, axes = plt.subplots(len(examples), 1, figsize=(7.5, 9.5))
for ax, (title, row) in zip(axes, examples.items(), strict=True):
    table = explain_row(row)
    print(
        f"{title}: amount €{row['amount_eur']:.2f}, p = {row['p']:.4f}, "
        f"base {row['base']:.2f} + contributions = log-odds {row['log_odds']:.2f}"
    )
    display(table.style.format({"shap_log_odds": "{:+.2f}"}))
    t = table.iloc[::-1]
    colors = [plots.ORANGE if v > 0 else plots.BLUE for v in t["shap_log_odds"]]
    ax.barh(np.arange(len(t)), t["shap_log_odds"], color=colors, height=0.6)
    ax.set_yticks(
        np.arange(len(t)),
        [
            f"{f} = {v}" if not isinstance(v, float) else f"{f} = {v:.3g}"
            for f, v in zip(t.index, t["value"], strict=True)
        ],
    )
    ax.axvline(0, color=plots.MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{title}: p = {row['p']:.3f}, €{row['amount_eur']:.0f}", fontsize=10)
axes[-1].set_xlabel("contribution to log-odds (orange: towards fraud, blue: towards legit)")
fig.tight_layout()
plots.save(fig, "shap_examples", folder="analysis")
plt.show()

# %% [markdown]
# **Were these really errors?** The simulator knows the true story of each payment:

# %%
truth = decisions.set_index("transaction_id")
pd.DataFrame(
    {
        title: {
            "label": "fraud" if truth.loc[row["transaction_id"], "is_fraud"] else "legit",
            "truth": truth.loc[row["transaction_id"], "scenario"],
            "label noise": truth.loc[row["transaction_id"], "label_noise"],
        }
        for title, row in examples.items()
    }
).T

# %% [markdown]
# ## 3b. How much of each error type is label noise?

# %%
fp = decisions[decisions["outcome"] == "legit blocked (FP)"]
fn = decisions[decisions["outcome"] == "fraud approved (FN)"]
fp_by_band = fp.groupby("amount_band")["label_noise"].apply(
    lambda s: (s == "unreported fraud").mean()
)
display(
    pd.DataFrame(
        {
            "false positives": fp["amount_band"].value_counts(),
            "of which unreported fraud": fp_by_band,
        }
    )
    .style.format({"of which unreported fraud": "{:.0%}"})
    .set_caption("Blocked 'legit' payments that were actually fraud, by amount")
)
fn_truth = fn["scenario"].where(fn["is_fraud_true"], "friendly fraud (truly legit)")
display(
    fn_truth.value_counts()
    .to_frame("false negatives")
    .style.set_caption("Approved 'fraud' payments by their true story")
)
truly_legit = ~decisions["is_fraud_true"]
real_fp = (decisions["outcome"] == "legit blocked (FP)") & (decisions["label_noise"] == "none")
print(f"measured false decline rate: {len(fp) / (~decisions['is_fraud']).sum():.4%}")
print(
    f"real false decline rate:     {real_fp.sum() / truly_legit.sum():.4%} "
    f"({real_fp.sum()} truly legit payments blocked)"
)

# %% [markdown]
# ## 4. False negatives: which fraud gets through?
#
# What happened to each TRUE scenario (simulator truth), split by whether it was labelled.

# %%
scenarios = pd.read_csv(run_dir / "scenarios.csv")
scenarios.style.format(
    {
        "blocked": "{:.1%}",
        "reviewed": "{:.1%}",
        "approved": "{:.1%}",
        "median_p": "{:.4f}",
        "median_amount": "€{:,.0f}",
    }
)

# %% [markdown]
# ## 5. Where are errors over-represented?
#
# `fp_rate` = share of legit payments in that group that were blocked; `fn_rate` = share of
# fraud that was approved. Over-representation 2.0 = twice as common among the errors as in
# the population.

# %%
for name in [
    "merchant_category",
    "payment_method",
    "border",
    "amount_band",
    "device_type",
    "hour_band",
    "account_age_band",
    "customer_segment",
    "home_country",
]:
    table = pd.read_csv(run_dir / f"profile_{name}.csv", index_col=0)
    display(
        table.style.format(
            {
                "legit": "{:,.0f}",
                "false_positives": "{:,.0f}",
                "fraud": "{:,.0f}",
                "false_negatives": "{:,.0f}",
                "fp_rate": "{:.3%}",
                "fn_rate": "{:.1%}",
                "fp_over_representation": "{:.2f}",
                "fn_over_representation": "{:.2f}",
            }
        ).set_caption(name)
    )

# %% [markdown]
# ## 6. Segment analysis
#
# PR-AUC is only computed with at least 30 frauds in the segment.

# %%
for name in [
    "channel",
    "payment_method",
    "merchant_category",
    "customer_segment",
    "home_country",
    "account_age_band",
    "customer_age_band",
    "amount_band",
    "border",
]:
    table = pd.read_csv(run_dir / f"segment_{name}.csv", index_col=0)
    display(
        table.style.format(
            {
                "transactions": "{:,}",
                "frauds": "{:,}",
                "fraud_rate": "{:.2%}",
                "pr_auc": "{:.3f}",
                "fraud_stopped": "{:.1%}",
                "fraud_eur_stopped": "{:.1%}",
                "block_precision": "{:.1%}",
                "false_decline_rate": "{:.3%}",
            },
            na_rep="–",
        ).set_caption(name)
    )

# %% [markdown]
# ## 7. Findings
#
# Full write-up: [`reports/analysis/phase9_findings.md`](../reports/analysis/phase9_findings.md).
#
# - **Most measured errors are label noise.** 327 of 455 blocked "legit" payments are
#   unreported fraud (all 198 blocked micro-charges are card testing); the real false decline
#   rate is 0.036%, not 0.128%. 153 of 429 approved "fraud" payments are legit disputes.
# - **Real misses are mostly stealth fraud** (221), designed to look like the customer, plus
#   55 others (card testing and cloning).
# - **What drives the model:** device age, cross-border, merchant category, device sharing and
#   novelty. The operating system, artificially strong in the data, is not in the top 15.
# - **Weak segments:** card-present (PR-AUC 0.667 vs 0.825 online), contactless (0.544),
#   domestic (0.602), and €5–20 payments, where approving is the cost-optimal choice.
# - **Real false positives** are mostly cross-border purchases of €100–500 by travellers.
