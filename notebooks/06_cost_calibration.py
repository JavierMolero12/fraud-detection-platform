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
# # 06 · Calibration, cost model and decision policy
#
# From a fraud probability to a decision: approve, review or block.
#
# - Model: the selected gradient boosting (Phase 7), trained 2024-10-01 → 2025-06-30.
# - Calibration and policy chosen on **validation** (Jul–Sep 2025) with `fraud policy optimise`.
# - Everything frozen and committed (`reports/policy/policy_v1.json`, commit `32ce00c`) before
#   the **single evaluation on test** (Oct–Dec 2025, `fraud policy evaluate-test`).
#
# Cost assumptions (`configs/costs.yaml`, all of them assumptions): chargeback fee €15,
# review €4, review stops 90% of the fraud it sees, €1 friction for reviewing a legit
# customer, a wrongful block costs 2% of the amount + €10, 60 reviews per day.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from fraudplatform.config import PROJECT_ROOT, load_cost_config
from fraudplatform.decision.costs import CostModel, evaluate_actions, threshold_actions
from fraudplatform.eda import plots

plots.apply_style()
pd.set_option("display.width", 160)
run_dir = PROJECT_ROOT / "data" / "processed" / "phase8"
policy_dir = PROJECT_ROOT / "reports" / "policy"
validation = json.loads((policy_dir / "validation_results.json").read_text(encoding="utf-8"))
test = json.loads((policy_dir / "test_evaluation.json").read_text(encoding="utf-8"))
policy = json.loads((policy_dir / "policy_v1.json").read_text(encoding="utf-8"))
cost_cfg = load_cost_config()
costs = CostModel.from_config(cost_cfg)
scores = pd.read_parquet(run_dir / "validation_scores.parquet")
print("deployed policy:", policy["policy_name"], "| calibration:", policy["calibration"])

# %% [markdown]
# ## 1. The optimal decision depends on the amount
#
# For a calibrated probability p, each action has an expected cost. Where two lines cross,
# the best action changes. A €5 payment is only worth reviewing above p ≈ 26%; a €1,000
# payment is worth reviewing above 0.55% and blocking above 19%. A single global threshold
# (and certainly 0.5) cannot be right for every amount.

# %%
amounts = np.geomspace(2, 5000, 200)
bounds = pd.DataFrame([costs.breakeven(a) for a in amounts], index=amounts)
display(pd.DataFrame(validation["breakeven_examples"]).T.style.format("{:.3f}"))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(amounts, bounds["approve_review"], color=plots.BLUE)
ax.plot(amounts, bounds["review_block"], color=plots.ORANGE)
ax.fill_between(amounts, 0, bounds["approve_review"], color=plots.BLUE, alpha=0.06, linewidth=0)
ax.fill_between(amounts, bounds["review_block"], 1, color=plots.ORANGE, alpha=0.06, linewidth=0)
ax.text(8, 0.04, "approve", color=plots.INK_SECONDARY)
ax.text(300, 0.10, "review", color=plots.INK_SECONDARY)
ax.text(8, 0.75, "block", color=plots.INK_SECONDARY)
ax.annotate(
    "approve → review",
    (amounts[-1], bounds["approve_review"].iloc[-1]),
    xytext=(-4, 6),
    textcoords="offset points",
    ha="right",
    color=plots.INK_SECONDARY,
)
ax.annotate(
    "review → block",
    (amounts[-1], bounds["review_block"].iloc[-1]),
    xytext=(-4, 6),
    textcoords="offset points",
    ha="right",
    color=plots.INK_SECONDARY,
)
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
ax.set_xlabel("transaction amount (EUR, log scale)")
ax.set_ylabel("fraud probability")
ax.set_title("Cheapest action by amount and probability (no capacity limit)")
fig.tight_layout()
plots.save(fig, "decision_regions_by_amount", folder="policy")
plt.show()

# %% [markdown]
# ## 2. Is the model calibrated?
#
# Reliability diagram on validation, each month calibrated by a calibrator fitted on the
# other two (cross-fitting). Equal-count bins; log scales, because 90% of predictions are
# below 0.1%.

# %%
display(
    pd.DataFrame(validation["calibration_heldout_month"])
    .T.style.format(
        {
            "brier": "{:.6f}",
            "log_loss": "{:.6f}",
            "ece": "{:.6f}",
            "mean_predicted": "{:.5f}",
            "observed_rate": "{:.5f}",
            "pr_auc": "{:.4f}",
        }
    )
    .set_caption("Fitted on Jul–Aug, scored on Sep 2025 (forward in time)")
)
reliability = pd.read_csv(policy_dir / "reliability.csv")
fig, ax = plt.subplots(figsize=(6.2, 5.0))
for method, color in zip(
    ["none", "platt", "isotonic"], [plots.BLUE, plots.ORANGE, "#1baf7a"], strict=True
):
    r = reliability[reliability["method"] == method]
    keep = r["observed"] > 0
    ax.plot(
        r.loc[keep, "predicted"],
        r.loc[keep, "observed"],
        "o-",
        color=color,
        markersize=6,
        label=method,
    )
lims = [2e-5, 0.3]
ax.plot(lims, lims, color=plots.MUTED, linewidth=1)
ax.annotate(
    "perfect calibration",
    (0.02, 0.02),
    xytext=(6, -12),
    textcoords="offset points",
    color=plots.INK_SECONDARY,
    fontsize=9,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("observed fraud rate (bin)")
ax.set_title("Reliability on validation (cross-fitted by month)")
ax.legend(loc="upper left")
fig.tight_layout()
plots.save(fig, "reliability_validation", folder="policy")
plt.show()

# %% [markdown]
# Equal-count bins put every prediction above ~0.4% (the region where decisions are made) in
# a single bin. Fixed bins in that region, with 95% Wilson intervals on the observed rate:

# %%
from fraudplatform.eda.stats import wilson_interval  # noqa: E402

edges = [0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
band = pd.cut(scores["p_calibrated"], edges, include_lowest=True)
table = scores.groupby(band, observed=True).agg(
    n=("is_fraud", "size"), frauds=("is_fraud", "sum"), predicted=("p_calibrated", "mean")
)
table["observed"] = table["frauds"] / table["n"]
table["ci_low"], table["ci_high"] = wilson_interval(table["frauds"], table["n"])
table.style.format(
    {
        "predicted": "{:.4f}",
        "observed": "{:.4f}",
        "ci_low": "{:.4f}",
        "ci_high": "{:.4f}",
        "n": "{:,}",
        "frauds": "{:,}",
    }
)

# %% [markdown]
# ## 3. Expected cost as a function of the thresholds (policy P1)
#
# Total realised cost on validation for every (review threshold, block threshold) pair,
# with the 60 reviews/day capacity enforced. Left: block threshold fixed at its optimum.
# Right: review threshold fixed at its optimum.

# %%
grid = pd.read_csv(policy_dir / "threshold_grid.csv")
best = grid.loc[grid["total_cost_eur"].idxmin()]
print(
    f"best thresholds: review >= {best['t_review']:.4f}, block >= {best['t_block']:.3f}, "
    f"cost €{best['total_cost_eur']:,.0f}"
)
y = scores["is_fraud"].to_numpy()
amount = scores["amount_eur"].to_numpy()
day = scores["ts"].dt.date.to_numpy()
realized = costs.realized(y, amount)
naive = threshold_actions(scores["p_calibrated"], day, 0.5, 0.5, cost_cfg.review_capacity_per_day)
naive_cost = evaluate_actions(naive, y, amount, realized, day, costs.reviewer_recall)
print(
    f"'block if p >= 0.5, never review': €{naive_cost['total_cost_eur']:,.0f} "
    f"({naive_cost['total_cost_eur'] / best['total_cost_eur'] - 1:+.0%} vs optimum)"
)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
row = grid[np.isclose(grid["t_block"], best["t_block"])].sort_values("t_review")
axes[0].plot(row["t_review"], row["total_cost_eur"] / 1000, color=plots.BLUE)
axes[0].axvline(best["t_review"], color=plots.MUTED, linewidth=1)
axes[0].set_xscale("log")
axes[0].set_xlabel(f"review threshold (block fixed at {best['t_block']:.2f})")
axes[0].set_ylabel("total cost on validation (k EUR)")
col = grid[np.isclose(grid["t_review"], best["t_review"])].sort_values("t_block")
col = col[col["t_block"] <= 1]
axes[1].plot(col["t_block"], col["total_cost_eur"] / 1000, color=plots.BLUE)
axes[1].axvline(best["t_block"], color=plots.MUTED, linewidth=1)
axes[1].axvline(0.5, color=plots.MUTED, linewidth=1)
axes[1].annotate(
    "0.5",
    (0.5, axes[1].get_ylim()[1]),
    xytext=(3, -12),
    textcoords="offset points",
    color=plots.INK_SECONDARY,
    fontsize=9,
)
axes[1].set_xlabel(f"block threshold (review fixed at {best['t_review']:.3f})")
fig.suptitle(
    "Threshold vs total cost (validation, Jul–Sep 2025)", x=0.01, ha="left", fontweight="semibold"
)
fig.tight_layout()
plots.save(fig, "threshold_vs_cost", folder="policy")
plt.show()

# %% [markdown]
# ## 4. Policies compared (validation)
#
# P1 uses one pair of thresholds for every transaction. P2 decides per transaction from the
# expected cost of each action, so it uses the amount. Both get the same review capacity.

# %%
cols = [
    "total_cost_eur",
    "cost_per_1000_txn",
    "blocked",
    "reviewed",
    "reviews_per_day_max",
    "fraud_blocked",
    "fraud_reviewed",
    "fraud_approved",
    "legit_blocked",
    "false_decline_rate",
    "fraud_eur_stopped_share",
]
fmt = {
    "total_cost_eur": "€{:,.0f}",
    "cost_per_1000_txn": "€{:,.1f}",
    "false_decline_rate": "{:.3%}",
    "fraud_eur_stopped_share": "{:.1%}",
}
pd.DataFrame(validation["policies_validation"]).T[cols].style.format(fmt)

# %% [markdown]
# ## 5. What if the assumptions are wrong?
#
# One assumption at a time moved to 50% and 150% of its value. *Regret* = cost of keeping the
# deployed policy (which still believes the original numbers) minus the cost of the policy we
# would have chosen knowing the true value. Small negative values are luck: the policy
# minimises *expected* cost, not the realised cost of one quarter.

# %%
sens = pd.read_csv(policy_dir / "sensitivity.csv")
sens["regret_pct"] = sens["regret_eur"] / sens["best_reoptimised_eur"]
display(
    sens[
        [
            "assumption",
            "factor",
            "value",
            "deployed_policy_eur",
            "best_reoptimised_eur",
            "regret_eur",
            "regret_pct",
        ]
    ].style.format(
        {
            "value": "{:g}",
            "deployed_policy_eur": "€{:,.0f}",
            "best_reoptimised_eur": "€{:,.0f}",
            "regret_eur": "€{:,.0f}",
            "regret_pct": "{:+.1%}",
        }
    )
)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = list(dict.fromkeys(sens["assumption"]))
ypos = {n: i for i, n in enumerate(names[::-1])}
for factor, color, label in [(0.5, plots.BLUE, "× 0.5"), (1.5, plots.ORANGE, "× 1.5")]:
    s = sens[sens["factor"] == factor]
    ax.plot(
        100 * s["regret_pct"],
        [ypos[n] for n in s["assumption"]],
        "o",
        color=color,
        markersize=8,
        label=label,
    )
ax.axvline(0, color=plots.MUTED, linewidth=1)
ax.set_yticks(list(ypos.values()), list(ypos.keys()))
ax.grid(axis="y", visible=False)
ax.set_xlabel("regret: extra cost of the deployed policy (%)")
ax.set_title("Which cost assumption matters most?")
ax.legend(loc="lower right")
fig.tight_layout()
plots.save(fig, "sensitivity_regret", folder="policy")
plt.show()

# %% [markdown]
# ## 6. The single evaluation on test (Oct–Dec 2025)

# %%
model_cols = [
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "roc_auc",
    "precision_at_alert_rate",
    "recall_at_alert_rate",
    "amount_recall_at_alert_rate",
    "brier",
    "mean_predicted",
]
print(
    f"test: {test['transactions']:,} transactions, {test['frauds']:,} frauds "
    f"(base rate {test['base_rate']:.3%})"
)
display(
    pd.Series({k: test["model"][k] for k in model_cols}).to_frame("test").T.style.format("{:.4f}")
)
print(f"validation PR-AUC of the same model: {validation['validation_raw_pr_auc']:.4f}")
pd.DataFrame(test["policies"]).T[cols].style.format(fmt)

# %% [markdown]
# ## 7. Findings
#
# Full write-up: [`reports/policy/phase8_findings.md`](../reports/policy/phase8_findings.md).
#
# - Thresholds must depend on the amount; the "default" 0.5 costs 49% more than the optimum.
# - The model was already calibrated (no class weights); Platt ≈ identity, isotonic overfits.
#   Weak spot: 10–20% predicted vs 21.7% observed, on 207 transactions.
# - Deployed: expected cost per transaction (P2), €62,925 on validation vs €946,616 approving
#   everything. Review capacity never binds.
# - Most sensitive assumption: how much fraud a review stops (+7.3% regret if it is 45%).
# - Test, evaluated once: PR-AUC 0.752 (validation 0.792), but the policy cost (€64,098) and
#   the share of fraud EUR stopped (95.5%) held up.
