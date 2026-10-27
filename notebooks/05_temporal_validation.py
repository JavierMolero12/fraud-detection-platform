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
# # 05 · Temporal validation, tuning and leakage experiments
#
# Reads the output of `fraud models tune` (`data/processed/phase7/`). Training window
# 2024-10-01 → 2025-06-30, split into four expanding folds (test months March to June 2025,
# 14-day gap before each). Validation (Jul–Sep 2025) is only used to confirm choices made with
# cross-validation. The test period is not loaded.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from fraudplatform.config import PROJECT_ROOT, load_model_config
from fraudplatform.eda import plots

plots.apply_style()
pd.set_option("display.width", 160)
run_dir = PROJECT_ROOT / "data" / "processed" / "phase7"
results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
cfg = load_model_config()
pd.DataFrame(results["folds"])

# %% [markdown]
# ## 1. H7: does shuffled cross-validation overstate performance?
#
# Same model (gradient boosting, Phase 6 defaults), same rows, two ways of splitting them.
# Shuffled K-fold lets the model train on transactions from the *future* of the ones it is
# tested on, and puts transactions of the same fraud campaign on both sides of the split.
# The out-of-time validation score (a model trained on the whole window and scored on the
# next three months) is the closest offline estimate of real deployment.

# %%
h7 = results["h7"]
h7_table = pd.DataFrame(
    {
        "temporal fold PR-AUC": h7["temporal_folds"],
        "random fold PR-AUC": h7["random_folds"],
    },
    index=[f["fold"] for f in results["folds"]],
)
display(h7_table.style.format("{:.4f}"))
summary = pd.Series(
    {
        "temporal CV mean": np.mean(h7["temporal_folds"]),
        "random CV mean": np.mean(h7["random_folds"]),
        "validation (out of time)": h7["validation_out_of_time"],
    }
)
display(summary.to_frame("PR-AUC").style.format("{:.4f}"))

fig, ax = plt.subplots(figsize=(7.5, 2.8))
for i, (_label, values, color) in enumerate(
    [
        ("temporal folds", h7["temporal_folds"], plots.BLUE),
        ("random folds", h7["random_folds"], plots.ORANGE),
    ]
):
    ax.plot(values, [i] * len(values), "o", color=color, markersize=8, alpha=0.85)
    ax.plot(np.mean(values), i, "|", color=plots.INK, markersize=18, markeredgewidth=2)
ax.axvline(h7["validation_out_of_time"], color=plots.MUTED, linewidth=1)
ax.annotate(
    f"validation {h7['validation_out_of_time']:.3f}",
    (h7["validation_out_of_time"], 1.45),
    xytext=(4, 0),
    textcoords="offset points",
    color=plots.INK_SECONDARY,
    fontsize=9,
)
ax.set_yticks([0, 1], ["temporal folds", "random folds"])
ax.set_ylim(-0.6, 1.7)
ax.grid(axis="y", visible=False)
ax.set_xlabel("PR-AUC per fold (bar = mean)")
ax.set_title("Same model, two ways to cross-validate")
fig.tight_layout()
plots.save(fig, "h7_random_vs_temporal", folder="models")
plt.show()

# %% [markdown]
# ## 2. Logistic regression: grid over the regularisation strength C

# %%
lr = pd.read_csv(run_dir / "lr_cv_results.csv")
lr.style.format({c: "{:.4f}" for c in lr.columns if "score" in c or "time" in c})

# %% [markdown]
# ## 3. Gradient boosting: randomized search (16 candidates × 4 temporal folds)
#
# Search space (`configs/model.yaml`): learning rate log-uniform 0.02–0.2, iterations
# {200, 400, 800}, leaves {15, 31, 63, 127}, min samples per leaf {20, 50, 100, 200}, L2
# log-uniform 0.001–10, share of features per split {0.5, 0.8, 1.0}. Metric: PR-AUC.

# %%
hgb = pd.read_csv(run_dir / "hgb_cv_results.csv")
hgb.head(10).style.format(
    {c: "{:.4f}" for c in hgb.columns if "score" in c}
    | {
        "param_model__learning_rate": "{:.3f}",
        "param_model__l2_regularization": "{:.3f}",
        "mean_fit_time": "{:.0f}",
    }
)

# %%
spread = hgb["mean_test_score"].agg(["max", "median", "min"])
print(
    f"16 candidates: best {spread['max']:.4f}, median {spread['median']:.4f}, "
    f"worst {spread['min']:.4f}"
)
print(
    f"default configuration through the same folds: "
    f"{results['default_hgb']['cv_temporal_mean']:.4f}"
)

# %% [markdown]
# ## 4. Do the tuned models beat the defaults on validation?

# %%
rows = []
for name, label in [
    ("default_hgb", "gradient boosting, Phase 6 defaults"),
    ("hist_gradient_boosting", "gradient boosting, tuned"),
    ("logistic_regression", "logistic regression, tuned"),
]:
    r = results[name]
    v = r["validation"]
    rows.append(
        {
            "model": label,
            "cv PR-AUC": r.get("cv_best", r.get("cv_temporal_mean")),
            "validation PR-AUC": v["pr_auc"],
            "precision @0.5%": v["precision_at_alert_rate"],
            "recall @0.5%": v["recall_at_alert_rate"],
            "brier": v["brier"],
            "best params": r.get("best_params", ""),
        }
    )
display(
    pd.DataFrame(rows).style.format(
        {
            "cv PR-AUC": "{:.4f}",
            "validation PR-AUC": "{:.4f}",
            "precision @0.5%": "{:.1%}",
            "recall @0.5%": "{:.1%}",
            "brier": "{:.5f}",
        }
    )
)
paired = pd.DataFrame(
    {
        name: results[name]["vs_default_hgb_on_validation"]
        for name in ["hist_gradient_boosting", "logistic_regression"]
    }
).T
paired.style.format(
    {"difference": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}", "share_a_better": "{:.1%}"}
).set_caption(
    "Paired day-block bootstrap on validation: tuned model minus default gradient boosting"
)

# %% [markdown]
# ## 5. H8: how much do leaky features inflate the offline numbers?
#
# The model is trained with the five deliberately leaky features in place of their clean
# counterparts. Offline, validation can be computed with leaky values too, because the
# extract already contains the future. In production the future does not exist: the only
# values the scoring service can compute are the point-in-time ones, so that is what the
# model would receive.

# %%
h8 = results["h8"]
h8_table = pd.Series(
    {
        "clean features: temporal CV": h8["clean_cv_temporal_mean"],
        "leaky features: temporal CV": h8["leaky_cv_temporal_mean"],
        "clean features: validation": h8["clean_validation"],
        "leaky features: validation, leaky values (offline report)": h8["leaky_validation_offline"],
        "leaky features: validation, values available at decision time (production)": h8[
            "leaky_validation_served"
        ],
    }
)
display(h8_table.to_frame("PR-AUC").style.format("{:.4f}"))
print("served leaky model minus clean model (paired bootstrap):", h8["served_vs_clean"])

fig, ax = plt.subplots(figsize=(7.5, 2.9))
labels = ["clean model", "leaky model\n(offline report)", "leaky model\n(production)"]
values = [h8["clean_validation"], h8["leaky_validation_offline"], h8["leaky_validation_served"]]
colors = [plots.BLUE, plots.ORANGE, plots.ORANGE]
y = np.arange(len(labels))[::-1]
ax.hlines(y, 0, values, color=colors, linewidth=2)
ax.plot(values, y, "o", color=plots.BLUE, markersize=8)
for yi, v, c in zip(y, values, colors, strict=True):
    ax.plot(v, yi, "o", color=c, markersize=8)
    ax.annotate(
        f"{v:.3f}",
        (v, yi),
        xytext=(6, -3),
        textcoords="offset points",
        color=plots.INK_SECONDARY,
        fontsize=9,
    )
ax.set_yticks(y, labels)
ax.set_xlim(0, 1.05)
ax.grid(axis="y", visible=False)
ax.set_xlabel("PR-AUC on validation")
ax.set_title("Leaky features: better offline, half as good in production")
fig.tight_layout()
plots.save(fig, "h8_leaky_features", folder="models")
plt.show()

# %% [markdown]
# ## 6. Findings
#
# Full write-up: [`reports/models/phase7_findings.md`](../reports/models/phase7_findings.md).
#
# - **H7 supported, modestly:** shuffled CV 0.817 vs temporal CV 0.799 vs out-of-time
#   validation 0.790. The bigger lesson is the month-to-month spread (0.768 to 0.822).
# - **Tuning is within noise:** best gradient boosting 0.803 in CV vs 0.799 default; on
#   validation +0.004 (CI −0.001 to +0.010). 13 of 16 candidates sit between 0.795 and 0.803.
#   Kept as the selected model (chosen by CV), but the defaults were already near-optimal.
# - **H8 supported, dramatically:** leaky features score 0.852 offline and 0.401 with the
#   values available in production, half as good as the honest model.
#
# Re-run in Phase 13 with a fixed row order; the original numbers are in section 5 of the
# findings.
