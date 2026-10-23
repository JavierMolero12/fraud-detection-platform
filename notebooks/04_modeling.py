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
# # 04 · Baselines and first models
#
# Reads the output of `fraud models compare` (trained on 2024-10-01 → 2025-06-30, evaluated on
# validation 2025-07-01 → 2025-09-30; test not touched). Default hyperparameters: tuning is
# Phase 7.
#
# Questions:
# 1. Does machine learning beat simple rules, and by how much?
# 2. Are the differences between models real or noise?
# 3. What do class weights do?
# 4. How much of the performance comes from signals the simulator made artificially strong?

# %%
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix

from fraudplatform.config import PROJECT_ROOT, load_feature_config, load_model_config, load_splits
from fraudplatform.db import connect
from fraudplatform.eda import plots
from fraudplatform.eda.data import load_features
from fraudplatform.models import baselines, pipelines
from fraudplatform.models.comparison import prepare, summary_table
from fraudplatform.models.metrics import bootstrap_pr_auc_difference, evaluate, top_k_mask
from fraudplatform.models.report import plot_pr_curves

plots.apply_style()
pd.set_option("display.width", 160)

run_dir = PROJECT_ROOT / "data" / "processed" / "phase6"
scores = pd.read_parquet(run_dir / "validation_scores.parquet")
results = pd.read_csv(run_dir / "results.csv")
cfg, feature_cfg, splits = load_model_config(), load_feature_config(), load_splits()
y = scores["is_fraud"].to_numpy()
days = scores["ts"].dt.date.to_numpy()
alert_rate = cfg.evaluation.alert_rate
print(f"validation: {len(scores):,} transactions, {y.sum():,} fraud ({y.mean():.3%})")
print(
    f"alert budget {alert_rate:.1%} -> {int(np.ceil(alert_rate * len(y))):,} alerts; "
    f"best possible recall = {np.ceil(alert_rate * len(y)) / y.sum():.1%}"
)

# %% [markdown]
# ## 1. Comparison on validation

# %%
summary_table(results).style.format(
    {
        "pr_auc": "{:.4f}",
        "roc_auc": "{:.4f}",
        "brier": "{:.4f}",
        "mean_predicted": "{:.4f}",
        "precision_at_alert_rate": "{:.1%}",
        "recall_at_alert_rate": "{:.1%}",
        "amount_recall_at_alert_rate": "{:.1%}",
        "fit_seconds": "{:.0f}",
    },
    na_rep="",
)

# %%
fig = plot_pr_curves(scores, "Precision-recall on validation (Jul–Sep 2025)")
plt.show()

# %% [markdown]
# ## 2. Reading the metrics on the best model
#
# **Confusion matrix** at the alert budget (flag the top 0.5% of scores) and at the
# "default" 0.5 probability threshold. The two flag a different number of transactions:
# 0.5 is a convention that ignores both review capacity and the cost of each error. Here it
# happens to flag MORE than the budget allows (2,268 vs 1,797 alerts), with lower precision.
# Phase 8 chooses thresholds from costs and capacity instead.

# %%
best = "M3_hist_gradient_boosting"
p = scores[best].to_numpy()
for label, flagged in [
    (f"top {alert_rate:.1%} of scores", top_k_mask(p, int(np.ceil(alert_rate * len(p))))),
    ("probability >= 0.5", p >= 0.5),
]:
    tn, fp, fn, tp = confusion_matrix(y, flagged).ravel()
    display(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["actual legit", "actual fraud"],
            columns=["not flagged", "flagged"],
        ).style.set_caption(
            f"{label}: precision {tp / max(tp + fp, 1):.1%}, recall {tp / (tp + fn):.1%}, "
            f"false positive rate {fp / (fp + tn):.3%}"
        )
    )

# %% [markdown]
# **ROC-AUC vs PR-AUC.** The expert rules have a respectable-looking ROC-AUC (0.79) but a
# PR-AUC of 0.26. ROC-AUC rewards ranking a random fraud above a random legit payment, and
# with 117 legit payments per fraud there are so many easy legit ones that a weak ranking
# still scores well. PR-AUC asks what share of the *alerts* are fraud: the question the
# review team cares about.

# %% [markdown]
# ## 3. Are the differences real? Paired day-block bootstrap
#
# Both models are scored on the same resampled days, so the comparison removes the
# day-to-day noise that makes separate intervals overlap.

# %%
pairs = [
    ("M3_hist_gradient_boosting", "M1_logistic_regression"),
    ("M3_hist_gradient_boosting", "M2_random_forest"),
    ("M1_logistic_regression", "M2_random_forest"),
    ("M1_logistic_regression", "B2_expert_rules"),
    ("M3_hist_gradient_boosting", "M3b_hist_gradient_boosting_balanced"),
    ("M1_logistic_regression", "M1b_logistic_regression_balanced"),
]
rows = []
for a, b in pairs:
    out = bootstrap_pr_auc_difference(y, scores[a], scores[b], days, reps=300, seed=11)
    rows.append({"a": a, "b": b, **out})
pd.DataFrame(rows).style.format(
    {"difference": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}", "share_a_better": "{:.1%}"}
)

# %% [markdown]
# ## 4. Class weights: ranking barely changes, probabilities break
#
# `class_weight="balanced"` makes each fraud count ~140× in the loss. It is the common answer
# to imbalance, and resampling (SMOTE, undersampling) has the same effect on the
# probabilities. What it does NOT do is add information.


# %%
def reliability(score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(score, np.linspace(0, 1, bins + 1))
    bucket = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, bins - 1)
    frame = pd.DataFrame({"bucket": bucket, "predicted": score, "observed": y})
    return frame.groupby("bucket").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size")
    )


for model in [
    "M3_hist_gradient_boosting",
    "M3b_hist_gradient_boosting_balanced",
    "M1_logistic_regression",
    "M1b_logistic_regression_balanced",
]:
    r = reliability(scores[model].to_numpy())
    print(f"{model}: mean predicted {scores[model].mean():.4f} vs observed {y.mean():.4f}")
    display(r.tail(3).style.format({"predicted": "{:.3f}", "observed": "{:.3f}"}))

# %% [markdown]
# ## 5. Anatomy of the expert rules
#
# Each rule on its own: how often it fires and how precise it is.

# %%
with connect() as conn:
    valid = prepare(load_features(conn, splits.validation.start, splits.validation.end))
flags = baselines.rule_flags(valid, cfg.rules)
definitions = baselines.rule_definitions(cfg.rules)
rule_rows = []
for name in flags:
    fired = flags[name].to_numpy()
    tp = int((fired & valid["is_fraud"].to_numpy()).sum())
    rule_rows.append(
        {
            "rule": name,
            "definition": definitions[name][0],
            "fires": int(fired.sum()),
            "precision": tp / max(fired.sum(), 1),
            "recall": tp / valid["is_fraud"].sum(),
        }
    )
any_rule = flags.any(axis=1).to_numpy()
tp_any = int((any_rule & valid["is_fraud"].to_numpy()).sum())
rule_rows.append(
    {
        "rule": "ANY rule",
        "definition": "",
        "fires": int(any_rule.sum()),
        "precision": tp_any / any_rule.sum(),
        "recall": tp_any / valid["is_fraud"].sum(),
    }
)
pd.DataFrame(rule_rows).style.format({"fires": "{:,}", "precision": "{:.1%}", "recall": "{:.1%}"})

# %% [markdown]
# ## 6. How much comes from signals the simulator made artificially strong?
#
# The EDA flagged the device operating system and type as unrealistically informative (they
# follow from the attacker device mix chosen in the simulator). Retrain the same gradient
# boosting with both columns masked to a constant, so the model cannot use them.

# %%
masked_cols = ["operating_system", "device_type"]
with connect() as conn:
    train = prepare(
        load_features(
            conn,
            splits.train.start + timedelta(days=feature_cfg.warmup_days),
            splits.train.end,
        )
    )
columns = pipelines.model_columns("tree")
ablation = {}
for label, mask in [("all features", False), ("OS and device type masked", True)]:
    tr, va = train.copy(), valid.copy()
    if mask:
        for c in masked_cols:
            tr[c] = "masked"
            va[c] = "masked"
    model = pipelines.build_hist_gradient_boosting(
        cfg.models["hist_gradient_boosting"], random_state=cfg.random_state
    ).fit(tr[columns], tr["is_fraud"])
    s = model.predict_proba(va[columns])[:, 1]
    ablation[label] = s
    m = evaluate(va["is_fraud"], s, va["amount_eur"], va["ts"].dt.date, alert_rate, True)
    print(
        f"{label:28s} PR-AUC {m['pr_auc']:.4f}  recall@{alert_rate:.1%} "
        f"{m['recall_at_alert_rate']:.1%}"
    )
diff = bootstrap_pr_auc_difference(
    va["is_fraud"],
    ablation["all features"],
    ablation["OS and device type masked"],
    va["ts"].dt.date.to_numpy(),
    reps=300,
    seed=11,
)
diff

# %% [markdown]
# ## 7. Findings
#
# Full write-up: [`reports/models/phase6_findings.md`](../reports/models/phase6_findings.md).
#
# - ML is far ahead of rules: PR-AUC 0.79 (gradient boosting) vs 0.26 (seven expert rules)
#   vs 0.03 (amount alone). At the same 0.5% alert budget it catches 63.2% of fraud with
#   91.8% precision; the rules catch 31.9% with 46.3% precision.
# - Gradient boosting beats logistic regression by +0.024 PR-AUC (paired CI +0.015 to +0.033):
#   real but modest. Logistic regression and random forest are indistinguishable.
# - Class weights do not help ranking (and hurt logistic regression by 0.046) and inflate
#   probabilities 9-17x. No class weights from here on.
# - Masking the artificially strong device OS/type costs only 0.007 PR-AUC: the model does
#   not depend on that simulator artefact, other device features carry the signal.
# - These numbers are high because the data is synthetic and its fraud patterns are cleaner
#   than real ones. The comparisons are the deliverable, not the absolute level.
