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
# # 03 · Feature analysis (train period only)
#
# Screening of the point-in-time features built in Phase 5
# (`features.transaction_features`, catalog in `docs/feature_catalog.md`).
#
# - **Same scope as the EDA:** train period only, eventual labels (retrospective view).
# - **Univariate only.** A feature that is weak alone can matter in combination, and two
#   strong features can be redundant. Whether a feature *adds* value is decided by ablation in
#   Phases 6-7, not here.
# - Two screening numbers per feature:
#   - **AUC**: plain univariate AUC (monotonic ranking; 0.5 = none, < 0.5 = inverse).
#   - **binned AUC**: AUC after replacing values by the fraud rate of their quantile bin, with
#     NULL as its own bin. Captures U-shapes and the information in "no history".

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from fraudplatform.config import load_feature_config, load_splits
from fraudplatform.db import connect
from fraudplatform.eda import plots, stats
from fraudplatform.eda.data import load_features
from fraudplatform.features import catalog

plots.apply_style()
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 80)

splits = load_splits()
feature_cfg = load_feature_config()
with connect() as conn:
    df = load_features(conn, splits.train.start, splits.train.end, include_leaky=True)
    build = conn.execute(
        "SELECT build_id, feature_set_version, rows, seconds, built_at "
        "FROM features.builds ORDER BY build_id DESC LIMIT 1"
    ).fetchone()
base_rate = df["is_fraud"].mean()
print(f"feature build {build[0]} (v{build[1]}): {build[2]:,} rows in {build[3]}s")
print(f"train rows: {len(df):,}   base rate: {base_rate:.3%}")

# %% [markdown]
# ## 1. Coverage
#
# NULL is meaningful here: no device (card present), no history yet (first transaction of a
# card or customer in the data), or no standard deviation (fewer than two past payments).

# %%
coverage = pd.DataFrame(
    {
        "group": [f.group for f in catalog.FEATURES],
        "kind": [f.kind for f in catalog.FEATURES],
        "null_share": [df[f.name].isna().mean() for f in catalog.FEATURES],
    },
    index=catalog.FEATURE_NAMES,
)
coverage[coverage["null_share"] > 0].style.format({"null_share": "{:.1%}"})

# %% [markdown]
# ## 2. Univariate screening

# %%
rows = []
for f in catalog.FEATURES:
    if f.kind == "categorical":
        rows.append(
            {
                "feature": f.name,
                "group": f.group,
                "auc": np.nan,
                "binned_auc": stats.binned_auc(
                    df[f.name].cat.codes.replace(-1, np.nan), df["is_fraud"]
                ),
                "cramers_v": stats.cramers_v(df[f.name], df["is_fraud"]),
            }
        )
    else:
        rows.append(
            {
                "feature": f.name,
                "group": f.group,
                "auc": stats.univariate_auc(df[f.name], df["is_fraud"]),
                "binned_auc": stats.binned_auc(df[f.name], df["is_fraud"]),
                "cramers_v": np.nan,
            }
        )
screen = pd.DataFrame(rows).sort_values("binned_auc", ascending=False).reset_index(drop=True)
screen.style.format({"auc": "{:.3f}", "binned_auc": "{:.3f}", "cramers_v": "{:.3f}"})

# %%
top = screen.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(top) + 1.2))
y = np.arange(len(top))
ax.hlines(y, 0.5, top["binned_auc"], color=plots.BLUE, linewidth=2)
ax.plot(top["binned_auc"], y, "o", color=plots.BLUE, markersize=8)
ax.set_yticks(y, top["feature"])
ax.grid(axis="y", visible=False)
ax.axvline(0.5, color=plots.MUTED, linewidth=1)
ax.set_xlim(0.45, 1.0)
ax.set_title("Top 25 features by binned univariate AUC (train)")
ax.set_xlabel("binned AUC (0.5 = no signal on its own)")
fig.tight_layout()
plots.save(fig, "binned_auc_top25", folder="features")
plt.show()

# %% [markdown]
# ## 3. Fraud rate by value of key features


# %%
def bucket_rates(column: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    bucket = pd.cut(df[column], edges, labels=labels, right=False)
    bucket = bucket.cat.add_categories("NULL").fillna("NULL")
    return stats.rate_table(df.assign(bucket=bucket), "bucket").rename(columns={"bucket": column})


inf = np.inf
for column, edges, labels in [
    ("card_txn_count_1h", [0, 1, 2, 3, inf], ["0", "1", "2", "3+"]),
    ("card_micro_count_1h", [0, 1, 2, inf], ["0", "1", "2+"]),
    ("device_unique_cards_24h", [0, 1, 2, 3, inf], ["0", "1", "2", "3+"]),
    ("device_unique_customers_30d", [0, 1, 2, 3, 6, inf], ["0", "1", "2", "3-5", "6+"]),
    (
        "amount_to_cust_median_90d",
        [0, 0.5, 1, 2, 5, 10, inf],
        ["< 0.5x", "0.5-1x", "1-2x", "2-5x", "5-10x", "≥ 10x"],
    ),
    ("device_age_days", [0, 1 / 24, 1, 30, inf], ["< 1 h", "1 h - 1 d", "1-30 d", "> 30 d"]),
    ("card_prior_fraud_reports", [0, 1, inf], ["0", "≥ 1"]),
]:
    display(bucket_rates(column, edges, labels).round(4))

# %%
flags = [
    "is_micro_amount",
    "is_night",
    "cross_border",
    "merchant_foreign",
    "is_card_present",
    "merchant_new_for_customer",
    "category_new_for_customer",
    "device_new_for_customer",
    "country_changed",
    "new_country_for_customer_180d",
    "impossible_travel",
]
flag_rows = []
for flag in flags:
    for value, label in ((1.0, "true"), (0.0, "false")):
        mask = df[flag] == value
        k, n = int(df.loc[mask, "is_fraud"].sum()), int(mask.sum())
        lo, hi = stats.wilson_interval(k, n)
        flag_rows.append(
            {
                "feature": flag,
                "value": label,
                "transactions": n,
                "fraud_rate": k / n if n else np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "lift": (k / n) / base_rate if n else np.nan,
            }
        )
pd.DataFrame(flag_rows).style.format(
    {
        "transactions": "{:,}",
        "fraud_rate": "{:.2%}",
        "ci_low": "{:.2%}",
        "ci_high": "{:.2%}",
        "lift": "{:.2f}",
    }
)

# %% [markdown]
# ## 4. Warm-up: history starts on 2024-07-01 for everybody
#
# Novelty and profile features need history. At the start of the data every merchant,
# category and country looks "new" because earlier history does not exist in the extract,
# not because the customer's behaviour is new. This bias fades as history accumulates.

# %%
df["month"] = df["ts"].dt.strftime("%Y-%m")
warmup = df.groupby("month").agg(
    merchant_new=("merchant_new_for_customer", "mean"),
    category_new=("category_new_for_customer", "mean"),
    country_new_180d=("new_country_for_customer_180d", "mean"),
    no_90d_history=("cust_txn_count_90d", lambda s: (s == 0).mean()),
)
display(warmup.style.format("{:.1%}"))

fig, ax = plt.subplots(figsize=(7.5, 3.4))
xs = np.arange(len(warmup))
ax.plot(xs, warmup["merchant_new"], color=plots.BLUE, marker="o", markersize=5)
ax.annotate(
    "merchant new for customer",
    (xs[-1], warmup["merchant_new"].iloc[-1]),
    xytext=(-4, 8),
    textcoords="offset points",
    ha="right",
    color=plots.INK_SECONDARY,
)
ax.plot(xs, warmup["category_new"], color=plots.ORANGE, marker="o", markersize=5)
ax.annotate(
    "category new for customer",
    (xs[-1], warmup["category_new"].iloc[-1]),
    xytext=(-4, 8),
    textcoords="offset points",
    ha="right",
    color=plots.INK_SECONDARY,
)
ax.set_xticks(xs, warmup.index, rotation=45)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0)
ax.set_title("Share of transactions flagged 'new', by month: the warm-up bias")
ax.legend(["merchant new", "category new"], loc="upper right")
fig.tight_layout()
plots.save(fig, "warmup_novelty", folder="features")
plt.show()

# %% [markdown]
# ## 5. Leaky vs point-in-time features
#
# Each deliberately leaky feature next to its clean counterpart (same idea, computed with
# information that would not exist at decision time). The time-travel test proves they leak;
# here we see how much better they *look*.

# %%
leak_rows = []
for leaky, clean in catalog.LEAKY_FEATURES.items():
    leak_rows.append(
        {
            "clean_feature": clean,
            "clean_binned_auc": stats.binned_auc(df[clean], df["is_fraud"]),
            "leaky_feature": leaky,
            "leaky_binned_auc": stats.binned_auc(df[leaky], df["is_fraud"]),
        }
    )
leak = pd.DataFrame(leak_rows)
leak["inflation"] = leak["leaky_binned_auc"] - leak["clean_binned_auc"]
leak.style.format(
    {"clean_binned_auc": "{:.3f}", "leaky_binned_auc": "{:.3f}", "inflation": "{:+.3f}"}
)

# %% [markdown]
# ## 6. Redundancy between numeric features
#
# Pairs with |Spearman ρ| ≥ 0.85: they carry largely the same information.

# %%
numeric = [c for c in catalog.NUMERIC + catalog.BOOLEAN]
rho = df[numeric].corr(method="spearman")
pairs = (
    rho.where(np.triu(np.ones(rho.shape, dtype=bool), k=1))
    .stack()
    .rename("spearman")
    .reset_index()
    .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
)
pairs[pairs["spearman"].abs() >= 0.85].sort_values("spearman", key=abs, ascending=False).round(3)

# %% [markdown]
# ## 7. Findings
#
# Full write-up with every number from this run:
# [`reports/features/feature_findings.md`](../reports/features/feature_findings.md). In short:
#
# - Strongest on their own: device age (new devices), merchant fraud rate as known at ts,
#   time since the card's previous use, merchant category, cross-border, 24 h card velocity.
# - Card velocity (H1) and device sharing (H3) behave as expected; relative amount (H2) is
#   not better than the raw amount on its own.
# - `card_prior_fraud_reports` is useless point-in-time (reports arrive too late) while its
#   naive leaky version looks excellent: +0.28 binned AUC of pure leakage.
# - Every leaky variant looks better than its clean counterpart.
# - Warm-up bias: novelty features are inflated in the first months; train from 2024-10-01.
