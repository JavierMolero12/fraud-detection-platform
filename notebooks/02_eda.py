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
# # 02 · Exploratory data analysis and statistics
#
# **Scope: the train period only (2024-07-01 → 2025-06-30).** What we learn here will shape
# feature design and modelling decisions. If we also looked at the validation, test or
# production periods, those decisions would be informed by data the model is later evaluated
# on: a quiet form of leakage called *data snooping*. Later periods appear only as row counts.
#
# **Labels:** the observed labels in `core.fraud_labels` (eventually reported fraud), not the
# simulator's ground truth. Labels of the train period are mature (the label window is 120
# days and the extract ends a year later), so a retrospective view is fair here. It would not
# be fair for recent months: see section 2.
#
# **Reading the numbers:** with 1.3M rows nearly every difference is statistically
# significant. Effect sizes (lift, Cramér's V, univariate AUC) and 95% intervals matter more
# than p-values.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from fraudplatform.config import load_splits
from fraudplatform.db import connect
from fraudplatform.eda import plots, stats
from fraudplatform.eda.data import load_transactions, split_sizes

plots.apply_style()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

splits = load_splits()
with connect() as conn:
    sizes = split_sizes(conn, splits)
    df = load_transactions(conn, splits.train.start, splits.train.end)

df["local_hour"] = df["local_ts"].dt.hour
df["local_weekday"] = df["local_ts"].dt.dayofweek
df["month"] = df["ts"].dt.strftime("%Y-%m")
df["cross_border"] = np.where(
    df["txn_country"].astype(str) == df["home_country"].astype(str), "domestic", "cross-border"
)
df["operating_system"] = df["operating_system"].cat.add_categories("none").fillna("none")
sizes

# %% [markdown]
# ## 1. Base rate and class imbalance
#
# The base rate is the share of transactions that are fraud. Every other number in the
# project is read against it: a model that finds nothing is "right" 1 - base rate of the time.

# %%
n = len(df)
frauds = int(df["is_fraud"].sum())
base_rate = frauds / n
ci_low, ci_high = stats.wilson_interval(frauds, n)
days = (splits.train.end - splits.train.start).days

pd.DataFrame(
    {
        "value": [
            f"{n:,}",
            f"{frauds:,}",
            f"{base_rate:.3%} (95% CI {ci_low:.3%} – {ci_high:.3%})",
            f"{n / days:,.0f}",
            f"{frauds / days:,.1f}",
            f"1 : {(n - frauds) / frauds:,.0f}",
        ]
    },
    index=[
        "transactions",
        "labelled fraud",
        "base rate",
        "transactions per day",
        "frauds per day",
        "fraud : legit",
    ],
)

# %% [markdown]
# ### Why accuracy is misleading here
#
# A "model" that approves everything never catches a fraud, yet its accuracy equals
# 1 - base rate. Its confusion matrix:

# %%
confusion = pd.DataFrame(
    [[n - frauds, 0], [frauds, 0]],
    index=["actual legit", "actual fraud"],
    columns=["predicted legit", "predicted fraud"],
)
display(confusion)
print(f"accuracy  = {(n - frauds) / n:.3%}")
print("recall    = 0.000%  (no fraud caught)")
print("precision = undefined (no alerts)")
print(f"PR-AUC of a random score ≈ base rate = {base_rate:.3%};  ROC-AUC of a random score = 0.5")

# %% [markdown]
# ### The base-rate effect (Bayes' rule)
#
# Suppose a detector catches 90% of fraud. How many of its alerts are real fraud?
# `precision = recall·p / (recall·p + FPR·(1 − p))` with p = base rate. Even a 1% false positive
# rate floods the alert queue, because legit transactions outnumber fraud ~117 to 1.

# %%
fpr_grid = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
pd.DataFrame(
    {
        "false_positive_rate": [f"{f:.2%}" for f in fpr_grid],
        "precision_at_90pct_recall": [
            f"{stats.precision_from_rates(base_rate, 0.9, f):.1%}" for f in fpr_grid
        ],
        "alerts_per_day": [round((0.9 * frauds + f * (n - frauds)) / days) for f in fpr_grid],
        "of_which_fraud": [round(0.9 * frauds / days) for _ in fpr_grid],
    }
)

# %% [markdown]
# ## 2. Labels: how late do they arrive?
#
# A label exists only once fraud is reported (chargeback, customer report or investigation).
# The delay decides (a) which labels a historical feature may use at time t and (b) how long
# we must wait before measuring a model in production.

# %%
fraud = df[df["is_fraud"]]
delay_days = (fraud["reported_at"] - fraud["ts"]).dt.total_seconds() / 86_400
within = {f"≤ {d} days": (delay_days <= d).mean() for d in (7, 14, 30, 60, 90, 120)}
display(pd.Series(within, name="share of fraud labels").map("{:.1%}".format).to_frame())
display(
    delay_days.groupby(fraud["label_source"], observed=True)
    .describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
    .round(1)
)
cleared = int(((~df["is_fraud"]) & df["label_source"].notna()).sum())
print(f"disputes investigated and cleared as legit: {cleared:,}")

# %%
x = np.sort(delay_days.to_numpy())
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=plots.BLUE)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
ax.set_xlim(0, 125)
ax.set_ylim(0, 1)
ax.set_title("Share of fraud labels known, by days after the transaction")
ax.set_xlabel("days between transaction and report")
ax.set_ylabel("labels known")
plots.save(fig, "label_delay_ecdf")
plt.show()

# %% [markdown]
# ## 3. Amount: skewness, outliers and signal

# %%
amount = df["amount_eur"]
log_amount = np.log10(amount)
shape = pd.DataFrame(
    {
        "skewness": [stats.stats.skew(amount), stats.stats.skew(log_amount)],
        "excess_kurtosis": [stats.stats.kurtosis(amount), stats.stats.kurtosis(log_amount)],
    },
    index=["amount_eur", "log10(amount_eur)"],
).round(2)
display(amount.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99, 0.999]).round(2).to_frame().T)
display(shape)

# %% [markdown]
# Outliers by Tukey's fences (Q1 − 1.5·IQR, Q3 + 1.5·IQR). On a skewed variable the raw-scale
# rule flags a large share of perfectly normal purchases; on the log scale it flags far fewer.
# Either way, the question that matters for fraud is whether the flagged rows are errors or
# signal.

# %%
rows = []
for name, values in (("raw scale", amount), ("log10 scale", log_amount)):
    low, high = stats.tukey_fences(values)
    upper = values > high
    rows.append(
        {
            "scale": name,
            "upper_fence_eur": round(10**high if name == "log10 scale" else high, 2),
            "flagged_share": f"{upper.mean():.2%}",
            "fraud_rate_flagged": f"{df.loc[upper, 'is_fraud'].mean():.2%}",
            "fraud_rate_rest": f"{df.loc[~upper, 'is_fraud'].mean():.2%}",
        }
    )
display(pd.DataFrame(rows))
print(f"univariate AUC of amount_eur: {stats.univariate_auc(amount, df['is_fraud']):.3f}")

# %%
bins = np.logspace(np.log10(0.5), np.log10(amount.max()), 60)
fig = plots.plot_two_distributions(
    amount[~df["is_fraud"]].to_numpy(),
    amount[df["is_fraud"]].to_numpy(),
    bins,
    "Fraud concentrates in micro-charges (spike at €1.00) and large tickets",
    "amount (EUR, log scale)",
    log_x=True,
)
plots.save(fig, "amount_distribution")
plt.show()

# %%
df["amount_decile"] = pd.qcut(amount, 10, labels=[f"D{i}" for i in range(1, 11)])
by_decile = stats.rate_table(df, "amount_decile")
edges = amount.groupby(df["amount_decile"], observed=True).agg(["min", "max"]).round(2)
display(by_decile.join(edges, on="amount_decile"))
fig = plots.plot_rate_over(
    by_decile, "amount_decile", base_rate, "Fraud rate by amount decile", "amount decile"
)
plots.save(fig, "rate_by_amount_decile")
plt.show()

# %% [markdown]
# ## 4. Time: hour, weekday and month
#
# Hours are LOCAL to the country where the payment happened (terminal or IP country).

# %%
by_hour = stats.rate_table(df, "local_hour")
fig = plots.plot_rate_over(
    by_hour, "local_hour", base_rate, "Fraud rate by local hour", "local hour of the payment"
)
plots.save(fig, "rate_by_hour")
plt.show()
display(by_hour.set_index("local_hour").T.round(4))

# %%
weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_weekday = stats.rate_table(df, "local_weekday")
fig = plots.plot_rate_over(
    by_weekday, "local_weekday", base_rate, "Fraud rate by weekday", "", xticks=weekday_names
)
plots.save(fig, "rate_by_weekday")
plt.show()
display(by_weekday.assign(local_weekday=weekday_names).round(4))
by_month = stats.rate_table(df, "month")
fig = plots.plot_rate_over(
    by_month, "month", base_rate, "Fraud rate by month (train period)", "transaction month"
)
fig.axes[0].tick_params(axis="x", rotation=45)
plots.save(fig, "rate_by_month")
plt.show()
display(by_month)

# %% [markdown]
# ## 5. Categorical attributes vs fraud
#
# Chi-square test of independence and Cramér's V (bias-corrected, 0 = no association,
# 1 = perfect). p-values are Benjamini-Hochberg adjusted for testing many attributes.

# %%
categorical = [
    "payment_method",
    "merchant_category",
    "merchant_risk_level",
    "merchant_channel",
    "customer_segment",
    "card_type",
    "device_type",
    "operating_system",
    "txn_country",
    "home_country",
    "cross_border",
]
association = stats.association_with_target(df, categorical, "is_fraud")
association.style.format(
    {"chi2": "{:,.0f}", "p_value": "{:.1e}", "p_adjusted_bh": "{:.1e}", "cramers_v": "{:.3f}"}
)

# %%
for column, title in [
    ("payment_method", "Fraud rate by payment method"),
    ("merchant_category", "Fraud rate by merchant category"),
    ("merchant_risk_level", "Fraud rate by merchant risk rating"),
    ("customer_segment", "Fraud rate by customer segment"),
    ("card_type", "Fraud rate by card type"),
    ("device_type", "Fraud rate by device type"),
    ("operating_system", "Fraud rate by device operating system"),
    ("cross_border", "Fraud rate: domestic vs cross-border"),
    ("txn_country", "Fraud rate by transaction country"),
]:
    table = stats.rate_table(df, column)
    fig = plots.plot_rate_by_category(table, column, base_rate, title)
    plots.save(fig, f"rate_by_{column}")
    plt.show()
    display(table.round(4))

# %% [markdown]
# ### Is the country effect real, or a confounder?
#
# Some transaction countries show fraud rates 10× the base rate. Before calling a country
# "risky", split each country by whether the payment is domestic or cross-border for the
# customer. If the domestic rates are similar everywhere, the country effect is just the
# share of cross-border payments in disguise.

# %%
by_country = df.pivot_table(
    index="txn_country",
    columns="cross_border",
    values="is_fraud",
    aggfunc=["size", "mean"],
    observed=True,
)
by_country.columns = [f"{stat}_{kind}" for stat, kind in by_country.columns]
by_country["cross_border_share"] = by_country["size_cross-border"] / (
    by_country["size_cross-border"] + by_country["size_domestic"]
)
display(
    by_country.style.format(
        {
            "size_cross-border": "{:,}",
            "size_domestic": "{:,}",
            "mean_cross-border": "{:.2%}",
            "mean_domestic": "{:.2%}",
            "cross_border_share": "{:.1%}",
        }
    )
)
df["presence"] = np.where(
    df["payment_method"].isin(["chip_pin", "contactless"]), "card present", "card not present"
)
stats.rate_table(df.assign(segment=df["presence"] + " · " + df["cross_border"]), "segment")

# %% [markdown]
# ## 6. Numeric attributes
#
# Univariate AUC: the probability that a random fraud has a higher value than a random legit
# transaction (0.5 = useless on its own; below 0.5 = fraud tends to have LOWER values).

# %%
numeric = ["amount_eur", "customer_age", "account_age_days", "card_age_days", "local_hour"]
pd.DataFrame(
    {
        "univariate_auc": [stats.univariate_auc(df[c], df["is_fraud"]) for c in numeric],
        "median_legit": [df.loc[~df["is_fraud"], c].median() for c in numeric],
        "median_fraud": [df.loc[df["is_fraud"], c].median() for c in numeric],
    },
    index=numeric,
).round(3)

# %%
age_bins = [-1, 7, 30, 90, 365, 3 * 365, 100 * 365]
age_labels = ["≤ 7 d", "8–30 d", "31–90 d", "91–365 d", "1–3 y", "> 3 y"]
df["account_age_bucket"] = pd.cut(df["account_age_days"], age_bins, labels=age_labels)
by_account_age = stats.rate_table(df, "account_age_bucket")
fig = plots.plot_rate_over(
    by_account_age, "account_age_bucket", base_rate, "Fraud rate by account age", "account age"
)
plots.save(fig, "rate_by_account_age")
plt.show()
display(by_account_age)

# %% [markdown]
# ## 7. Interactions (preview for feature design)
#
# Fraud rate by channel and amount band. A model that only adds effects (logistic regression
# without interaction terms) cannot represent a pattern that exists in one channel only.

# %%
df["amount_band"] = pd.cut(
    amount, [0, 5, 20, 100, 500, np.inf], labels=["< 5", "5–20", "20–100", "100–500", "≥ 500"]
)
pivot = df.pivot_table(
    index="payment_method",
    columns="amount_band",
    values="is_fraud",
    aggfunc="mean",
    observed=True,
)
counts = df.pivot_table(
    index="payment_method",
    columns="amount_band",
    values="is_fraud",
    aggfunc="size",
    observed=True,
)
display(pivot.style.format("{:.2%}").set_caption("fraud rate"))
display(counts.style.format("{:,}").set_caption("transactions"))

# %% [markdown]
# ## 8. Associations between attributes
#
# Strongly associated predictors carry overlapping information (and make linear-model
# coefficients unstable). Cramér's V for categorical pairs, Spearman for numeric pairs.

# %%
pair_columns = [c for c in categorical if c != "operating_system"]
cramer = pd.DataFrame(index=pair_columns, columns=pair_columns, dtype=float)
for i, a in enumerate(pair_columns):
    for b in pair_columns[i:]:
        value = 1.0 if a == b else stats.cramers_v(df[a], df[b])
        cramer.loc[a, b] = cramer.loc[b, a] = value
fig = plots.plot_heatmap(cramer, "Cramér's V between categorical attributes")
plots.save(fig, "cramers_v_matrix")
plt.show()

# %%
spearman = df[numeric].corr(method="spearman")
fig = plots.plot_heatmap(
    spearman, "Spearman correlation between numeric attributes", diverging=True
)
plots.save(fig, "spearman_matrix")
plt.show()
spearman.round(3)

# %% [markdown]
# ## 9. Findings
#
# The full write-up, with every number from this run and the caveats about which signals
# are artificial, is in [`reports/eda/eda_findings.md`](../reports/eda/eda_findings.md).
# In short:
#
# - Base rate 0.848%: accuracy is meaningless; precision collapses at modest false positive
#   rates; review capacity will drive the decision policy.
# - Only 42.9% of fraud labels exist two weeks after the transaction.
# - Amount is extremely skewed and U-shaped (micro-charges and large tickets); outliers are
#   signal.
# - Cross-border is the strongest attribute; the apparent country effect is a confounder of it.
# - Night hours, merchant category, device type and card type matter; weekday, customer age
#   and account age barely do on their own.
# - Several strong signals (operating system, night hours) are strong *because of how the
#   simulator was built*.
