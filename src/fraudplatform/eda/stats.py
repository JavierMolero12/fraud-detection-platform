"""Statistics used in the exploratory analysis.

With ~1.4M rows almost every difference is "statistically significant", so these helpers
favour *effect sizes* and *uncertainty intervals* over p-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilson_interval(successes, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a proportion.

    Preferred over the textbook normal interval (p ± z·sqrt(p(1-p)/n)) because fraud rates
    are close to 0: the normal interval can go below 0 and is badly calibrated for small
    counts, which is exactly the situation of a rare class split into many categories.
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(df: pd.DataFrame, by: str, target: str = "is_fraud") -> pd.DataFrame:
    """Fraud rate per category with a 95% Wilson interval and lift over the base rate."""
    base_rate = df[target].mean()
    out = df.groupby(by, observed=True)[target].agg(transactions="size", frauds="sum")
    out["fraud_rate"] = out["frauds"] / out["transactions"]
    out["ci_low"], out["ci_high"] = wilson_interval(out["frauds"], out["transactions"])
    out["lift"] = out["fraud_rate"] / base_rate
    return out.reset_index()


def cramers_v(x: pd.Series, y: pd.Series, bias_correction: bool = True) -> float:
    """Strength of association between two categorical variables, from 0 (none) to 1.

    Uses the Bergsma (2013) bias correction: the raw statistic is inflated for variables
    with many categories, which would make high-cardinality columns look more related.
    """
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    if bias_correction:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def association_with_target(df: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    """Chi-square test of independence and Cramér's V of each categorical column vs target.

    p-values are adjusted for testing many columns (Benjamini-Hochberg). They are reported
    for completeness; with this sample size Cramér's V is the number to read.
    """
    rows = []
    for col in columns:
        table = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        rows.append(
            {
                "feature": col,
                "categories": table.shape[0],
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": cramers_v(df[col], df[target]),
            }
        )
    out = pd.DataFrame(rows)
    out["p_adjusted_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def benjamini_hochberg(p_values) -> np.ndarray:
    """False discovery rate adjusted p-values (Benjamini-Hochberg step-up)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def univariate_auc(values, labels) -> float:
    """P(value of a random positive > value of a random negative); ties count one half.

    It is the ROC-AUC of using the raw column as a score, computed from the Mann-Whitney U
    statistic. 0.5 = no ranking power; below 0.5 the relationship is inverted.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    mask = ~np.isnan(values)
    values, labels = values[mask], labels[mask]
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(values)
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def binned_auc(values, labels, bins: int = 20) -> float:
    """AUC after replacing each value by the fraud rate of its bin; missing values get a bin.

    Plain univariate AUC only rewards monotonic relationships (U-shaped local hour scores
    0.40). Binning first captures any shape, and treating NULL as its own bin keeps the
    information in "no history" / "no device". The bin rates are estimated on the same data,
    so this is an optimistic screening number, not a model metric.
    """
    s = pd.Series(np.asarray(values, dtype=float))
    y = pd.Series(np.asarray(labels, dtype=bool))
    if s.nunique(dropna=True) <= bins:
        codes = s.astype(object).where(s.notna(), "missing").astype(str)
    else:
        # Point masses (e.g. 98% zeros in a velocity count) get their own bin; otherwise the
        # quantile edges collapse and the whole column ends up in a single bin.
        counts = s.value_counts()
        heavy = s.isin(counts[counts > len(s) / bins].index)
        codes = pd.Series("missing", index=s.index, dtype=object)
        codes[heavy] = "value=" + s[heavy].astype(str)
        rest = s.notna() & ~heavy
        if rest.any():
            codes[rest] = pd.qcut(s[rest], q=bins, duplicates="drop").astype(str)
    rate = y.groupby(codes).mean()
    return univariate_auc(codes.map(rate).to_numpy(dtype=float), y.to_numpy())


def tukey_fences(values, k: float = 1.5) -> tuple[float, float]:
    """Classic box-plot outlier fences: [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def precision_from_rates(base_rate: float, recall: float, false_positive_rate: float) -> float:
    """Bayes' rule: share of alerts that are fraud, given prevalence, recall and FPR.

    Shows the base-rate effect: with rare fraud, even a small false positive rate produces
    more false alarms than true ones.
    """
    true_alerts = recall * base_rate
    false_alerts = false_positive_rate * (1 - base_rate)
    return true_alerts / (true_alerts + false_alerts)
