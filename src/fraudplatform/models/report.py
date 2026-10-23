"""Comparison report: Markdown table + precision-recall curves on validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraudplatform.eda import plots
from fraudplatform.markdown import md_table
from fraudplatform.models.comparison import summary_table

# Validated categorical order (blue, orange, aqua, yellow, magenta). Three of these are below
# 3:1 contrast on the light surface, so the chart is always published next to its table.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CURVES = [
    "B1_amount",
    "B2_expert_rules",
    "M1_logistic_regression",
    "M2_random_forest",
    "M3_hist_gradient_boosting",
]


def plot_pr_curves(scores: pd.DataFrame, title: str):
    plots.apply_style()
    y = scores["is_fraud"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, color in zip(CURVES, SERIES_COLORS, strict=True):
        precision, recall, _ = precision_recall_curve(y, scores[name].to_numpy())
        ax.plot(recall, precision, color=color, label=name, drawstyle="steps-post")
    base = y.mean()
    ax.axhline(base, color=plots.MUTED, linewidth=1)
    ax.annotate(
        f"random score = base rate {base:.2%}",
        (1, base),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        color=plots.INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("recall (share of fraud caught)")
    ax.set_ylabel("precision (share of alerts that are fraud)")
    ax.set_title(title)
    # empty region between the model curves (top) and the baselines (bottom)
    ax.legend(loc="center left", bbox_to_anchor=(0.36, 0.42))
    fig.tight_layout()
    return fig


def write_report(
    results: pd.DataFrame,
    scores: pd.DataFrame,
    alert_rate: float,
    path: Path,
    context: dict[str, str],
) -> Path:
    table = summary_table(results)
    fmt = table.copy()
    for col in ("pr_auc", "roc_auc", "brier", "mean_predicted"):
        fmt[col] = fmt[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    for col in ("precision_at_alert_rate", "recall_at_alert_rate", "amount_recall_at_alert_rate"):
        fmt[col] = fmt[col].map(lambda v: f"{v:.1%}")
    fig = plot_pr_curves(scores, "Precision-recall on validation (Jul–Sep 2025)")
    figure = plots.save(fig, "pr_curves_validation", folder="models")
    plt.close(fig)
    rel_fig = Path("..") / "figures" / "models" / figure.name
    lines = [
        "# Phase 6: baselines vs first models (validation)",
        "",
        *[f"- {k}: {v}" for k, v in context.items()],
        "",
        f"Alert metrics assume we can flag the top **{alert_rate:.1%}** of transactions.",
        "PR-AUC interval: 95% day-block bootstrap.",
        "",
        md_table(fmt),
        "",
        f"![PR curves]({rel_fig.as_posix()})",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
