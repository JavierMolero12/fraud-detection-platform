"""Matplotlib charts for the analysis notebooks.

Visual rules (validated palette, light surface): one hue per series in a fixed order
(blue = all / legit, orange = fraud), thin 2px lines, >= 8px markers, solid hairline grid,
recessive axes, text in neutral ink (never in the series colour), base rate drawn as a muted
reference line with a direct label. Every chart is printed next to the table it comes from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from fraudplatform.config import PROJECT_ROOT

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # categorical slot 1: all transactions / legit
ORANGE = "#eb6834"  # categorical slot 2: fraud

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8f8e", "#b8302f"]
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


def _pct_axis(ax, axis: str = "x") -> None:
    fmt = mpl.ticker.PercentFormatter(xmax=1, decimals=1)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _base_rate_line(ax, base_rate: float, vertical: bool) -> None:
    if vertical:
        ax.axvline(base_rate, color=MUTED, linewidth=1, zorder=1)
        ax.annotate(
            f"base rate {base_rate:.2%}",
            (base_rate, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            va="top",
            color=INK_SECONDARY,
            fontsize=9,
        )
    else:
        ax.axhline(base_rate, color=MUTED, linewidth=1, zorder=1)
        ax.annotate(
            f"base rate {base_rate:.2%}",
            (0, base_rate),
            xycoords=("axes fraction", "data"),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
        )


def plot_rate_by_category(
    table: pd.DataFrame, category: str, base_rate: float, title: str, sort: bool = True
):
    """Dot + 95% interval per category (horizontal), with the base rate as reference."""
    t = table.sort_values("fraud_rate") if sort else table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.hlines(y, t["ci_low"], t["ci_high"], color=BLUE, linewidth=2, zorder=2)
    ax.plot(t["fraud_rate"], y, "o", color=BLUE, markersize=8, zorder=3)
    ax.set_yticks(y, t[category].astype(str))
    ax.grid(axis="y", visible=False)
    _pct_axis(ax, "x")
    ax.set_xlim(left=0)
    _base_rate_line(ax, base_rate, vertical=True)
    ax.set_title(title)
    ax.set_xlabel("fraud rate (95% Wilson interval)")
    fig.tight_layout()
    return fig


def plot_rate_over(
    table: pd.DataFrame, x: str, base_rate: float, title: str, xlabel: str, xticks=None
):
    """Fraud rate along an ordered axis (hour, weekday, month, decile) with its interval."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    xs = np.arange(len(table))
    ax.fill_between(xs, table["ci_low"], table["ci_high"], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(xs, table["fraud_rate"], color=BLUE, marker="o", markersize=5)
    ax.set_xticks(xs, xticks if xticks is not None else table[x].astype(str))
    _pct_axis(ax, "y")
    ax.set_ylim(bottom=0)
    _base_rate_line(ax, base_rate, vertical=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraud rate")
    fig.tight_layout()
    return fig


def plot_two_distributions(
    legit: np.ndarray, fraud: np.ndarray, bins: np.ndarray, title: str, xlabel: str, log_x=False
):
    """Normalised histograms of legit vs fraud (shares, so the rare class is visible)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for values, color, label in ((legit, BLUE, "legit"), (fraud, ORANGE, "fraud")):
        weights = np.full(len(values), 1 / len(values))
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            color=color,
            linewidth=2,
            label=label,
        )
    if log_x:
        ax.set_xscale("log")
    _pct_axis(ax, "y")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("share of transactions")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_heatmap(matrix: pd.DataFrame, title: str, diverging: bool = False):
    """Association matrix. Sequential blue for 0..1, blue-gray-red for -1..1."""
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.6, 0.55 * n + 1.4))
    cmap, vmin = (DIVERGING, -1) if diverging else (SEQUENTIAL, 0)
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=vmin, vmax=1)
    ax.set_xticks(range(n), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), matrix.index)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            value = matrix.iloc[i, j]
            strong = abs(value) > 0.55
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if strong else INK_SECONDARY,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save(fig, name: str, folder: str = "eda") -> Path:
    path = FIGURE_DIR / folder / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
