"""Self-contained HTML report of the monitoring replay: inline SVG charts, no dependencies.

Charts follow the project's data-viz rules: one y-axis per chart, thin 2px lines, a legend from
two series up, hover tooltips on every point (enlarged hit areas), status colour only for
alerts and always with a symbol and a label, a data table under every chart, light and dark
themes from CSS variables.
"""

from __future__ import annotations

import html
import math
from datetime import date, timedelta

import pandas as pd

WIDTH, HEIGHT = 720, 190
LEFT, RIGHT, TOP, BOTTOM = 52, 132, 16, 26
SERIES = ("var(--series-1)", "var(--series-2)", "var(--series-3)")

CSS = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --critical: #d03b3b; --warning: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
body { margin: 0; background: var(--page); }
.viz-root { background: var(--page); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px 16px; }
main { max-width: 860px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
p, li { color: var(--text-secondary); } .lead { color: var(--text-primary); }
figure { margin: 16px 0 8px; padding: 12px; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; }
figcaption { font-weight: 600; margin-bottom: 2px; }
figcaption span { display: block; font-weight: 400; color: var(--text-secondary); font-size: 13px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-muted); }
svg .label { fill: var(--text-secondary); }
.hit { fill: transparent; } .hit:hover { fill: currentColor; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--text-secondary);
  margin: 6px 0 0; }
.legend i { display: inline-block; width: 14px; height: 2px; vertical-align: middle;
  margin-right: 6px; border-radius: 1px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid);
  white-space: nowrap; }
td:last-child { white-space: normal; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details { margin: 4px 0 0; font-size: 13px; color: var(--text-secondary); }
.tag { display: inline-block; padding: 0 6px; border-radius: 4px; font-size: 12px;
  border: 1px solid var(--border); color: var(--text-primary); }
.scroll { overflow-x: auto; }
@media (max-width: 600px) { .viz-root { padding: 16px; } h1 { font-size: 20px; } }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _nice_max(value: float) -> float:
    if value <= 0 or math.isnan(value):
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * exp:
            return step * exp
    return 10 * exp


class Chart:
    """One line/bar chart on a shared date axis (one y-axis, never two)."""

    def __init__(self, start: date, end: date, y_max: float, y_format, y_min: float = 0.0):
        self.start, self.end = start, end
        self.y_min, self.y_max, self.fmt = y_min, y_max, y_format
        self.parts: list[str] = []
        self.legend: list[tuple[str, str, str]] = []

    def x(self, day: date) -> float:
        span = (self.end - self.start).days
        return LEFT + (WIDTH - LEFT - RIGHT) * (day - self.start).days / span

    def y(self, value: float) -> float:
        frac = (value - self.y_min) / (self.y_max - self.y_min)
        return TOP + (HEIGHT - TOP - BOTTOM) * (1 - min(max(frac, 0), 1.05))

    def axes(self, ticks: int = 4) -> None:
        """Gridlines at round steps (1, 2 or 5 x 10^n), about `ticks` of them."""
        raw = (self.y_max - self.y_min) / ticks
        exp = 10 ** math.floor(math.log10(raw))
        step = max(s * exp for s in (1, 2, 5) if s * exp <= raw * 1.0001)
        n = int(round((self.y_max - self.y_min) / step))
        for i in range(n + 1):
            v = self.y_min + step * i
            if v > self.y_max * 1.0001 + 1e-12:
                break
            y = self.y(v)
            stroke = "var(--axis)" if i == 0 else "var(--grid)"
            self.parts.append(
                f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" '
                f'stroke="{stroke}" stroke-width="1"/>'
                f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{_esc(self.fmt(v))}</text>'
            )
        month = date(self.start.year, self.start.month, 1)
        while month <= self.end:
            if month >= self.start:
                x = self.x(month)
                self.parts.append(
                    f'<text x="{x:.1f}" y="{HEIGHT - 6}" text-anchor="middle">'
                    f"{month.strftime('%b')}</text>"
                )
            month = date(month.year + month.month // 12, month.month % 12 + 1, 1)

    def vline(self, day: date, label: str) -> None:
        x = self.x(day)
        self.parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{HEIGHT - BOTTOM}" '
            f'stroke="var(--text-secondary)" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text class="label" x="{x - 4:.1f}" y="{TOP + 10}" text-anchor="end">'
            f"{_esc(label)}</text>"
        )

    def hline(self, value: float, label: str) -> None:
        y = self.y(value)
        self.parts.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="5 4"/>'
            f'<text class="label" x="{WIDTH - RIGHT + 6}" y="{y + 4:.1f}">{_esc(label)}</text>'
        )

    def line(self, days, values, name: str, slot: int = 0, label_end: bool = True) -> None:
        colour = SERIES[slot]
        points = [(d, v) for d, v in zip(days, values, strict=True) if not pd.isna(v)]
        if not points:
            return
        coords = " ".join(f"{self.x(d):.1f},{self.y(v):.1f}" for d, v in points)
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for d, v in points:  # hover: enlarged hit area, marker and tooltip
            self.parts.append(
                f'<circle class="hit" style="color:{colour}" cx="{self.x(d):.1f}" '
                f'cy="{self.y(v):.1f}" r="7"><title>{_esc(name)} · {d:%d %b %Y}: '
                f"{_esc(self.fmt(v))}</title></circle>"
            )
        if label_end:
            d, v = points[-1]
            self.parts.append(
                f'<text class="label" x="{self.x(d) + 8:.1f}" y="{self.y(v) + 4:.1f}">'
                f"{_esc(name)}</text>"
            )
        self.legend.append(("line", colour, name))

    def bars(self, days, values, name: str, width_days: int = 7, slot: int = 0) -> None:
        colour = SERIES[slot]
        for d, v in zip(days, values, strict=True):
            x0, x1 = self.x(d) + 1, self.x(d + timedelta(days=width_days)) - 1  # 2px gap
            y_top, y_base = self.y(v), self.y(self.y_min)
            r = min(4, (x1 - x0) / 2, max(y_base - y_top, 0))
            path = (
                f"M{x0:.1f},{y_base:.1f} V{y_top + r:.1f} Q{x0:.1f},{y_top:.1f} {x0 + r:.1f},"
                f"{y_top:.1f} H{x1 - r:.1f} Q{x1:.1f},{y_top:.1f} {x1:.1f},{y_top + r:.1f} "
                f"V{y_base:.1f} Z"
            )
            self.parts.append(
                f'<path d="{path}" fill="{colour}"><title>{_esc(name)} · week of '
                f"{d:%d %b %Y}: {_esc(self.fmt(v))}</title></path>"
            )
        self.legend.append(("bar", colour, name))

    def alerts(self, days, values, label: str, what: str = "") -> None:
        """Alert markers; `what` names the series in the tooltip, the legend shows `label` once."""
        for d, v in zip(days, values, strict=True):
            x, y = self.x(d), self.y(v) - 9
            triangle = f"M{x:.1f},{y - 5:.1f} L{x + 5:.1f},{y + 4:.1f} L{x - 5:.1f},{y + 4:.1f} Z"
            tip = f"{label}{' (' + what + ')' if what else ''} · {d:%d %b %Y}: {self.fmt(v)}"
            self.parts.append(
                f'<path d="{triangle}" '
                f'fill="var(--critical)" stroke="var(--surface-1)" stroke-width="1.5">'
                f"<title>{_esc(tip)}</title></path>"
            )
        entry = ("alert", "var(--critical)", label)
        if len(days) and entry not in self.legend:
            self.legend.append(entry)

    def svg(self, description: str) -> str:
        body = "".join(self.parts)
        return (
            f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{_esc(description)}">'
            f"{body}</svg>"
        )

    def legend_html(self) -> str:
        # One data series: the title names it, no legend. The alert symbol always gets one.
        data = [i for i in self.legend if i[0] != "alert"]
        items = (data if len(data) >= 2 else []) + [i for i in self.legend if i[0] == "alert"]
        if not items:
            return ""
        spans = []
        for kind, colour, name in items:
            if kind == "alert":
                spans.append(f'<span><span style="color:{colour}">▲</span> {_esc(name)}</span>')
            else:
                h = "10px" if kind == "bar" else "2px"
                spans.append(
                    f'<span><i style="background:{colour};height:{h}"></i>{_esc(name)}</span>'
                )
        return f'<div class="legend">{"".join(spans)}</div>'


def figure(title: str, subtitle: str, chart: Chart, table: pd.DataFrame, description: str) -> str:
    return (
        f"<figure><figcaption>{_esc(title)}<span>{_esc(subtitle)}</span></figcaption>"
        f"{chart.svg(description)}{chart.legend_html()}"
        f"<details><summary>Data table</summary><div class='scroll'>{table_html(table)}</div>"
        f"</details></figure>"
    )


def table_html(df: pd.DataFrame) -> str:
    def cell(v, numeric):
        if isinstance(v, float):
            text = "" if math.isnan(v) else f"{v:.4g}"
        else:
            text = "" if v is None else str(v)
        return f'<td class="num">{_esc(text)}</td>' if numeric else f"<td>{_esc(text)}</td>"

    numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
    head = "".join(
        f'<th class="{"num" if n else ""}">{_esc(c)}</th>'
        for c, n in zip(df.columns, numeric, strict=True)
    )
    rows = "".join(
        "<tr>" + "".join(cell(v, n) for v, n in zip(r, numeric, strict=True)) + "</tr>"
        for r in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def page(title: str, body: str) -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{CSS}</style></head>"
        f'<body><div class="viz-root"><main>{body}</main></div></body></html>'
    )


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _ratio(v: float) -> str:
    return f"{v:.2f}x"


def _psi(v: float) -> str:
    return f"{v:.3f}"


def _noisy_note(alerts: pd.DataFrame, noisy: list[str]) -> str:
    psi_alerts = alerts[alerts["rule"] == "input_psi"]
    if not len(psi_alerts) or not set(psi_alerts["subject"]) <= set(noisy):
        return ""
    return (
        f"Every input_psi alert is {', '.join(noisy)}, which alerts from the first run: it drifts "
        "by construction, so its 'first after drift' says nothing about the drift."
    )


def _times(n: int) -> str:
    return {0: "never", 1: "once", 2: "twice"}.get(n, f"{n} times")


def _dates(series) -> pd.Series:
    return pd.to_datetime(pd.Series(series)).dt.date


def build_report(
    tables: dict[str, pd.DataFrame],
    evaluation: dict,
    reference_metrics: dict[str, float],
    drift_start: date,
    start: date,
    end: date,
) -> str:
    """The whole page. `evaluation` holds the frames and numbers of monitoring.evaluation."""
    s = evaluation["summary"]
    m = tables["metrics"].pivot(index="as_of", columns="metric", values="value")
    m.index = _dates(m.index)
    alerts = tables["alerts"].assign(as_of=_dates(tables["alerts"]["as_of"]).values)
    drift_label = f"drift injected {drift_start:%d %b}"
    figures = []

    # 1. input drift, fixed reference (the rule fixed in advance)
    fd = tables["feature_drift"].assign(as_of=_dates(tables["feature_drift"]["as_of"]).values)
    stable = fd[~fd["feature"].isin(evaluation["noisy_features"])]
    top = stable.loc[stable.groupby("as_of")["psi"].idxmax(), ["as_of", "feature", "psi"]]
    c = Chart(start, end, 0.12, _psi)
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.hline(0.10, "alert rule 0.10")
    c.hline(s["reference_week_noise_max"], "noise in reference")
    c.line(list(top["as_of"]), list(top["psi"]), "largest PSI", label_end=False)
    figures.append(
        figure(
            "Input drift against the reference period (rule fixed in advance)",
            f"Largest weekly PSI among the {s['stable_inputs']} stable model inputs. "
            f"{', '.join(evaluation['noisy_features'])} is left out: it drifts by construction "
            f"(PSI {s['noisy_feature_psi_min']:.2f} to {s['noisy_feature_psi_max']:.2f} "
            "every week).",
            c,
            top.rename(columns={"as_of": "as of", "psi": "PSI"}),
            "Largest input PSI per week; a step in March stays below the 0.10 rule.",
        )
    )

    # 2. input drift, rolling reference (post hoc)
    roll = evaluation["rolling_production"]
    rtop = roll.loc[roll.groupby("week_start")["psi"].idxmax(), ["week_start", "feature", "psi"]]
    rtop = rtop.assign(as_of=[d + timedelta(days=7) for d in rtop["week_start"]])
    c = Chart(start, end, _nice_max(max(rtop["psi"].max(), 0.04)), _psi)
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.hline(evaluation["rolling_threshold"], f"post-hoc rule {evaluation['rolling_threshold']}")
    c.hline(s["rolling_max_drift_free"], "max in drift-free weeks")
    c.line(list(rtop["as_of"]), list(rtop["psi"]), "largest rolling PSI", label_end=False)
    figures.append(
        figure(
            "Input drift against the previous four weeks (post hoc)",
            "Each week against the four before it: an abrupt change stands out, slow ageing "
            "cancels out. Designed after seeing the replay; checked on drift-free weeks.",
            c,
            rtop[["as_of", "feature", "psi"]].rename(columns={"as_of": "as of", "psi": "PSI"}),
            "Rolling PSI: one clear peak in the first full week of March.",
        )
    )

    # 3. decision rates
    rates = pd.DataFrame(
        {
            "as of": list(m.index),
            "review / reference": (m["review_rate"] / reference_metrics["review_rate"]).values,
            "block / reference": (m["block_rate"] / reference_metrics["block_rate"]).values,
        }
    ).dropna()
    c = Chart(start, end, 2.0, _ratio)
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.hline(1.5, "alert x1.5")
    c.hline(1.25, "warning x1.25")
    c.line(list(rates["as of"]), list(rates["review / reference"]), "review rate", slot=0)
    c.line(list(rates["as of"]), list(rates["block / reference"]), "block rate", slot=1)
    for rule, col in (("review_rate", "review / reference"), ("block_rate", "block / reference")):
        hit = set(alerts.loc[(alerts["rule"] == rule) & (alerts["severity"] == "alert"), "as_of"])
        sel = rates[rates["as of"].isin(hit)]
        c.alerts(list(sel["as of"]), list(sel[col]), "alert fired", rule.replace("_", " "))
    figures.append(
        figure(
            "Decisions: review and block rates relative to the reference",
            f"Last week's rates divided by the reference rates "
            f"({_pct(reference_metrics['review_rate'])} reviewed, "
            f"{_pct(reference_metrics['block_rate'])} blocked).",
            c,
            rates.round(3),
            "Review and block rates rise above the reference from April.",
        )
    )

    # 4. delay-adjusted calibration
    oe = pd.DataFrame(
        {
            "as of": list(m.index),
            "observed / expected": m["recent_observed_to_expected"].values,
            "observed": m["recent_observed_reports"].values,
            "expected": m["recent_expected_reports"].values,
            "Poisson p": m["recent_poisson_p"].values,
        }
    ).dropna(subset=["observed / expected"])
    c = Chart(start, end, 2.0, _ratio, y_min=0.5)
    c.axes(3)
    c.vline(drift_start, drift_label)
    c.hline(1.25, "alert x1.25")
    c.hline(1.0, "as expected")
    c.line(
        list(oe["as of"]), list(oe["observed / expected"]), "observed / expected", label_end=False
    )
    hit = set(alerts.loc[alerts["rule"] == "label_calibration", "as_of"])
    sel = oe[oe["as of"].isin(hit)]
    c.alerts(list(sel["as of"]), list(sel["observed / expected"]), "alert fired")
    figures.append(
        figure(
            "Labels: frauds reported so far vs what the model expected by now",
            "Payments of the last 8 weeks. Expected = sum of the model's probabilities times the "
            "share of frauds usually reported at each payment's age.",
            c,
            oe.round(4),
            "Observed over expected reports climbs from April and then alerts.",
        )
    )

    # 5. mature-cohort recall
    rec = pd.DataFrame({"as of": list(m.index), "recall": m["mature_cohort_recall"].values})
    rec = rec.dropna()
    c = Chart(start, end, 1.0, _pct, y_min=0.6)
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.hline(reference_metrics["recall"], "reference")
    c.hline(reference_metrics["recall"] - evaluation["recall_drop"], "alert")
    c.line(list(rec["as of"]), list(rec["recall"]), "recall", label_end=False)
    hit = set(alerts.loc[alerts["rule"] == "mature_recall", "as_of"])
    sel = rec[rec["as of"].isin(hit)]
    c.alerts(list(sel["as of"]), list(sel["recall"]), "alert fired")
    figures.append(
        figure(
            "Labels: recall on the newest mature cohort",
            "Share of known frauds that were reviewed or blocked, on the most recent week with "
            f"{evaluation['maturity_share']:.0%} of its labels expected in "
            f"(about {evaluation['maturity_days']:.0f} days old). After the last production "
            "week matures, the line repeats that week.",
            c,
            rec.round(4),
            "Recall on mature cohorts drops below the alert line from July.",
        )
    )

    # 6-7. what really happened (evaluation only)
    truth = evaluation["truth"]
    c = Chart(start, end, _nice_max(truth["drift_scenario_frauds"].max()), lambda v: f"{v:.0f}")
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.bars(list(truth["week_start"]), list(truth["drift_scenario_frauds"]), "new-pattern frauds")
    figures.append(
        figure(
            "What really happened: frauds of the new pattern per week",
            "Simulator ground truth (wallet provisioning). Not available to the monitoring.",
            c,
            truth[["week_start", "true_frauds", "drift_scenario_frauds", "reported_frauds"]],
            "New-pattern frauds start in March and grow every month.",
        )
    )
    c = Chart(start, end, 1.0, _pct, y_min=0.6)
    c.axes(4)
    c.vline(drift_start, drift_label)
    c.hline(reference_metrics["recall"], "reference")
    c.line(list(truth["week_start"]), list(truth["recall_reported"]), "recall", label_end=False)
    figures.append(
        figure(
            "What really happened: recall per payment week, all labels known today",
            "Share of reported frauds reviewed or blocked. Known only months later.",
            c,
            truth[
                [
                    "week_start",
                    "recall_reported",
                    "recall_drift_scenario",
                    "mean_predicted",
                    "observed_rate",
                ]
            ].round(4),
            "True weekly recall drifts down after the new pattern appears.",
        )
    )

    timeline = evaluation["timeline"].rename(
        columns={
            "runs_alerting": "runs",
            "runs_before_drift": "runs before drift",
            "first_alert": "first alert",
            "first_after_drift": "first after drift",
            "days_after_drift": "days after drift",
        }
    )
    runs = tables["metrics"]["as_of"].nunique()
    body = (
        "<h1>Model monitoring: replay of simulated production</h1>"
        f"<p class='lead'>{runs} weekly runs from {min(m.index):%d %b %Y} to "
        f"{max(m.index):%d %b %Y}, each using only what was known that day. The simulator "
        f"injected a drift on {drift_start:%d %b %Y}: more online and wallet payments, and a new "
        "wallet fraud pattern. The monitoring did not know it.</p><ul>"
        f"<li>Decision rates alerted first, {s['first_rate_alert_days']} days after the drift, "
        f"but the same rules had alerted {_times(s['rate_alert_runs_before_drift'])} before it. "
        f"The label rules, silent before the drift, alerted after "
        f"{s['first_label_alert_days']} days and recommended retraining after "
        f"{s['first_retrain_days']}; recall on mature cohorts only after "
        f"{s['first_mature_recall_days']}.</li>"
        f"<li>Input drift with the conventional 0.10 threshold never caught it: the largest "
        f"weekly PSI of a stable input after the drift was "
        f"{s['stable_psi_max_after_drift']:.3f} (before it, "
        f"{s['stable_psi_max_before_drift']:.3f}).</li>"
        f"<li>A rolling reference (post hoc) would have caught that step after "
        f"{s['rolling_first_days']} days with no false alarm in {s['rolling_drift_free_weeks']} "
        "drift-free weeks. But that step is the change of legitimate behaviour, not the new "
        "fraud.</li></ul>"
        "<h2>Alert timeline (rules fixed in advance)</h2>"
        f"<div class='scroll'>{table_html(timeline)}</div>"
        f"<p>{_noisy_note(alerts, evaluation['noisy_features'])}</p>"
        "<h2>Signals over time</h2>" + "".join(figures) + "<h2>Post-hoc alternatives</h2>"
        "<p>Designed after seeing the replay, so hypotheses for the next version of the rules, "
        "not results. Drift-free weeks: the first half of 2025, never used to design them.</p>"
        f"<div class='scroll'>{table_html(evaluation['posthoc'])}</div>"
    )
    return page("Model monitoring", body)
