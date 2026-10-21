"""Markdown sanity report of a simulated dataset.

This is a check on the generator (did it produce what the configuration intended?), not an EDA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudplatform.markdown import md_table as _md_table
from fraudplatform.simulation.generator import SimulationResult
from fraudplatform.simulation.legit import LEGIT
from fraudplatform.simulation.randomness import iso_to_epoch


def _pct(x: float, digits: int = 3) -> str:
    return f"{100 * x:.{digits}f}%"


def build_summary(result: SimulationResult, dataset_hash: str) -> str:
    clean = result.clean_transactions
    labels = result.raw["fraud_labels"].copy()
    labels["is_fraud"] = labels["is_fraud"] == "true"
    labels["reported_s"] = iso_to_epoch(labels["reported_at"])
    lab = clean[["transaction_id", "ts_s", "is_fraud_true", "scenario", "amount_eur"]].merge(
        labels, on="transaction_id", how="left"
    )
    lab["labelled_fraud"] = lab["is_fraud"].eq(True)
    n = len(clean)
    n_true = int(clean["is_fraud_true"].sum())
    fraud = lab[lab["is_fraud_true"]]

    out = [
        f"# Simulated dataset v{result.simulator_version} ({result.scale})",
        "",
        f"- Seed: `{result.seed}`",
        f"- Dataset hash: `{dataset_hash}`",
        f"- Period: {pd.to_datetime(clean['ts_s'].min(), unit='s')} -> "
        f"{pd.to_datetime(clean['ts_s'].max(), unit='s')} (UTC)",
        "",
        "## Raw tables (including injected defects)",
        "",
        _md_table(
            pd.DataFrame({"table": list(result.raw), "rows": [len(t) for t in result.raw.values()]})
        ),
        "",
        "## Fraud prevalence",
        "",
        "| metric | value |",
        "|---|---|",
        f"| clean transactions | {n:,} |",
        f"| true fraud transactions | {n_true:,} ({_pct(n_true / n)}) |",
        f"| labelled as fraud (eventually) | {int(lab['labelled_fraud'].sum()):,} "
        f"({_pct(lab['labelled_fraud'].mean())}) |",
        f"| true fraud never labelled | {int((~fraud['labelled_fraud']).sum()):,} "
        f"({_pct((~fraud['labelled_fraud']).mean(), 1)} of true fraud) |",
        f"| legit labelled as fraud (friendly fraud) | "
        f"{int((lab['labelled_fraud'] & ~lab['is_fraud_true']).sum()):,} |",
        f"| disputes cleared as legit | {int(lab['is_fraud'].eq(False).sum()):,} |",
        "",
        "## Scenarios",
        "",
    ]

    gt = result.ground_truth
    by = fraud.groupby("scenario").agg(
        transactions=("transaction_id", "size"),
        median_amount_eur=("amount_eur", "median"),
        labelled=("labelled_fraud", "mean"),
    )
    by["incidents"] = gt[gt["is_fraud_true"]].groupby("scenario")["incident_id"].nunique()
    by["share_of_fraud"] = by["transactions"] / by["transactions"].sum()
    by = by.reset_index()[
        ["scenario", "incidents", "transactions", "share_of_fraud", "median_amount_eur", "labelled"]
    ]
    by["share_of_fraud"] = by["share_of_fraud"].map(lambda x: _pct(x, 1))
    by["labelled"] = by["labelled"].map(lambda x: _pct(x, 1))
    by["median_amount_eur"] = by["median_amount_eur"].round(2)
    out += [_md_table(by), ""]

    delay_days = (lab["reported_s"] - lab["ts_s"]) / 86_400
    d = delay_days[lab["labelled_fraud"]]
    out += [
        "## Label delay (labelled fraud)",
        "",
        "| median days | p90 days | max days | reported within 30 days |",
        "|---|---|---|---|",
        f"| {d.median():.1f} | {d.quantile(0.9):.1f} | {d.max():.1f} "
        f"| {_pct((d <= 30).mean(), 1)} |",
        "",
        "## Monthly fraud rate (by transaction month)",
        "",
    ]
    month = pd.to_datetime(lab["ts_s"], unit="s").dt.strftime("%Y-%m")
    monthly = lab.groupby(month).agg(
        transactions=("transaction_id", "size"),
        true_fraud_rate=("is_fraud_true", "mean"),
        labelled_fraud_rate=("labelled_fraud", "mean"),
    )
    monthly["true_fraud_rate"] = monthly["true_fraud_rate"].map(_pct)
    monthly["labelled_fraud_rate"] = monthly["labelled_fraud_rate"].map(_pct)
    out += [_md_table(monthly.reset_index(names="month")), ""]

    pm = (
        clean.assign(kind=np.where(clean["scenario"] == LEGIT, "legit", "fraud"))
        .groupby(["kind", "payment_method"])
        .size()
        .unstack(0, fill_value=0)
    )
    pm = (pm / pm.sum()).map(lambda x: _pct(x, 1)).reset_index()
    out += ["## Payment method mix", "", _md_table(pm), ""]

    defects = (
        result.dq_defects.groupby(["table_name", "defect_type"]).size().reset_index(name="rows")
    )
    out += ["## Injected data quality defects", "", _md_table(defects), ""]
    return "\n".join(out)
