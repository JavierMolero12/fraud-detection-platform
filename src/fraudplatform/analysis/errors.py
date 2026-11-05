"""Error and segment analysis of the deployed policy, on VALIDATION.

The test period has been used once (decision 022). Studying its errors in order to change
the model would turn it into a second validation set, so all analysis here uses validation.

Errors are defined at the DECISION level (what the business experiences), not at a threshold:
- false positive = legitimate (by label) payment that was BLOCKED;
- false negative = fraudulent (by label) payment that was APPROVED.

Because the data is simulated, the true scenario of every transaction is known
(`sim.ground_truth`). It is used ONLY here, to explain errors, never as a model input. It also
shows how much of the "error" is label noise: unreported fraud counted as legit, and legit
payments disputed as fraud (friendly fraud).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import average_precision_score

from fraudplatform.decision.costs import APPROVE, BLOCK, REVIEW

EXTRA_SQL = """
SELECT t.transaction_id,
       cu.country_code                         AS home_country,
       t.country_code                          AS txn_country,
       g.is_fraud_true,
       g.scenario
FROM core.transactions t
JOIN core.customers cu USING (customer_id)
JOIN sim.ground_truth g USING (transaction_id)
WHERE t.ts >= %(start)s AND t.ts < %(end)s
"""


def load_context(conn: psycopg.Connection, start: date, end: date) -> pd.DataFrame:
    """Attributes that are not model features (home country) plus the simulator truth."""
    cur = conn.execute(EXTRA_SQL, {"start": start, "end": end})
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Columns describing each decision against the label and against the truth."""
    out = df.copy()
    y = out["is_fraud"].to_numpy(dtype=bool)
    action = out["action"].to_numpy()
    out["outcome"] = np.select(
        [
            y & (action == BLOCK),
            y & (action == REVIEW),
            y & (action == APPROVE),
            ~y & (action == BLOCK),
            ~y & (action == REVIEW),
        ],
        [
            "fraud blocked",
            "fraud reviewed",
            "fraud approved (FN)",
            "legit blocked (FP)",
            "legit reviewed",
        ],
        default="legit approved",
    )
    truth = out["is_fraud_true"].to_numpy(dtype=bool)
    out["label_noise"] = np.select(
        [truth & ~y, ~truth & y], ["unreported fraud", "friendly fraud"], default="none"
    )
    return out


def error_profile(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """For each value of `column`: is it over-represented among the errors?

    over_representation = share among errors / share among the reference population
    (legit payments for false positives, fraud for false negatives). 2.0 = twice as common
    among errors as it should be if errors were spread evenly.
    """
    legit = df[~df["is_fraud"]]
    fraud = df[df["is_fraud"]]
    fp = legit[legit["outcome"] == "legit blocked (FP)"]
    fn = fraud[fraud["outcome"] == "fraud approved (FN)"]
    table = pd.DataFrame(
        {
            "legit": legit[column].value_counts(),
            "false_positives": fp[column].value_counts(),
            "fraud": fraud[column].value_counts(),
            "false_negatives": fn[column].value_counts(),
        }
    ).fillna(0)
    table["fp_rate"] = table["false_positives"] / table["legit"].replace(0, np.nan)
    table["fn_rate"] = table["false_negatives"] / table["fraud"].replace(0, np.nan)
    table["fp_over_representation"] = (table["false_positives"] / max(len(fp), 1)) / (
        table["legit"] / len(legit)
    )
    table["fn_over_representation"] = (table["false_negatives"] / max(len(fn), 1)) / (
        table["fraud"] / max(len(fraud), 1)
    ).replace(0, np.nan)
    return table.sort_values("legit", ascending=False)


def segment_metrics(
    df: pd.DataFrame, column: str, reviewer_recall: float, min_frauds: int = 30
) -> pd.DataFrame:
    """Model and policy quality inside each segment."""
    rows = []
    for value, g in df.groupby(column, observed=True):
        y = g["is_fraud"].to_numpy(dtype=bool)
        a = g["action"].to_numpy()
        amount = g["amount_eur"].to_numpy()
        blocked, reviewed = a == BLOCK, a == REVIEW
        fraud_eur = amount[y].sum()
        rows.append(
            {
                column: value,
                "transactions": len(g),
                "frauds": int(y.sum()),
                "fraud_rate": y.mean(),
                "pr_auc": average_precision_score(y, g["p"]) if y.sum() >= min_frauds else np.nan,
                "fraud_stopped": ((blocked | reviewed) & y).sum() / max(y.sum(), 1),
                "fraud_eur_stopped": (
                    amount[blocked & y].sum() + reviewer_recall * amount[reviewed & y].sum()
                )
                / max(fraud_eur, 1e-9),
                "block_precision": (blocked & y).sum() / max(blocked.sum(), 1),
                "false_decline_rate": (blocked & ~y).sum() / max((~y).sum(), 1),
            }
        )
    return pd.DataFrame(rows).set_index(column)


def scenario_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """What happened to each TRUE scenario (simulator truth), split by whether it was labelled."""
    frame = df.assign(
        truth=np.where(df["is_fraud_true"], df["scenario"], "legit"),
        labelled_fraud=df["is_fraud"],
    )
    table = frame.groupby(["truth", "labelled_fraud"]).agg(
        transactions=("transaction_id", "size"),
        blocked=("action", lambda a: (a == BLOCK).mean()),
        reviewed=("action", lambda a: (a == REVIEW).mean()),
        approved=("action", lambda a: (a == APPROVE).mean()),
        median_p=("p", "median"),
        median_amount=("amount_eur", "median"),
    )
    return table.reset_index()
