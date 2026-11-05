"""Tiny helpers to render Markdown reports without extra dependencies."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def rows_table(columns: Sequence[str], rows: Sequence[Sequence]) -> str:
    return md_table(pd.DataFrame(list(rows), columns=list(columns)))
