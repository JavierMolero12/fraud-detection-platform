"""Data quality defects injected into the raw extract.

Real source systems produce duplicates, broken references and malformed values. We inject them
deliberately, and log every injected defect, so that the Data Quality stage (Phase 3) can be
*measured*: for each rule we know exactly which rows it should catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudplatform.config import DefectParams
from fraudplatform.simulation.randomness import DAY, iso_to_epoch, to_iso
from fraudplatform.simulation.reference import COUNTRY_CODES

INVALID_CURRENCIES = np.array(["XXX", "EURO", "eur", "", "E$"])


class DefectInjector:
    def __init__(self, params: DefectParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self.log: list[pd.DataFrame] = []

    def _record(self, table: str, keys, defect: str, detail) -> None:
        keys = np.asarray(keys)
        self.log.append(
            pd.DataFrame(
                {
                    "table_name": table,
                    "record_key": keys,
                    "defect_type": defect,
                    "detail": np.broadcast_to(np.asarray(detail, dtype=object), keys.shape),
                }
            )
        )

    def _take(self, available: np.ndarray, rate: float, n_ref: int, mask=None) -> np.ndarray:
        """Pick rows not already used by another defect, so each row has at most one defect."""
        candidates = available if mask is None else available & mask
        pool = np.flatnonzero(candidates)
        k = min(round(rate * n_ref), len(pool))
        rows = self.rng.choice(pool, size=k, replace=False) if k else np.array([], dtype=np.int64)
        available[rows] = False
        return np.sort(rows)

    def transactions(self, txns: pd.DataFrame, card_issued_s: pd.Series, cards: pd.DataFrame):
        p = self.params
        rng = self.rng
        df = txns.copy()
        n = len(df)
        free = np.ones(n, dtype=bool)
        tid = df["transaction_id"].to_numpy()
        card_present = df["payment_method"].isin(["chip_pin", "contactless"]).to_numpy()
        ecommerce = (df["payment_method"] == "ecommerce").to_numpy()

        rows = self._take(free, p.invalid_amount, n)
        variant = rng.integers(0, 3, len(rows))
        amount = df["amount"].to_numpy()[rows]
        new = np.where(variant == 0, "0.00", np.where(variant == 1, np.char.add("-", amount), ""))
        df.iloc[rows, df.columns.get_loc("amount")] = new
        self._record("transactions", tid[rows], "invalid_amount", np.where(new == "", "null", new))

        rows = self._take(free, p.invalid_currency, n)
        new = INVALID_CURRENCIES[rng.integers(0, len(INVALID_CURRENCIES), len(rows))]
        df.iloc[rows, df.columns.get_loc("currency")] = new
        self._record("transactions", tid[rows], "invalid_currency", new)

        rows = self._take(free, p.invalid_timestamp, n)
        variant = rng.integers(0, 3, len(rows))
        issued = card_issued_s.loc[df["card_id"].to_numpy()[rows]].to_numpy()
        future = to_iso(
            np.datetime64("2031-01-01", "s").astype(np.int64)
            + rng.integers(0, 365 * DAY, len(rows))
        )
        before_card = to_iso(issued - rng.integers(DAY, 300 * DAY, len(rows)))
        new = np.where(
            variant == 0, future, np.where(variant == 1, before_card, "2025-13-45T25:61:00Z")
        )
        df.iloc[rows, df.columns.get_loc("timestamp")] = new
        kind = np.array(["future", "before_card_issued", "unparseable"])[variant]
        self._record("transactions", tid[rows], "invalid_timestamp", kind)

        rows = self._take(free, p.missing_customer_id, n)
        df.iloc[rows, df.columns.get_loc("customer_id")] = ""
        self._record("transactions", tid[rows], "missing_customer_id", "null")

        rows = self._take(free, p.orphan_reference, n)
        variant = rng.integers(0, 3, len(rows))
        for v, col, kind in (
            (0, "merchant_id", "unknown_merchant"),
            (1, "customer_id", "unknown_customer"),
            (2, "card_id", "card_of_other_customer"),
        ):
            r = rows[variant == v]
            if col == "card_id":
                other = cards["card_id"].to_numpy()[rng.integers(0, len(cards), len(r))]
                owner = cards.set_index("card_id")["customer_id"].loc[other].to_numpy()
                ok = owner != df["customer_id"].to_numpy()[r]
                r, new = r[ok], other[ok]
            else:
                prefix = "M" if col == "merchant_id" else "C"
                new = np.array([f"{prefix}9{x:07d}" for x in rng.integers(0, 10**7, len(r))])
            df.iloc[r, df.columns.get_loc(col)] = new
            self._record("transactions", tid[r], "orphan_reference", kind)

        rows = self._take(free, p.country_mismatch, n, mask=card_present)
        current = df["country"].to_numpy()[rows]
        new = COUNTRY_CODES[rng.integers(0, len(COUNTRY_CODES), len(rows))]
        clash = new == current
        new[clash] = np.where(current[clash] == "ES", "FR", "ES")
        df.iloc[rows, df.columns.get_loc("country")] = new
        self._record("transactions", tid[rows], "country_mismatch", new)

        rows = self._take(free, p.sentinel_device, n, mask=ecommerce)
        new = np.where(rng.random(len(rows)) < 0.7, "unknown", "00000000")
        df.iloc[rows, df.columns.get_loc("device_id")] = new
        self._record("transactions", tid[rows], "sentinel_device", new)

        # Duplicates are appended right after their original row (retries arrive together).
        order = np.arange(n, dtype=float)
        exact = self._take(free, p.exact_duplicate, n)
        self._record("transactions", tid[exact], "exact_duplicate", "identical row")
        conflict = self._take(free, p.duplicate_id_conflict, n)
        self._record("transactions", tid[conflict], "duplicate_id_conflict", "same id, new content")
        dup_conflict = df.iloc[conflict].copy()
        amounts = dup_conflict["amount"].astype(float).to_numpy()
        dup_conflict["amount"] = np.char.mod("%.2f", amounts * rng.uniform(0.5, 1.5, len(conflict)))
        ts = iso_to_epoch(dup_conflict["timestamp"])
        dup_conflict["timestamp"] = to_iso(ts + rng.integers(1, 600, len(conflict)))

        out = pd.concat([df, df.iloc[exact], dup_conflict], ignore_index=True)
        sort_key = np.concatenate([order, exact + 0.5, conflict + 0.5])
        return out.iloc[np.argsort(sort_key, kind="stable")].reset_index(drop=True)

    def customers(self, customers: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = customers.copy()
        n = len(df)
        free = np.ones(n, dtype=bool)
        rows = self._take(free, p.customer_missing_country, n)
        df.iloc[rows, df.columns.get_loc("country")] = ""
        self._record("customers", df["customer_id"].to_numpy()[rows], "missing_country", "null")
        dup = self._take(free, p.customer_duplicate, n)
        self._record(
            "customers", df["customer_id"].to_numpy()[dup], "exact_duplicate", "identical row"
        )
        return pd.concat([df, df.iloc[dup]], ignore_index=True)

    def cards(self, cards: pd.DataFrame) -> pd.DataFrame:
        k = round(self.params.card_orphan_customer * len(cards))
        if k == 0:
            return cards
        rng = self.rng
        orphans = cards.iloc[rng.choice(len(cards), size=k, replace=False)].copy()
        orphans["card_id"] = [f"K9{x:08d}" for x in rng.choice(10**8, size=k, replace=False)]
        orphans["customer_id"] = [f"C9{x:07d}" for x in rng.integers(0, 10**7, k)]
        self._record("cards", orphans["card_id"].to_numpy(), "orphan_reference", "unknown_customer")
        return pd.concat([cards, orphans], ignore_index=True)

    def defect_log(self) -> pd.DataFrame:
        columns = ["table_name", "record_key", "defect_type", "detail"]
        return (
            pd.concat(self.log, ignore_index=True)[columns]
            if self.log
            else pd.DataFrame(columns=columns)
        )
