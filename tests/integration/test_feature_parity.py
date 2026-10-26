"""Parity: the API's single-transaction features equal the offline batch features.

Two implementations of the same 50 definitions exist (decision 017): window functions for the
batch build, lookups for the API. For a sample of stored transactions, recompute every feature
through the online path, as if the transaction were arriving now, and compare the MODEL INPUT
exactly: equal floats, not "close". A 1e-14 difference once flipped a tree split (decision 024).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fraudplatform.config import load_feature_config
from fraudplatform.db import load_dataset
from fraudplatform.eda.data import load_features
from fraudplatform.features.builder import build_features
from fraudplatform.features.catalog import CATEGORICAL, FEATURE_NAMES
from fraudplatform.models.pipelines import prepare
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.serving.online_features import (
    IncomingTransaction,
    OnlineFeatureBuilder,
    UnknownEntity,
    amount_in_eur,
)
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def built(conn, small, tmp_path_factory):
    load_dataset(conn, write_dataset(small, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    cfg = load_feature_config()
    build_features(conn, cfg)
    return OnlineFeatureBuilder(cfg)


def _incoming(row) -> IncomingTransaction:
    return IncomingTransaction(
        customer_id=row["customer_id"],
        card_id=row["card_id"],
        merchant_id=row["merchant_id"],
        ts=row["ts"],
        amount=float(row["amount"]),
        currency=row["currency_code"],
        country=row["country_code"],
        payment_method=row["payment_method"],
        device_id=row["device_id"],
    )


def _differences(online: pd.DataFrame, offline: pd.DataFrame) -> list[tuple]:
    """(row, feature, online, offline) for every value that is not exactly equal."""
    out = []
    for name in FEATURE_NAMES:
        a, b = online[name], offline[name]
        if name in CATEGORICAL:
            differ = a.astype(str) != b.astype(str)
        else:
            a, b = a.astype(float), b.astype(float)
            differ = ~((a == b) | (a.isna() & b.isna()))
        out += [(i, name, a[i], b[i]) for i in differ[differ].index]
    return out


def test_online_model_input_is_identical_to_offline(conn, built):
    # a stratified sample: every payment method, with and without history, some fraud
    ids = [
        r[0]
        for r in conn.execute(
            """
            SELECT transaction_id FROM (
                SELECT transaction_id,
                       ROW_NUMBER() OVER (PARTITION BY payment_method ORDER BY md5(transaction_id))
                           AS rn
                FROM core.transactions
            ) s WHERE rn <= 100
            UNION
            (SELECT transaction_id FROM core.fraud_labels WHERE is_fraud
             ORDER BY md5(transaction_id) LIMIT 100)
            """
        ).fetchall()
    ]
    assert len(ids) >= 400
    cur = conn.execute(
        "SELECT * FROM core.transactions WHERE transaction_id = ANY(%s) ORDER BY transaction_id",
        (ids,),
    )
    rows = [dict(zip([d.name for d in cur.description], r, strict=True)) for r in cur.fetchall()]
    online = pd.concat(
        [prepare(built.frame(conn, _incoming(row))) for row in rows], ignore_index=True
    )
    offline = prepare(load_features(conn, date(2000, 1, 1), date(2100, 1, 1)))
    offline = offline.set_index("transaction_id").loc[[r["transaction_id"] for r in rows]]
    offline = offline.reset_index()[list(FEATURE_NAMES)]
    assert not _differences(online, offline), _differences(online, offline)[:10]


def test_fx_conversion_matches_the_data_quality_stage(conn, built):
    rows = conn.execute(
        "SELECT customer_id, card_id, merchant_id, ts, amount, currency_code, country_code, "
        "payment_method, device_id, amount_eur FROM core.transactions "
        "WHERE currency_code <> 'EUR' LIMIT 200"
    ).fetchall()
    assert rows
    for r in rows:
        txn = IncomingTransaction(r[0], r[1], r[2], r[3], float(r[4]), r[5], r[6], r[7], r[8])
        assert amount_in_eur(conn, txn) == pytest.approx(float(r[9]), abs=0.005)


def test_unknown_entities_are_rejected(conn, built):
    row = conn.execute(
        "SELECT customer_id, card_id, merchant_id, ts, amount, currency_code, country_code, "
        "payment_method FROM core.transactions LIMIT 1"
    ).fetchone()
    good = IncomingTransaction(*row[:4], float(row[4]), row[5], row[6], row[7])
    for bad in (
        good.__class__(**{**good.__dict__, "customer_id": "C_DOES_NOT_EXIST"}),
        good.__class__(**{**good.__dict__, "merchant_id": "M_DOES_NOT_EXIST"}),
        good.__class__(**{**good.__dict__, "card_id": "K_DOES_NOT_EXIST"}),
    ):
        with pytest.raises(UnknownEntity):
            built.frame(conn, bad)
