"""Feature values on a hand-built scenario where every expected value is known in advance.

Timeline of customer HC1 / card HK1 (all UTC, 2025):
    HT0  01-10 10:00:00  20 EUR  chip_pin   grocery ES
    HT1  01-10 10:00:00  30 EUR  chip_pin   grocery ES     <- same second as HT0
    HT2  01-10 10:30:00   2 EUR  ecommerce  electronics US, device HD1  (fraud, reported 01-11)
    HT3  01-10 10:31:00   3 EUR  ecommerce  electronics US, device HD1
    HT4  01-12 10:00:00  25 EUR  chip_pin   grocery ES
Customer HC2 / card HK2:
    HT5  01-12 11:00:00  50 EUR  ecommerce  electronics US, device HD1 (same device as HC1)
"""

from __future__ import annotations

import pytest

from fraudplatform.config import load_feature_config
from fraudplatform.db import load_dataset
from fraudplatform.features.builder import build_features
from fraudplatform.features.catalog import FEATURE_NAMES, KEY_COLUMNS, META_COLUMNS
from fraudplatform.quality.runner import run_data_quality
from fraudplatform.simulation.writer import write_dataset

pytestmark = pytest.mark.integration

SCENARIO = """
INSERT INTO core.customers VALUES ('HC1', '1990-01-01', 'ES', '2020-01-01', 'standard'),
                                  ('HC2', '1985-01-01', 'ES', '2020-01-01', 'standard');
INSERT INTO core.cards VALUES ('HK1', 'HC1', 'debit', '2020-01-02'),
                              ('HK2', 'HC2', 'debit', '2020-01-02');
INSERT INTO core.merchants VALUES ('HM1', 'grocery', 'physical', 'ES', 'low', '2019-01-01'),
                                  ('HM2', 'electronics', 'online', 'US', 'high', '2019-01-01');
INSERT INTO core.devices VALUES ('HD1', 'desktop', 'linux', 'US', '2025-01-10 10:30:00+00');
INSERT INTO core.transactions VALUES
 ('HT0', 'HC1', 'HK1', 'HM1', NULL,  '2025-01-10 10:00:00+00', 20, 'EUR', 20, 'ES', 'chip_pin'),
 ('HT1', 'HC1', 'HK1', 'HM1', NULL,  '2025-01-10 10:00:00+00', 30, 'EUR', 30, 'ES', 'chip_pin'),
 ('HT2', 'HC1', 'HK1', 'HM2', 'HD1', '2025-01-10 10:30:00+00',  2, 'EUR',  2, 'US', 'ecommerce'),
 ('HT3', 'HC1', 'HK1', 'HM2', 'HD1', '2025-01-10 10:31:00+00',  3, 'EUR',  3, 'US', 'ecommerce'),
 ('HT4', 'HC1', 'HK1', 'HM1', NULL,  '2025-01-12 10:00:00+00', 25, 'EUR', 25, 'ES', 'chip_pin'),
 ('HT5', 'HC2', 'HK2', 'HM2', 'HD1', '2025-01-12 11:00:00+00', 50, 'EUR', 50, 'US', 'ecommerce');
INSERT INTO core.fraud_labels VALUES ('HT2', true, '2025-01-11 12:00:00+00', 'chargeback');
"""


@pytest.fixture(scope="module")
def features(conn, tiny, tmp_path_factory):
    load_dataset(conn, write_dataset(tiny, tmp_path_factory.mktemp("dataset")))
    run_data_quality(conn)
    conn.execute(SCENARIO)
    conn.commit()
    build_features(conn, load_feature_config())
    cur = conn.execute(
        "SELECT * FROM features.transaction_features WHERE transaction_id LIKE 'HT%' "
        "ORDER BY transaction_id"
    )
    columns = [d.name for d in cur.description]
    return {row[0]: dict(zip(columns, row, strict=True)) for row in cur.fetchall()}


def test_columns_follow_the_catalog(conn, features):
    cur = conn.execute("SELECT * FROM features.transaction_features LIMIT 0")
    assert tuple(d.name for d in cur.description) == KEY_COLUMNS + FEATURE_NAMES + META_COLUMNS


def test_one_row_per_core_transaction(conn, features):
    n_core, n_features, n_leaky = conn.execute(
        "SELECT (SELECT COUNT(*) FROM core.transactions), "
        "(SELECT COUNT(*) FROM features.transaction_features), "
        "(SELECT COUNT(*) FROM features.transaction_features_leaky)"
    ).fetchone()
    assert n_core == n_features == n_leaky


def test_same_second_transactions_are_not_history(features):
    for tid in ("HT0", "HT1"):
        f = features[tid]
        assert f["card_txn_count_1h"] == 0
        assert f["card_seconds_since_prev"] is None
        assert f["merchant_new_for_customer"] is True
        assert f["country_changed"] is None


def test_card_velocity(features):
    ht2, ht3 = features["HT2"], features["HT3"]
    assert ht2["card_txn_count_1h"] == 2 and ht2["card_txn_count_5m"] == 0
    assert ht2["card_amount_sum_1h"] == pytest.approx(50)
    assert ht2["card_micro_count_1h"] == 0  # HT2 itself is micro, but it is not history
    assert ht2["card_seconds_since_prev"] == pytest.approx(1800)
    assert ht3["card_txn_count_5m"] == 1 and ht3["card_txn_count_1h"] == 3
    assert ht3["card_micro_count_1h"] == 1
    assert ht3["card_unique_merchants_24h"] == 2 and ht3["card_unique_countries_24h"] == 2
    ht4 = features["HT4"]
    assert ht4["card_txn_count_24h"] == 0 and ht4["card_txn_count_7d"] == 4


def test_novelty_and_device(features):
    ht2, ht3, ht5 = features["HT2"], features["HT3"], features["HT5"]
    assert ht2["merchant_new_for_customer"] is True and ht2["device_new_for_customer"] is True
    assert ht3["merchant_new_for_customer"] is False and ht3["device_new_for_customer"] is False
    assert ht3["device_txn_count_1h"] == 1
    assert features["HT4"]["device_new_for_customer"] is None  # card present: no device
    assert ht5["device_new_for_customer"] is True
    assert ht5["device_unique_customers_30d"] == 1  # HC1 before, HC2 itself not counted
    assert ht5["device_unique_cards_24h"] == 0  # HD1 last used 2 days earlier


def test_geography(features):
    ht2, ht3, ht4 = features["HT2"], features["HT3"], features["HT4"]
    assert ht2["cross_border"] is True and ht2["country_changed"] is True
    assert ht2["distance_from_prev_km"] > 7000  # Spain -> USA centroids
    assert ht2["implied_speed_kmh"] > 900 and ht2["impossible_travel"] is True
    assert ht3["country_changed"] is False and ht3["distance_from_prev_km"] == 0
    assert ht4["country_changed"] is True


def test_labels_are_only_used_once_reported(features):
    # HT2 is fraud, but only reported on 01-11: unknown when HT3 happens on 01-10.
    assert features["HT3"]["card_prior_fraud_reports"] == 0
    assert features["HT4"]["card_prior_fraud_reports"] == 1
    assert features["HT5"]["device_prior_fraud_reports"] == 1
    assert features["HT5"]["merchant_fraud_reports_90d"] == 1
    assert features["HT4"]["merchant_fraud_reports_90d"] == 0  # the fraud was at HM2


def test_customer_amount_profile(features):
    ht4 = features["HT4"]
    assert ht4["cust_txn_count_90d"] == 4
    assert ht4["amount_to_cust_median_90d"] == pytest.approx(25 / 11.5)  # median(20,30,2,3)
    assert features["HT0"]["amount_to_cust_median_90d"] is None  # no history


def test_no_negative_values(conn, features):
    negatives = conn.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE card_txn_count_5m < 0 OR card_txn_count_1h < 0
                              OR card_txn_count_24h < 0 OR card_txn_count_7d < 0
                              OR card_amount_sum_24h < 0 OR cust_txn_count_90d < 0
                              OR card_unique_merchants_24h < 0 OR device_unique_cards_24h < 0),
          COUNT(*) FILTER (WHERE card_seconds_since_prev <= 0 OR cust_seconds_since_prev <= 0),
          COUNT(*) FILTER (WHERE account_age_days < 0 OR device_age_days < 0 OR customer_age < 18),
          COUNT(*) FILTER (WHERE distance_from_prev_km < 0
                              OR merchant_fraud_rate_90d_smoothed NOT BETWEEN 0 AND 1)
        FROM features.transaction_features
        """
    ).fetchone()
    assert negatives == (0, 0, 0, 0)
