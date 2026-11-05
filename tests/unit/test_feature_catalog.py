from __future__ import annotations

import re

from fraudplatform.config import load_feature_config
from fraudplatform.features.builder import feature_files
from fraudplatform.features.catalog import (
    CATEGORICAL,
    FEATURE_NAMES,
    FEATURES,
    LEAKY_FEATURES,
    render_markdown,
)


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_leaky_variants_map_to_real_features():
    assert set(LEAKY_FEATURES.values()) <= set(FEATURE_NAMES)


def test_no_identifier_is_a_feature():
    assert not [f for f in FEATURE_NAMES if f.endswith("_id")]


def test_country_is_not_a_raw_feature():
    """EDA: the country effect is a confounder of cross-border; encode mismatch instead."""
    assert "txn_country" not in FEATURE_NAMES and "home_country" not in FEATURE_NAMES


def test_every_feature_is_built_by_the_sql():
    sql = "\n".join(p.read_text(encoding="utf-8") for p in feature_files())
    missing = [f for f in FEATURE_NAMES if not re.search(rf"\b{f}\b", sql)]
    assert missing == []


def test_feature_sql_never_reads_simulator_truth():
    for path in feature_files():
        code = re.sub(r"--[^\n]*", "", path.read_text(encoding="utf-8"))
        assert not re.search(r"\bsim\.", code), path.name


def test_catalog_renders_every_feature():
    text = render_markdown()
    assert all(f"`{f.name}`" in text for f in FEATURES)
    assert set(CATEGORICAL) <= set(FEATURE_NAMES)


def test_feature_config_loads():
    cfg = load_feature_config()
    assert cfg.night_hours[0] <= cfg.night_hours[1]
    assert set(cfg.sql_settings()) >= {"features.version", "features.micro_amount_eur"}
