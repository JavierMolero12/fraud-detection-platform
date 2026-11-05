"""Cost model, policies and calibration. Expected values are computed by hand."""

from __future__ import annotations

import numpy as np
import pytest

from fraudplatform.config import load_cost_config
from fraudplatform.decision.calibration import (
    apply_calibrator,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from fraudplatform.decision.costs import (
    APPROVE,
    BLOCK,
    REVIEW,
    CostModel,
    bayes_actions,
    evaluate_actions,
    threshold_actions,
    threshold_grid,
)

DESIGN = CostModel(15.0, 4.0, 0.90, 1.0, 0.02, 10.0)  # the Phase 1 design assumptions


@pytest.mark.parametrize(
    ("amount", "approve_review", "review_block", "approve_block"),
    [
        (20, 0.154, 0.419, 0.229),  # table in docs/phase1_design.md, section 10.3
        (200, 0.026, 0.261, 0.061),
    ],
)
def test_breakeven_matches_the_design_document(amount, approve_review, review_block, approve_block):
    b = DESIGN.breakeven(amount)
    assert b["approve_review"] == pytest.approx(approve_review, abs=5e-4)
    assert b["review_block"] == pytest.approx(review_block, abs=5e-4)
    assert b["approve_block"] == pytest.approx(approve_block, abs=5e-4)


def test_config_matches_the_design_assumptions():
    assert CostModel.from_config(load_cost_config()) == DESIGN


def test_realized_costs_by_hand():
    c = DESIGN.realized(y=[1, 0], amount=[200, 200])
    # fraud: approve 215; review 4 + 0.1 * 215 = 25.5; block 0
    np.testing.assert_allclose(c[0], [215, 25.5, 0])
    # legit: approve 0; review 4 + 1 = 5; block 0.02 * 200 + 10 = 14
    np.testing.assert_allclose(c[1], [0, 5, 14])


def test_bayes_policy_picks_the_cheapest_expected_action():
    p = np.array([0.01, 0.10, 0.50])  # for 200 EUR: approve < 0.026 <= review < 0.261 <= block
    actions = bayes_actions(p, [200] * 3, ["d"] * 3, DESIGN, capacity=10)
    assert list(actions) == [APPROVE, REVIEW, BLOCK]


def test_capacity_is_never_exceeded_and_goes_to_the_most_valuable_reviews():
    p = np.full(6, 0.10)  # all would be reviewed without a limit
    amount = np.array([50, 1000, 200, 20, 500, 100.0])
    actions = bayes_actions(p, amount, ["d"] * 6, DESIGN, capacity=2)
    assert (actions == REVIEW).sum() == 2
    assert set(np.flatnonzero(actions == REVIEW)) == {1, 4}  # the two largest amounts


def test_threshold_policy_respects_daily_capacity_per_day():
    score = np.array([0.9, 0.5, 0.4, 0.3, 0.5, 0.4, 0.01])
    day = np.array(["a", "a", "a", "a", "b", "b", "b"])
    actions = threshold_actions(score, day, t_review=0.2, t_block=0.8, capacity=2)
    assert actions[0] == BLOCK
    assert list(actions[1:4]) == [REVIEW, REVIEW, APPROVE]  # day a: overflow approved
    assert list(actions[4:]) == [REVIEW, REVIEW, APPROVE]


def test_threshold_grid_finds_the_cheapest_pair():
    rng = np.random.default_rng(0)
    y = rng.random(3000) < 0.02
    score = np.clip(np.where(y, 0.6, 0.05) + rng.normal(0, 0.1, 3000), 0, 1)
    amount = rng.exponential(100, 3000)
    realized = DESIGN.realized(y, amount)
    grid = threshold_grid(
        score, realized, np.repeat(np.arange(30), 100), 1000, [0.1, 0.2, 0.3], [0.4, 0.7, 1.01]
    )
    best = grid.loc[grid["total_cost_eur"].idxmin()]
    approve_all = realized[:, APPROVE].sum()
    assert best["total_cost_eur"] < approve_all
    assert len(grid) == 9


def test_evaluate_actions_counts():
    y = np.array([True, False, True, False])
    actions = np.array([BLOCK, BLOCK, REVIEW, APPROVE])
    out = evaluate_actions(
        actions,
        y,
        [100, 50, 10, 5.0],
        DESIGN.realized(y, [100, 50, 10, 5.0]),
        ["d"] * 4,
        reviewer_recall=0.9,
    )
    assert (out["fraud_blocked"], out["legit_blocked"], out["fraud_reviewed"]) == (1, 1, 1)
    assert out["total_cost_eur"] == pytest.approx(0 + 11 + (4 + 0.1 * 25) + 0)
    assert out["fraud_eur_stopped_share"] == pytest.approx((100 + 0.9 * 10) / 110)


# ---------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------
def _miscalibrated(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.beta(0.3, 20, n)  # rare positives
    y = rng.random(n) < true_p
    reported = np.clip(true_p * 4, 0, 0.999)  # overconfident by 4x, like class weights
    return reported, y


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_fixes_an_overconfident_score(method):
    p_fit, y_fit = _miscalibrated(seed=1)
    p_new, y_new = _miscalibrated(seed=2)
    cal = fit_calibrator(method, p_fit, y_fit)
    fixed = apply_calibrator(method, cal, p_new)
    assert expected_calibration_error(fixed, y_new) < expected_calibration_error(p_new, y_new) / 2
    assert abs(fixed.mean() - y_new.mean()) < 0.003


def test_none_calibration_is_identity_and_tables_have_equal_counts():
    p, y = _miscalibrated()
    np.testing.assert_array_equal(apply_calibrator("none", fit_calibrator("none", p, y), p), p)
    table = reliability_table(p, y, bins=10)
    assert table["n"].nunique() == 1 and len(table) == 10


@pytest.mark.parametrize("method", ["none", "platt", "isotonic"])
def test_calibrator_json_round_trip_gives_identical_probabilities(method):
    import json

    from fraudplatform.decision.calibration import apply_calibration_dict, calibrator_to_dict

    p_fit, y_fit = _miscalibrated(seed=3)
    p_new, _ = _miscalibrated(seed=4)
    cal = fit_calibrator(method, p_fit, y_fit)
    spec = json.loads(json.dumps(calibrator_to_dict(method, cal)))
    np.testing.assert_allclose(
        apply_calibration_dict(spec, p_new), apply_calibrator(method, cal, p_new), atol=1e-12
    )


def test_sensitivity_regret_is_positive_when_an_assumption_is_wrong():
    """Regression: the deployed P2 policy used to be re-decided with the CHANGED costs, which
    is re-optimising it, so every regret came out as exactly 0."""
    from fraudplatform.decision.optimise import sensitivity_analysis

    rng = np.random.default_rng(5)
    n = 4000
    y = rng.random(n) < 0.03
    p = np.clip(np.where(y, 0.4, 0.02) + rng.normal(0, 0.08, n), 0.0005, 0.999)
    amount = rng.exponential(150, n)
    day = np.repeat(np.arange(40), 100)
    table = sensitivity_analysis(
        {"type": "expected_cost"}, p, y, amount, day, DESIGN, 50, [0.5, 3.0]
    )
    assert (table["regret_eur"] >= -1e-6).all()
    assert (table["regret_eur"] > 0).any()


def test_test_evaluation_refuses_to_run_twice(tmp_path, monkeypatch):
    from fraudplatform.decision import test_evaluation

    marker = tmp_path / "test_evaluation.json"
    marker.write_text("{}")
    monkeypatch.setattr(test_evaluation, "MARKER", marker)
    with pytest.raises(test_evaluation.TestAlreadyEvaluated):
        test_evaluation.evaluate_on_test(None, None, None, tmp_path)
