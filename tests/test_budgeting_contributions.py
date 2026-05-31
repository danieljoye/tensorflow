"""Tests for the risk-contribution decomposition (BUILD_PLAN §2)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.budgeting import (
    ConditionalBudget,
    align_budget,
    budget_from_weights,
    decompose_risk,
    equal_risk_budget,
    group_risk_budget,
    marginal_risk_contributions,
    percentage_risk_contributions,
    portfolio_volatility,
    total_risk_contributions,
)
from riskbudget.core.errors import ConfigurationError
from riskbudget.core.types import RiskBudget

COV = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]])


def test_trc_sums_to_volatility() -> None:
    w = np.array([0.5, 0.3, 0.2])
    trc = total_risk_contributions(w, COV)
    vol = portfolio_volatility(w, COV)
    assert np.isclose(trc.sum(), vol)


def test_percentage_contributions_sum_to_one() -> None:
    w = np.array([0.4, 0.4, 0.2])
    pcr = percentage_risk_contributions(w, COV)
    assert np.isclose(pcr.sum(), 1.0)


def test_marginal_matches_definition() -> None:
    w = np.array([0.5, 0.3, 0.2])
    vol = portfolio_volatility(w, COV)
    mrc = marginal_risk_contributions(w, COV)
    assert np.allclose(mrc, (COV @ w) / vol)


def test_decompose_risk_verifies_euler_identity() -> None:
    w = np.array([0.3, 0.3, 0.4])
    decomp = decompose_risk(w, COV)
    assert decomp.verify()
    assert np.isclose(decomp.total.sum(), decomp.volatility)
    assert np.isclose(decomp.percentage.sum(), 1.0)


def test_zero_vol_returns_zeros() -> None:
    zero_cov = np.zeros((3, 3))
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(marginal_risk_contributions(w, zero_cov), 0.0)
    assert np.allclose(percentage_risk_contributions(w, zero_cov), 0.0)


def test_equal_and_from_weights_helpers() -> None:
    erc = equal_risk_budget(["A", "B", "C"])
    assert np.allclose(erc.as_array(["A", "B", "C"]), 1 / 3)
    b = budget_from_weights({"A": 3.0, "B": 1.0})
    assert np.isclose(b.budgets["A"], 0.75)


def test_group_risk_budget_equal_split() -> None:
    b = group_risk_budget({"eq": 0.6, "bond": 0.4}, {"A": "eq", "B": "eq", "C": "bond"})
    assert np.isclose(b.budgets["A"], 0.3)
    assert np.isclose(b.budgets["B"], 0.3)
    assert np.isclose(b.budgets["C"], 0.4)


def test_group_risk_budget_within_weights() -> None:
    b = group_risk_budget({"eq": 1.0}, {"A": "eq", "B": "eq"}, within_group={"A": 3.0, "B": 1.0})
    assert np.isclose(b.budgets["A"], 0.75)


def test_group_risk_budget_rejects_unknown_group() -> None:
    with pytest.raises(ConfigurationError):
        group_risk_budget({"eq": 1.0}, {"A": "missing"})


def test_group_risk_budget_empty_inputs() -> None:
    with pytest.raises(ConfigurationError):
        group_risk_budget({}, {"A": "eq"})
    with pytest.raises(ConfigurationError):
        group_risk_budget({"eq": 1.0}, {})


def test_group_risk_budget_group_without_members() -> None:
    with pytest.raises(ConfigurationError):
        group_risk_budget({"eq": 0.5, "bond": 0.5}, {"A": "eq", "B": "eq"})


def test_group_risk_budget_negative_share() -> None:
    with pytest.raises(ConfigurationError):
        group_risk_budget({"eq": -1.0}, {"A": "eq"})


def test_group_risk_budget_bad_within_weight() -> None:
    with pytest.raises(ConfigurationError):
        group_risk_budget({"eq": 1.0}, {"A": "eq", "B": "eq"}, within_group={"A": -1.0})


def test_contribution_shape_validation() -> None:
    from riskbudget.core.errors import ValidationError

    with pytest.raises(ValidationError):
        portfolio_volatility(np.array([0.5, 0.5]), COV)  # 2 weights vs 3x3 cov


def test_align_budget() -> None:
    b = RiskBudget.equal(["A", "B"])
    assert np.allclose(align_budget(b, ["B", "A"]), [0.5, 0.5])


def test_conditional_budget_flat_signal_is_erc() -> None:
    cb = ConditionalBudget(
        lambda s: None if s is None else RiskBudget.from_weights({"A": s, "B": 1.0}),
        assets=["A", "B"],
    )
    assert np.isclose(cb.budget_for(None).budgets["A"], 0.5)
    moved = cb.budget_for(3.0)
    assert moved.budgets["A"] > 0.5


def test_conditional_budget_validates_coverage() -> None:
    cb = ConditionalBudget(
        lambda s: RiskBudget.equal(["X", "Y", "Z"]),
        assets=["A", "B"],
    )
    with pytest.raises(ConfigurationError):
        cb.budget_for("anything")
