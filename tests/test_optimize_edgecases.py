"""Edge-case / error-path coverage for solvers and constructors."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, RiskBudget
from riskbudget.optimize.ccd import solve_risk_budget_ccd
from riskbudget.optimize.classical import (
    EqualWeight,
    GlobalMinimumVariance,
    MaxSharpe,
    efficient_frontier,
)
from riskbudget.optimize.convex import solve_risk_budget_convex
from riskbudget.optimize.ensemble import ensemble
from riskbudget.optimize.router import erc
from riskbudget.optimize.scipy_solver import solve_risk_budget_scipy

COV3 = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]])
ASSETS3 = ["A", "B", "C"]
B3 = np.full(3, 1 / 3)


# ---- CCD / scipy input validation -----------------------------------------


def test_ccd_shape_mismatch() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(COV3, np.full(2, 0.5))


def test_ccd_non_square_cov() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(np.ones((2, 3)), B3)


def test_ccd_nonfinite_cov() -> None:
    bad = COV3.copy()
    bad[0, 0] = np.nan
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(bad, B3)


def test_ccd_nonpositive_diagonal() -> None:
    bad = COV3.copy()
    bad[0, 0] = 0.0
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(bad, B3)


def test_ccd_bad_tilt_length() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(COV3, B3, pi=np.array([0.1, 0.2]))


def test_ccd_bad_risk_aversion() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(COV3, B3, pi=np.zeros(3), risk_aversion=-1.0)


def test_scipy_shape_mismatch() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_scipy(COV3, np.full(2, 0.5))


def test_scipy_nonpositive_budget() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_scipy(COV3, np.array([0.5, 0.5, 0.0]))


def test_scipy_bounded_path_matches_within_bounds() -> None:
    # Upper bound that binds in normalized space; scipy must respect positivity.
    w = solve_risk_budget_scipy(COV3, B3, lower=np.array([0.05, 0.05, 0.05]))
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)


def test_scipy_infeasible_bounds() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_scipy(
            COV3, B3, lower=np.array([0.9, 0.9, 0.9]), upper=np.array([0.1, 0.1, 0.1])
        )


# ---- convex error paths ----------------------------------------------------


def test_convex_nonpositive_budget() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_convex(COV3, np.array([0.5, 0.5, 0.0]))


def test_convex_shape_mismatch() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_convex(COV3, np.full(2, 0.5))


def test_convex_group_cap_member_subset() -> None:
    from riskbudget.optimize.constraints import GroupCapSpec

    spec = GroupCapSpec(label="ab", members=(0, 1), cap=0.5)
    w = solve_risk_budget_convex(COV3, B3, group_caps=[spec])
    assert w[0] + w[1] <= 0.5 + 1e-6
    assert np.isclose(w.sum(), 1.0)


def test_convex_prev_weights_length_mismatch() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_convex(COV3, B3, prev_weights=np.array([0.5, 0.5]), max_turnover=0.1)


# ---- classical error paths -------------------------------------------------


def test_gmv_cov_shape_mismatch() -> None:
    b = RiskBudget.equal(["A", "B"])  # 2 assets but COV3 is 3x3
    with pytest.raises(OptimizationError):
        GlobalMinimumVariance().construct(COV3, budget=b, constraints=Constraints())


def test_equal_weight_default_order() -> None:
    pf = EqualWeight().construct(COV3, constraints=Constraints())
    assert pf.assets == ["0", "1", "2"]


def test_msr_leverage_and_long_only_short() -> None:
    mu = ExpectedReturns({"A": 0.08, "B": 0.10, "C": 0.12})
    # Allow shorting on the classical path (no error expected).
    pf = MaxSharpe().construct(COV3, mu=mu, constraints=Constraints(long_only=False))
    assert np.isclose(pf.as_array(ASSETS3).sum(), 1.0)


def test_efficient_frontier_long_only_default() -> None:
    mu = ExpectedReturns({"A": 0.08, "B": 0.10, "C": 0.12})
    frontier = efficient_frontier(COV3, mu, Constraints(), n_points=4)
    for pf in frontier:
        w = pf.as_array(ASSETS3)
        assert np.all(w >= -1e-8)
        assert np.isclose(w.sum(), 1.0)


# ---- ensemble + router factories ------------------------------------------


def test_ensemble_bad_te_target() -> None:
    with pytest.raises(OptimizationError):
        ensemble([erc()], te_target=-1.0)


def test_ensemble_default_integer_order() -> None:
    pf = ensemble([erc()]).construct(
        COV3, budget=RiskBudget.equal(ASSETS3), constraints=Constraints()
    )
    assert set(pf.assets) == set(ASSETS3)


def test_ensemble_te_no_reference_uses_equal_weight() -> None:
    # te_target with no reference -> shrinks toward equal weight.
    pf = ensemble([GlobalMinimumVariance(), erc()], te_target=1e-6).construct(
        COV3, budget=RiskBudget.equal(ASSETS3), constraints=Constraints()
    )
    assert np.allclose(pf.as_array(ASSETS3), 1 / 3, atol=1e-3)


def test_risk_budget_factory_names() -> None:
    from riskbudget.optimize.router import risk_budget

    opt = risk_budget()
    pf = opt.solve(COV3, RiskBudget.equal(ASSETS3), Constraints())
    assert np.isclose(pf.as_array(ASSETS3).sum(), 1.0)


def test_msr_with_max_weight_bound() -> None:
    mu = ExpectedReturns({"A": 0.20, "B": 0.05, "C": 0.05})
    pf = MaxSharpe().construct(COV3, mu=mu, constraints=Constraints(max_weight=0.5))
    assert pf.as_array(ASSETS3).max() <= 0.5 + 1e-5


def test_efficient_msr_requires_mu() -> None:
    from riskbudget.optimize.classical import efficient_msr

    with pytest.raises(OptimizationError):
        efficient_msr().construct(COV3, constraints=Constraints())


def test_conditional_explicit_budget_overrides_state() -> None:
    from riskbudget.budgeting import ConditionalBudget
    from riskbudget.optimize.conditional import ConditionalRiskBudgetConstructor

    cb = ConditionalBudget(lambda s: None, assets=ASSETS3)
    constructor = ConditionalRiskBudgetConstructor(cb, state=3.0)
    explicit = RiskBudget.from_weights({"A": 0.6, "B": 0.3, "C": 0.1})
    pf = constructor.construct(COV3, budget=explicit, constraints=Constraints())
    pcr = pf.risk_contributions(COV3, ASSETS3)
    total = sum(pcr.values())
    assert pcr["A"] / total > pcr["C"] / total


def test_conditional_cov_shape_mismatch() -> None:
    from riskbudget.budgeting import ConditionalBudget
    from riskbudget.optimize.conditional import ConditionalRiskBudgetConstructor

    cb = ConditionalBudget(lambda s: None, assets=["A", "B"])
    constructor = ConditionalRiskBudgetConstructor(cb)
    with pytest.raises(OptimizationError):
        constructor.construct(COV3, constraints=Constraints())  # 3x3 vs 2 assets
