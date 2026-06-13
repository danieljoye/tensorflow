"""Tests for the risk-budget solvers: CCD, scipy, convex, router (BUILD_PLAN §12.1)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import RiskBudget
from riskbudget.optimize.ccd import solve_risk_budget_ccd
from riskbudget.optimize.convex import solve_risk_budget_convex
from riskbudget.optimize.router import RiskBudgetOptimizer, route_risk_budget
from riskbudget.optimize.scipy_solver import solve_risk_budget_scipy

# Fixed synthetic fixtures.
COV3 = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]])
ASSETS3 = ["A", "B", "C"]


def _trc(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    vol = float(np.sqrt(w @ cov @ w))
    return w * (cov @ w) / vol


def test_ccd_erc_equal_risk_contributions() -> None:
    b = np.full(3, 1 / 3)
    w = solve_risk_budget_ccd(COV3, b)
    trc = _trc(w, COV3)
    assert np.allclose(trc, trc.mean(), atol=1e-7)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)


def test_ccd_two_asset_analytic() -> None:
    # Diagonal cov: ERC weights ∝ 1/σ, so w ∝ (1/σ1, 1/σ2).
    cov = np.diag([0.04, 0.16])  # σ = 0.2, 0.4
    w = solve_risk_budget_ccd(cov, np.array([0.5, 0.5]))
    expected = np.array([1 / 0.2, 1 / 0.4])
    expected = expected / expected.sum()
    assert np.allclose(w, expected, atol=1e-6)


def test_ccd_arbitrary_budget_matched() -> None:
    b = np.array([0.6, 0.3, 0.1])
    w = solve_risk_budget_ccd(COV3, b)
    pcr = _trc(w, COV3)
    pcr = pcr / pcr.sum()
    assert np.allclose(pcr, b, atol=1e-6)


def test_ccd_vs_scipy_agree() -> None:
    for b in (np.full(3, 1 / 3), np.array([0.5, 0.3, 0.2]), np.array([0.7, 0.2, 0.1])):
        w_ccd = solve_risk_budget_ccd(COV3, b)
        w_sp = solve_risk_budget_scipy(COV3, b)
        assert np.allclose(w_ccd, w_sp, atol=1e-6)


def test_ccd_vs_convex_agree_unconstrained() -> None:
    b = np.array([0.5, 0.3, 0.2])
    w_ccd = solve_risk_budget_ccd(COV3, b)
    w_cvx = solve_risk_budget_convex(COV3, b)
    assert np.allclose(w_ccd, w_cvx, atol=1e-5)


def test_leverage_rescales() -> None:
    w = solve_risk_budget_ccd(COV3, np.full(3, 1 / 3), leverage=2.0)
    assert np.isclose(w.sum(), 2.0)


def test_ccd_expected_return_tilt_runs() -> None:
    b = np.full(3, 1 / 3)
    base = solve_risk_budget_ccd(COV3, b)
    tilted = solve_risk_budget_ccd(COV3, b, pi=np.array([0.0, 0.0, 0.10]), risk_aversion=1.0)
    # A positive tilt on asset C should raise its weight vs. plain ERC.
    assert tilted[2] > base[2]


def test_router_separable_uses_ccd_result() -> None:
    b = RiskBudget.equal(ASSETS3)
    w = route_risk_budget(COV3, b, Constraints(), assets=ASSETS3)
    assert np.allclose(w, solve_risk_budget_ccd(COV3, b.as_array(ASSETS3)))


def test_router_max_weight_binds() -> None:
    b = RiskBudget.equal(ASSETS3)
    w = route_risk_budget(COV3, b, Constraints(max_weight=0.4), assets=ASSETS3)
    assert w.max() <= 0.4 + 1e-6


def test_router_group_cap_binds() -> None:
    b = RiskBudget.equal(ASSETS3)
    con = Constraints(group_caps={"eq": 0.5}, groups={"A": "eq", "B": "eq"})
    w = route_risk_budget(COV3, b, con, assets=ASSETS3)
    assert w[0] + w[1] <= 0.5 + 1e-6


def test_optimizer_solve_returns_portfolio() -> None:
    opt = RiskBudgetOptimizer()
    pf = opt.solve(COV3, RiskBudget.equal(ASSETS3), Constraints())
    assert set(pf.weights) == set(ASSETS3)
    rc = pf.risk_contributions(COV3, ASSETS3)
    vals = np.array(list(rc.values()))
    assert np.allclose(vals, vals.mean(), atol=1e-7)


def test_turnover_constraint_reduces_trade() -> None:
    b = RiskBudget.equal(ASSETS3)
    prev = {"A": 0.33, "B": 0.34, "C": 0.33}
    opt = RiskBudgetOptimizer(prev_weights=prev)
    pf = opt.solve(COV3, b, Constraints(max_turnover=0.05))
    realized = sum(abs(pf.weights[a] - prev[a]) for a in ASSETS3)
    assert realized <= 0.05 + 1e-6


def test_risk_budget_raises_on_shorting() -> None:
    opt = RiskBudgetOptimizer()
    with pytest.raises(OptimizationError):
        opt.solve(COV3, RiskBudget.equal(ASSETS3), Constraints(long_only=False))


def test_risk_budget_raises_on_negative_min_weight() -> None:
    opt = RiskBudgetOptimizer()
    with pytest.raises(OptimizationError):
        opt.solve(COV3, RiskBudget.equal(ASSETS3), Constraints(min_weight=-0.1))


def test_construct_adapter_requires_budget() -> None:
    opt = RiskBudgetOptimizer()
    with pytest.raises(OptimizationError):
        opt.construct(COV3, constraints=Constraints())


def test_ccd_rejects_nonpositive_budget() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(COV3, np.array([0.5, 0.5, 0.0]))


def test_ccd_nonconvergence_raises() -> None:
    with pytest.raises(OptimizationError):
        solve_risk_budget_ccd(COV3, np.full(3, 1 / 3), maxiter=0)


# Property-based invariant: weights positive and sum to 1 for random budgets/cov.
def test_property_weights_positive_sum_one() -> None:
    rng = np.random.default_rng(42)
    for _ in range(25):
        n = int(rng.integers(2, 8))
        a = rng.normal(size=(n, n))
        cov = a @ a.T + np.eye(n) * 0.01
        raw = rng.uniform(0.1, 1.0, size=n)
        b = raw / raw.sum()
        w = solve_risk_budget_ccd(cov, b)
        assert np.all(w > 0)
        assert np.isclose(w.sum(), 1.0)
        pcr = _trc(w, cov)
        pcr = pcr / pcr.sum()
        assert np.allclose(pcr, b, atol=1e-5)
