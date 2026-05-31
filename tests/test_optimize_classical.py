"""Tests for classical constructors: equal-weight, GMV, MSR, EF (BUILD_PLAN §12.2)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, RiskBudget
from riskbudget.optimize.classical import (
    EfficientMSR,
    efficient_frontier,
    efficient_msr,
    equal_weight,
    gmv,
    gmv_sigma_closed_form,
    msr,
)

COV3 = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]])
ASSETS3 = ["A", "B", "C"]
MU3 = ExpectedReturns({"A": 0.08, "B": 0.10, "C": 0.12})
BUDGET3 = RiskBudget.equal(ASSETS3)


def test_equal_weight() -> None:
    pf = equal_weight().construct(COV3, budget=BUDGET3, constraints=Constraints())
    assert np.allclose(pf.as_array(ASSETS3), 1 / 3)


def test_gmv_beats_random_on_variance() -> None:
    pf = gmv().construct(COV3, budget=BUDGET3, constraints=Constraints())
    gmv_var = float(pf.as_array(ASSETS3) @ COV3 @ pf.as_array(ASSETS3))
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.uniform(0, 1, size=3)
        r /= r.sum()
        assert gmv_var <= float(r @ COV3 @ r) + 1e-9


def test_gmv_closed_form_sanity() -> None:
    pf = gmv().construct(COV3, budget=BUDGET3, constraints=Constraints())
    vol = pf.volatility(COV3, ASSETS3)
    # Long-only constrained GMV vol >= unconstrained closed-form (here equal as
    # the unconstrained GMV happens to be long-only on this fixture).
    assert vol >= gmv_sigma_closed_form(COV3) - 1e-6


def test_msr_two_asset_analytic_tangency() -> None:
    # Toy 2-asset, uncorrelated. Analytic tangency (long-only, full-invest) for
    # min wᵀΣw s.t. (μ-rf)ᵀw=1: w ∝ Σ⁻¹(μ-rf), then normalize.
    cov = np.diag([0.04, 0.09])
    mu = ExpectedReturns({"X": 0.10, "Y": 0.15})
    rf = 0.02
    pf = msr(risk_free=rf).construct(cov, mu=mu, constraints=Constraints())
    w = pf.as_array(["X", "Y"])
    excess = np.array([0.10, 0.15]) - rf
    raw = np.linalg.solve(cov, excess)
    expected = raw / raw.sum()
    assert np.allclose(w, expected, atol=1e-5)


def test_msr_requires_mu() -> None:
    with pytest.raises(OptimizationError):
        msr().construct(COV3, constraints=Constraints())


def test_msr_guards_below_riskfree() -> None:
    mu = ExpectedReturns({"A": 0.01, "B": 0.02, "C": 0.0})
    with pytest.raises(OptimizationError):
        msr(risk_free=0.05).construct(COV3, mu=mu, constraints=Constraints())


def test_efficient_msr_is_constructor() -> None:
    pf = efficient_msr(risk_free=0.0).construct(COV3, mu=MU3, constraints=Constraints())
    assert np.isclose(pf.as_array(ASSETS3).sum(), 1.0)
    assert isinstance(efficient_msr(), EfficientMSR)


def test_efficient_frontier_monotone_and_convex() -> None:
    frontier = efficient_frontier(COV3, MU3, Constraints(), n_points=10)
    rets = np.array([float(p.as_array(ASSETS3) @ MU3.as_array(ASSETS3)) for p in frontier])
    vols = np.array([p.volatility(COV3, ASSETS3) for p in frontier])
    # Returns strictly increasing across the grid.
    assert np.all(np.diff(rets) > -1e-9)
    # Variance is convex in target return (second difference non-negative).
    var = vols**2
    assert np.all(np.diff(var, 2) > -1e-6)


def test_efficient_frontier_min_points() -> None:
    with pytest.raises(OptimizationError):
        efficient_frontier(COV3, MU3, Constraints(), n_points=1)
