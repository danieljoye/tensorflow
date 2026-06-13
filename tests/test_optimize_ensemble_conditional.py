"""Tests for ensemble + conditional construction (BUILD_PLAN §2, §11)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.budgeting import ConditionalBudget
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget
from riskbudget.optimize.classical import equal_weight, gmv
from riskbudget.optimize.conditional import ConditionalRiskBudgetConstructor
from riskbudget.optimize.ensemble import EnsembleConstructor, ensemble
from riskbudget.optimize.router import RiskBudgetOptimizer, erc

COV3 = np.array([[0.04, 0.006, 0.0], [0.006, 0.09, 0.0], [0.0, 0.0, 0.16]])
ASSETS3 = ["A", "B", "C"]
BUDGET3 = RiskBudget.equal(ASSETS3)
MU3 = ExpectedReturns({"A": 0.08, "B": 0.10, "C": 0.12})


def test_ensemble_is_mean_of_constituents() -> None:
    ens = ensemble([gmv(), erc()])
    blended = ens.construct(COV3, budget=BUDGET3, constraints=Constraints())
    g = gmv().construct(COV3, budget=BUDGET3, constraints=Constraints()).as_array(ASSETS3)
    e = erc().solve(COV3, BUDGET3, Constraints()).as_array(ASSETS3)
    mean = (g / g.sum() + e / e.sum()) / 2
    assert np.allclose(blended.as_array(ASSETS3), mean)


def test_ensemble_degrades_when_constituent_fails() -> None:
    # MSR needs mu; with no mu it raises and should be skipped, leaving erc only.
    from riskbudget.optimize.classical import msr

    ens = ensemble([msr(), erc()])
    blended = ens.construct(COV3, budget=BUDGET3, constraints=Constraints())
    e = erc().solve(COV3, BUDGET3, Constraints()).as_array(ASSETS3)
    assert np.allclose(blended.as_array(ASSETS3), e / e.sum())


def test_ensemble_all_fail_raises() -> None:
    from riskbudget.optimize.classical import msr

    ens = ensemble([msr()])
    with pytest.raises(OptimizationError):
        ens.construct(COV3, budget=BUDGET3, constraints=Constraints())


def test_ensemble_te_overlay_reduces_tracking_error() -> None:
    # Reference = equal weight. A tight te_target shrinks toward it, lowering TE.
    ref = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    base = ensemble([gmv(), erc()]).construct(COV3, budget=BUDGET3, constraints=Constraints())
    ref_vec = np.array([1 / 3, 1 / 3, 1 / 3])

    def te(w: np.ndarray) -> float:
        d = w - ref_vec
        return float(np.sqrt(d @ COV3 @ d))

    base_te = te(base.as_array(ASSETS3))
    controlled = ensemble([gmv(), erc()], reference=ref, te_target=base_te / 2).construct(
        COV3, budget=BUDGET3, constraints=Constraints()
    )
    assert te(controlled.as_array(ASSETS3)) <= base_te / 2 + 1e-6


def test_ensemble_requires_constituent() -> None:
    with pytest.raises(OptimizationError):
        EnsembleConstructor([])


def test_conditional_reduces_to_erc_on_flat_signal() -> None:
    cb = ConditionalBudget(
        lambda s: None if s is None else RiskBudget.from_weights({"A": s, "B": 1.0, "C": 1.0}),
        assets=ASSETS3,
    )
    constructor = ConditionalRiskBudgetConstructor(cb, state=None)
    pf = constructor.construct(COV3, constraints=Constraints())
    erc_pf = RiskBudgetOptimizer().solve(COV3, BUDGET3, Constraints())
    assert np.allclose(pf.as_array(ASSETS3), erc_pf.as_array(ASSETS3))


def test_conditional_shifts_on_active_signal() -> None:
    cb = ConditionalBudget(
        lambda s: RiskBudget.from_weights({"A": s, "B": 1.0, "C": 1.0}),
        assets=ASSETS3,
    )
    flat = ConditionalRiskBudgetConstructor(cb, state=1.0).construct(
        COV3, constraints=Constraints()
    )
    tilted = ConditionalRiskBudgetConstructor(cb, state=5.0).construct(
        COV3, constraints=Constraints()
    )
    # More budget to A -> A carries more risk -> larger weight.
    assert tilted.weights["A"] > flat.weights["A"]


def test_conditional_with_state_immutability() -> None:
    cb = ConditionalBudget(lambda s: None, assets=ASSETS3)
    c1 = ConditionalRiskBudgetConstructor(cb)
    c2 = c1.with_state(3.0)
    assert c1.state is None
    assert c2.state == 3.0


def test_ensemble_returns_portfolio_type() -> None:
    blended = ensemble([equal_weight(), erc()]).construct(
        COV3, budget=BUDGET3, constraints=Constraints()
    )
    assert isinstance(blended, Portfolio)
