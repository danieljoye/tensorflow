"""Solvers, constraints, and constructors for risk-budget & classical portfolios.

The package exposes three risk-budget solver paths (CCD default, scipy oracle,
cvxpy for non-separable constraints), a router/``Optimizer`` over them, the
classical benchmark constructors (equal-weight, GMV, MSR, Efficient-MSR,
efficient frontier), conditional / state-dependent construction, and the ensemble
"diversifying the diversifiers" blend.

§5.2 factory callables (registry-facing): ``erc``, ``risk_budget``, ``gmv``,
``msr``, ``efficient_msr``, ``efficient_frontier``, ``ensemble`` (plus
``equal_weight``).

Source: Spinu (2013); Maillard–Roncalli–Teïletche (JPM 2010); Cornuéjols–Tütüncü
(2006); Martellini (2008); Amenc–Goltz–Lodh–Martellini (2012);
Martellini–Milhau–Tarelli (2015). BUILD_PLAN §2, §3.1, §11, §12.1, §12.2.
"""

from __future__ import annotations

from riskbudget.optimize.ccd import solve_risk_budget_ccd
from riskbudget.optimize.classical import (
    EfficientMSR,
    EqualWeight,
    GlobalMinimumVariance,
    MaxSharpe,
    efficient_frontier,
    efficient_msr,
    equal_weight,
    gmv,
    gmv_sigma_closed_form,
    msr,
)
from riskbudget.optimize.conditional import ConditionalRiskBudgetConstructor
from riskbudget.optimize.constraints import (
    GroupCapSpec,
    has_group_caps,
    is_separable,
    require_long_only,
    resolve_bounds,
    resolve_group_caps,
    resolve_leverage,
)
from riskbudget.optimize.convex import solve_risk_budget_convex
from riskbudget.optimize.ensemble import EnsembleConstructor, ensemble
from riskbudget.optimize.router import (
    RiskBudgetOptimizer,
    erc,
    risk_budget,
    route_risk_budget,
)
from riskbudget.optimize.scipy_solver import solve_risk_budget_scipy

__all__ = [
    "ConditionalRiskBudgetConstructor",
    "EfficientMSR",
    "EnsembleConstructor",
    "EqualWeight",
    "GlobalMinimumVariance",
    "GroupCapSpec",
    "MaxSharpe",
    "RiskBudgetOptimizer",
    "efficient_frontier",
    "efficient_msr",
    "ensemble",
    "equal_weight",
    "erc",
    "gmv",
    "gmv_sigma_closed_form",
    "has_group_caps",
    "is_separable",
    "msr",
    "require_long_only",
    "resolve_bounds",
    "resolve_group_caps",
    "resolve_leverage",
    "risk_budget",
    "route_risk_budget",
    "solve_risk_budget_ccd",
    "solve_risk_budget_convex",
    "solve_risk_budget_scipy",
]
