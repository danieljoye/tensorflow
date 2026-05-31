"""Route a risk-budget solve to the cheapest path the ``Constraints`` allow.

Mirrors pyrb's ``_lambda_solve`` dispatch (BUILD_PLAN §3.1, §12.1):

- **Pure long-only + leverage** → the fast pure-NumPy CCD path
  (:func:`riskbudget.optimize.ccd.solve_risk_budget_ccd`).
- **Per-asset box bounds, group/sector caps, turnover, arbitrary linear** → the
  cvxpy log-barrier program
  (:func:`riskbudget.optimize.convex.solve_risk_budget_convex`). Box bounds on the
  *normalized* weight couple the coordinates through the ``Σw=1`` normalization, so
  they are not CCD-separable. The scipy log-barrier
  (:func:`riskbudget.optimize.scipy_solver.solve_risk_budget_scipy`) is the
  unbounded cross-check oracle.

The risk-budget path is **inherently long-only** (BUILD_PLAN §3.1): if the
constraints would permit shorting, :func:`route_risk_budget` raises
:class:`OptimizationError` — it never silently long-onlys a shorting request.

Source: pyrb (Richard–Roncalli 2019); Spinu (2013); BUILD_PLAN §3.1, §11, §12.1.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import Portfolio, RiskBudget
from riskbudget.optimize.ccd import solve_risk_budget_ccd
from riskbudget.optimize.constraints import (
    is_separable,
    require_long_only,
    resolve_bounds,
    resolve_group_caps,
    resolve_leverage,
)
from riskbudget.optimize.convex import solve_risk_budget_convex


def route_risk_budget(
    cov: np.ndarray,
    budget: RiskBudget,
    constraints: Constraints,
    *,
    assets: list[str] | None = None,
    prev_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the risk-budget program by the cheapest feasible path; return weights.

    Parameters
    ----------
    cov:
        Covariance matrix aligned to ``assets`` (or ``budget.assets`` when
        ``assets`` is ``None``).
    budget:
        The target :class:`RiskBudget`.
    constraints:
        Feasibility limits. Must be long-only (else ``OptimizationError``).
    assets:
        Explicit ordering for ``cov``/``budget``. Defaults to ``budget.assets``.
    prev_weights:
        Previous-period weights for a turnover constraint (used only by the
        convex path when ``constraints.max_turnover`` is set).

    Returns
    -------
    np.ndarray
        Long-only weights in the resolved asset order, scaled to the leverage.
    """
    require_long_only(constraints, context="Risk budgeting")

    order = list(budget.assets) if assets is None else [str(a) for a in assets]
    b = budget.as_array(order)
    n = b.size
    leverage = resolve_leverage(constraints)
    lower, upper = resolve_bounds(constraints, n)
    # Per-asset box bounds on the *normalized* weight (beyond plain positivity)
    # couple the coordinates through the Σw=1 normalization, so they are not
    # CCD-separable. Only a pure long-only + leverage problem hits the fast path.
    has_upper = bool(np.any(np.isfinite(upper)))
    has_lower = bool(np.any(lower > 0.0))

    if is_separable(constraints) and not has_upper and not has_lower:
        return solve_risk_budget_ccd(cov, b, leverage=leverage)

    # Box bounds / group caps / turnover / general linear -> cvxpy (solves in w).
    group_caps = resolve_group_caps(constraints, order)
    return solve_risk_budget_convex(
        cov,
        b,
        leverage=leverage,
        lower=lower,
        upper=upper,
        group_caps=group_caps,
        prev_weights=prev_weights,
        max_turnover=constraints.max_turnover,
    )


class RiskBudgetOptimizer:
    """The default :class:`~riskbudget.core.interfaces.Optimizer` (risk-budget path).

    Implements ``solve(cov, budget, constraints) -> Portfolio`` by routing to the
    cheapest solver the constraints allow. Long-only only (BUILD_PLAN §3.1).

    Parameters
    ----------
    prev_weights:
        Optional previous-period weights (keyed by asset) used when a turnover
        constraint is active. Supplied by the backtester between rebalances.
    """

    def __init__(self, prev_weights: dict[str, float] | None = None) -> None:
        self._prev_weights = dict(prev_weights) if prev_weights else None

    def solve(
        self,
        cov: np.ndarray,
        budget: RiskBudget,
        constraints: Constraints,
    ) -> Portfolio:
        """Return a :class:`Portfolio` whose risk contributions match ``budget``.

        ``cov`` must be ordered to match ``budget.assets``. Raises
        :class:`OptimizationError` on infeasibility, non-convergence, or if the
        constraints permit shorting.
        """
        order = list(budget.assets)
        prev = None
        if self._prev_weights is not None:
            prev = np.array([self._prev_weights.get(a, 0.0) for a in order], dtype=float)
        weights = route_risk_budget(cov, budget, constraints, assets=order, prev_weights=prev)
        return Portfolio(dict(zip(order, (float(w) for w in weights), strict=True)))

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: object | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        """:class:`PortfolioConstructor` adapter for the risk-budget path.

        ``budget`` is required; ``mu`` is ignored (the risk-budget path needs no
        expected returns). Raises :class:`OptimizationError` if ``budget`` is
        missing.
        """
        if budget is None:
            raise OptimizationError("The risk-budget constructor requires a `budget`.")
        return self.solve(cov, budget, constraints)


def erc() -> RiskBudgetOptimizer:
    """Factory for the ERC / risk-budget optimizer (§5.2 name ``erc``).

    The constructor takes the budget at solve time; for ERC the caller passes
    :meth:`RiskBudget.equal`. Same object backs the ``risk_budget`` name.
    """
    return RiskBudgetOptimizer()


def risk_budget() -> RiskBudgetOptimizer:
    """Factory for the general risk-budget optimizer (§5.2 name ``risk_budget``)."""
    return RiskBudgetOptimizer()


__all__ = ["RiskBudgetOptimizer", "erc", "risk_budget", "route_risk_budget"]
