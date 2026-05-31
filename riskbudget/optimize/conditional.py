"""Conditional / state-dependent risk-budget construction (Martellini–Milhau–Tarelli).

"Toward Conditional Risk Parity" (JAI 2015): the target risk contributions ``bᵢ``
may depend on an observable *state* variable (a rate level, valuation score,
yield-curve slope, ...). This constructor resolves the budget for the current
state via a pluggable :class:`~riskbudget.budgeting.budget.ConditionalBudget`
(a callable ``state -> RiskBudget``) and then solves the standard long-only
risk-budget program. When the signal is flat (``state is None`` or the mapping
declines), the budget falls back to ERC — so conditional risk parity reduces to
plain risk parity, addressing unconditional risk parity's structural bond
overweight in changing rate regimes.

Source: Martellini, Milhau & Tarelli (JAI 2015); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from riskbudget.budgeting.budget import ConditionalBudget
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget
from riskbudget.optimize.router import RiskBudgetOptimizer


class ConditionalRiskBudgetConstructor:
    """Solve a risk-budget portfolio whose budget depends on an observable state.

    Parameters
    ----------
    conditional_budget:
        The pluggable ``state -> RiskBudget`` mapping (ERC fallback on a flat
        signal).
    state:
        The current state value passed to the mapping at construction time.
        ``None`` (default) selects the ERC fallback.
    """

    def __init__(
        self,
        conditional_budget: ConditionalBudget,
        state: Any = None,
    ) -> None:
        self._cbudget = conditional_budget
        self._state = state
        self._optimizer = RiskBudgetOptimizer()

    @property
    def state(self) -> Any:
        """The state value driving the budget resolution."""
        return self._state

    def with_state(self, state: Any) -> ConditionalRiskBudgetConstructor:
        """Return a new constructor bound to ``state`` (immutable update)."""
        return ConditionalRiskBudgetConstructor(self._cbudget, state)

    def resolve_budget(self) -> RiskBudget:
        """The :class:`RiskBudget` for the current state (ERC fallback if flat)."""
        return self._cbudget.budget_for(self._state)

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        """Construct the conditional risk-budget portfolio.

        ``mu`` is ignored. An explicit ``budget`` overrides the state-resolved one
        (so the backtester can still inject a budget if desired); otherwise the
        budget comes from the conditional mapping. The asset order is taken from
        the resolved/used budget.
        """
        used_budget = budget if budget is not None else self.resolve_budget()
        order = list(used_budget.assets)
        sigma = np.asarray(cov, dtype=float)
        if sigma.shape != (len(order), len(order)):
            raise OptimizationError(
                f"Covariance shape {sigma.shape} inconsistent with {len(order)} assets."
            )
        return self._optimizer.solve(sigma, used_budget, constraints)


__all__ = ["ConditionalRiskBudgetConstructor"]
