"""Risk-contribution math and risk-budget specification helpers (BUILD_PLAN §2).

- :mod:`riskbudget.budgeting.contributions` — MRC / TRC / percentage
  contributions and the full additive :class:`RiskDecomposition`
  (``Σᵢ TRCᵢ = σ(w)``).
- :mod:`riskbudget.budgeting.budget` — construction/validation helpers for
  :class:`~riskbudget.core.types.RiskBudget` (ERC default, per-group expansion)
  plus the conditional / state-dependent :class:`ConditionalBudget`.
"""

from __future__ import annotations

from riskbudget.budgeting.budget import (
    ConditionalBudget,
    align_budget,
    budget_from_weights,
    equal_risk_budget,
    group_risk_budget,
)
from riskbudget.budgeting.contributions import (
    RiskDecomposition,
    decompose_risk,
    marginal_risk_contributions,
    percentage_risk_contributions,
    portfolio_volatility,
    total_risk_contributions,
)

__all__ = [
    "ConditionalBudget",
    "RiskDecomposition",
    "align_budget",
    "budget_from_weights",
    "decompose_risk",
    "equal_risk_budget",
    "group_risk_budget",
    "marginal_risk_contributions",
    "percentage_risk_contributions",
    "portfolio_volatility",
    "total_risk_contributions",
]
