"""Shared exception hierarchy for the risk-budgeting system.

Every package raises errors that derive from :class:`RiskBudgetError` so callers
(the API layer in particular) can catch the whole family with a single
``except RiskBudgetError`` and map it to a sensible response.

Subclasses are grouped by the stage of the pipeline that raises them:

- :class:`ValidationError` — bad/inconsistent inputs to the core data types.
- :class:`DataError` — problems sourcing or shaping market data.
- :class:`RiskModelError` — covariance estimation failures.
- :class:`OptimizationError` — the weight solver failed or did not converge.
- :class:`BacktestError` — the walk-forward engine could not produce a result.
- :class:`ConfigurationError` — invalid run configuration (schedules, budgets).
"""

from __future__ import annotations


class RiskBudgetError(Exception):
    """Base class for all errors raised by the riskbudget package."""


class ValidationError(RiskBudgetError):
    """Raised when a core data type receives invalid or inconsistent input.

    Examples: a return matrix containing NaNs where they are not allowed, a risk
    budget whose weights do not sum to one, or a covariance matrix that is not
    square/symmetric.
    """


class DataError(RiskBudgetError):
    """Raised when market data cannot be sourced, parsed, or aligned."""


class RiskModelError(RiskBudgetError):
    """Raised when a covariance estimate cannot be produced or is degenerate."""


class OptimizationError(RiskBudgetError):
    """Raised when the weight optimizer fails, is infeasible, or won't converge."""


class BacktestError(RiskBudgetError):
    """Raised when the backtest engine cannot produce a result."""


class ConfigurationError(RiskBudgetError):
    """Raised for invalid run configuration (constraints, schedules, budgets)."""


__all__ = [
    "BacktestError",
    "ConfigurationError",
    "DataError",
    "OptimizationError",
    "RiskBudgetError",
    "RiskModelError",
    "ValidationError",
]
