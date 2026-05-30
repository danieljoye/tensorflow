"""Core contracts for the Portfolio Risk Budgeting system.

This package is the *frozen contract* (BUILD_PLAN §5) that every downstream agent
imports. It contains the shared, validated data types, the pipeline protocols,
and the exception hierarchy. Everything a downstream agent needs is re-exported
here so it can be imported straight from ``riskbudget.core``::

    from riskbudget.core import Portfolio, RiskBudget, RiskModel, Optimizer
"""

from __future__ import annotations

from riskbudget.core.errors import (
    BacktestError,
    ConfigurationError,
    DataError,
    OptimizationError,
    RiskBudgetError,
    RiskModelError,
    ValidationError,
)
from riskbudget.core.interfaces import (
    Backtester,
    Constraints,
    DataSource,
    Optimizer,
    RebalanceFrequency,
    RebalanceSchedule,
    RiskModel,
)
from riskbudget.core.types import (
    BacktestResult,
    Portfolio,
    PriceData,
    ReturnMatrix,
    ReturnMethod,
    RiskBudget,
)

__all__ = [
    # types
    "ReturnMethod",
    "ReturnMatrix",
    "PriceData",
    "RiskBudget",
    "Portfolio",
    "BacktestResult",
    # interfaces
    "RebalanceFrequency",
    "Constraints",
    "RebalanceSchedule",
    "DataSource",
    "RiskModel",
    "Optimizer",
    "Backtester",
    # errors
    "RiskBudgetError",
    "ValidationError",
    "DataError",
    "RiskModelError",
    "OptimizationError",
    "BacktestError",
    "ConfigurationError",
]
