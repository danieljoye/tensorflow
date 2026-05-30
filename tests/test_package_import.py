"""Smoke tests: every §5 contract is importable from ``riskbudget.core``."""

from __future__ import annotations

import riskbudget


def test_version() -> None:
    assert riskbudget.__version__


def test_all_contracts_importable_from_core() -> None:
    from riskbudget.core import (  # noqa: F401
        Backtester,
        BacktestError,
        BacktestResult,
        ConfigurationError,
        Constraints,
        DataError,
        DataSource,
        OptimizationError,
        Optimizer,
        Portfolio,
        PriceData,
        RebalanceSchedule,
        ReturnMatrix,
        RiskBudget,
        RiskBudgetError,
        RiskModel,
        RiskModelError,
        ValidationError,
    )
