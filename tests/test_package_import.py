"""Smoke tests: every §5 contract is importable from ``riskbudget.core``."""

from __future__ import annotations

import riskbudget


def test_version() -> None:
    assert riskbudget.__version__


def test_all_contracts_importable_from_core() -> None:
    from riskbudget.core import (  # noqa: F401
        Allocator,
        AllocatorParams,
        Backtester,
        BacktestError,
        BacktestResult,
        ConfigurationError,
        Constraints,
        DataError,
        DataSource,
        ExpectedReturns,
        MeanModel,
        OptimizationError,
        Optimizer,
        Portfolio,
        PortfolioConstructor,
        PriceData,
        RebalanceSchedule,
        ReturnMatrix,
        RiskBudget,
        RiskBudgetError,
        RiskModel,
        RiskModelError,
        ValidationError,
    )


def test_wave_05_contracts_importable_from_core() -> None:
    from riskbudget.core import (  # noqa: F401
        Allocator,
        AllocatorParams,
        ExpectedReturns,
        MeanModel,
        PortfolioConstructor,
    )
