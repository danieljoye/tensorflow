"""Portfolio Risk Budgeting system — curated public API.

A Python toolkit that constructs portfolios by *risk budget* rather than by
capital weight: given a target allocation of risk across assets, it estimates a
risk model, solves for weights whose risk contributions match the budget,
backtests the strategy, and reports performance and risk-contribution analytics.

This top-level module is the **stable surface** (BUILD_PLAN §3.1): the core data
types and a thin ``construct`` / ``backtest`` / ``compare`` layer driven by a
single :class:`StrategySpec` and the shared :data:`REGISTRY`. Internal module
paths may move; this surface should not.

Importing :mod:`riskbudget` pulls only the numeric/optimization stack — never the
optional API/dashboard/plotting deps. ``fastapi``, ``uvicorn``, ``streamlit`` and
``plotly`` are imported lazily by the surfaces that need them (the API, the
dashboard, and the reporting figure builders), so ``import riskbudget`` works in a
headless / minimal environment.

Examples
--------
>>> import numpy as np
>>> from riskbudget import StrategySpec, backtest, compare
>>> cov = (np.eye(3) * 0.04).tolist()
>>> erc = StrategySpec(
...     name="ERC",
...     assets=["A", "B", "C"],
...     method="erc",
...     data_source={"name": "synthetic", "params": {"cov": cov}},
...     schedule={"frequency": "monthly", "lookback": 60},
... )
>>> result = backtest(erc)  # doctest: +SKIP
"""

from __future__ import annotations

from riskbudget.compare import compare
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
    Allocator,
    Backtester,
    Constraints,
    DataSource,
    MeanModel,
    Optimizer,
    PortfolioConstructor,
    RebalanceSchedule,
    RiskModel,
)
from riskbudget.core.types import (
    AllocatorParams,
    BacktestResult,
    ExpectedReturns,
    Portfolio,
    PriceData,
    ReturnMatrix,
    RiskBudget,
)
from riskbudget.registry import REGISTRY, Registry, default_registry
from riskbudget.run import backtest, construct, fetch_prices
from riskbudget.spec import StrategySpec, strategy_spec_from_dict
from riskbudget.universe import default_universe, load_universe

__version__ = "0.1.0"

__all__ = [
    "REGISTRY",
    # core types
    "AllocatorParams",
    "Allocator",
    "BacktestResult",
    # errors
    "BacktestError",
    "Backtester",
    "ConfigurationError",
    "Constraints",
    "DataError",
    "DataSource",
    "ExpectedReturns",
    "MeanModel",
    "OptimizationError",
    "Optimizer",
    "Portfolio",
    "PortfolioConstructor",
    "PriceData",
    "RebalanceSchedule",
    "Registry",
    "ReturnMatrix",
    "RiskBudget",
    "RiskBudgetError",
    "RiskModel",
    "RiskModelError",
    # spec + registry + run + compare
    "StrategySpec",
    "ValidationError",
    "__version__",
    "backtest",
    "compare",
    "construct",
    "default_registry",
    "default_universe",
    "fetch_prices",
    "load_universe",
    "strategy_spec_from_dict",
]
