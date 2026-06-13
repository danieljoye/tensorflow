"""Walk-forward backtesting (BUILD_PLAN §4, §5, §7, Agent 5).

The :class:`~riskbudget.backtest.engine.WalkForwardBacktester` implements the
:class:`~riskbudget.core.interfaces.Backtester` protocol: it steps through a
:class:`~riskbudget.core.interfaces.RebalanceSchedule`, estimates the risk model
from a **trailing lookback window only** (no look-ahead), solves weights via any
:class:`~riskbudget.core.interfaces.PortfolioConstructor` /
:class:`~riskbudget.core.interfaces.Optimizer` passed in (ERC, GMV, MSR,
equal-weight, …), holds to the next rebalance, and accrues net-of-cost returns
into a :class:`~riskbudget.core.types.BacktestResult`.

Transaction costs live in :mod:`riskbudget.backtest.costs` — a proportional
(bps-on-turnover) model charged at each rebalance.

Public factory callables follow the §5.2 registry convention:

- Backtester: ``"walkforward"`` (:func:`~riskbudget.backtest.engine.walkforward`).
- Cost models: ``"proportional"``
  (:func:`~riskbudget.backtest.costs.proportional_cost`) and ``"none"``
  (:func:`~riskbudget.backtest.costs.no_cost`).

The :data:`BACKTESTER_FACTORIES` / :data:`COST_MODEL_FACTORIES` maps plus the
:func:`register` hook let Agent 11 assemble its registry from one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from riskbudget.backtest.costs import (
    CostModel,
    ProportionalCost,
    compute_turnover,
    no_cost,
    proportional_cost,
)
from riskbudget.backtest.engine import WalkForwardBacktester, walkforward

# §5.2 name -> factory maps.
BACKTESTER_FACTORIES: dict[str, Callable[..., Any]] = {
    "walkforward": walkforward,
}

COST_MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "proportional": proportional_cost,
    "none": no_cost,
}


def register(registry: Any) -> None:
    """Register this package's factories with the integration registry (§5.2).

    Expects the registry to expose ``register_backtester(name, factory)`` and
    ``register_cost_model(name, factory)``; falls back to a generic
    ``register(kind, name, factory)`` if those are not present. An unknown
    registry shape raises ``AttributeError`` (surfaced to the integrator), never a
    silent no-op.
    """
    for name, factory in BACKTESTER_FACTORIES.items():
        if hasattr(registry, "register_backtester"):
            registry.register_backtester(name, factory)
        else:
            registry.register("backtester", name, factory)
    for name, factory in COST_MODEL_FACTORIES.items():
        if hasattr(registry, "register_cost_model"):
            registry.register_cost_model(name, factory)
        else:
            registry.register("cost_model", name, factory)


__all__ = [
    "BACKTESTER_FACTORIES",
    "COST_MODEL_FACTORIES",
    "CostModel",
    "ProportionalCost",
    "WalkForwardBacktester",
    "compute_turnover",
    "no_cost",
    "proportional_cost",
    "register",
    "walkforward",
]
