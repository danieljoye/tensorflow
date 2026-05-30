"""Frozen interface contracts for the risk-budgeting pipeline (BUILD_PLAN §5).

Every downstream agent codes against the :class:`~typing.Protocol` definitions
here, never against another agent's concrete classes. The protocols are
``runtime_checkable`` so tests can assert ``isinstance(impl, RiskModel)`` etc.

This module also defines two small configuration value types that appear in the
protocol signatures:

- :class:`Constraints` — feasibility limits handed to an :class:`Optimizer`
  (long-only, leverage target, per-asset and per-group caps, turnover cap).
- :class:`RebalanceSchedule` — when the :class:`Backtester` re-solves weights and
  how much history each estimate may look back over.

Conventions
-----------
- Covariance matrices are plain ``numpy.ndarray`` of shape ``(N, N)`` ordered to
  match an explicit asset-id list (the optimizer reconciles ordering via
  :meth:`RiskBudget.as_array` / :meth:`Portfolio.as_array`).
- Dates are :class:`datetime.date` at the public boundary; engines may use a
  richer pandas index internally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from riskbudget.core.errors import ConfigurationError
from riskbudget.core.types import (
    AllocatorParams,
    BacktestResult,
    ExpectedReturns,
    Portfolio,
    PriceData,
    ReturnMatrix,
    RiskBudget,
)

# How rebalance dates are spaced. ``"none"`` means a single solve at the start
# (buy-and-hold of the initial weights).
RebalanceFrequency = Literal[
    "none",
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "annual",
]


# ---------------------------------------------------------------------------
# Configuration value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """Feasibility limits applied by an :class:`Optimizer`.

    All fields are optional; the default is a fully invested, long-only book
    (``leverage == 1.0``). Optimizers must honor every field that is set and may
    raise :class:`~riskbudget.core.errors.OptimizationError` if the set is
    infeasible.

    Attributes
    ----------
    long_only:
        If ``True`` (default) all weights must be ``>= 0``.
    leverage:
        Target gross exposure (sum of absolute weights). ``1.0`` is fully
        invested. ``None`` leaves gross exposure unconstrained (the solver still
        normalizes to its own convention).
    min_weight, max_weight:
        Optional per-asset bounds applied to every asset. ``min_weight`` is
        ignored for assets when ``long_only`` already implies a tighter bound.
    group_caps:
        Optional mapping ``group_label -> max combined absolute weight`` for a
        group of assets. The grouping itself is supplied via :attr:`groups`.
    groups:
        Optional mapping ``asset_id -> group_label`` used to evaluate
        :attr:`group_caps`.
    max_turnover:
        Optional cap on one-period turnover (sum of absolute weight changes from
        the previous portfolio) applied during backtest rebalances. ``None``
        means unconstrained.

    Raises
    ------
    ConfigurationError
        If numeric limits are negative/non-finite or inconsistent.
    """

    long_only: bool = True
    leverage: float | None = 1.0
    min_weight: float | None = None
    max_weight: float | None = None
    group_caps: Mapping[str, float] = field(default_factory=dict)
    groups: Mapping[str, str] = field(default_factory=dict)
    max_turnover: float | None = None

    def __post_init__(self) -> None:
        if self.leverage is not None and (not np.isfinite(self.leverage) or self.leverage <= 0):
            raise ConfigurationError("Constraints.leverage must be finite and positive.")
        if self.max_turnover is not None and (
            not np.isfinite(self.max_turnover) or self.max_turnover < 0
        ):
            raise ConfigurationError("Constraints.max_turnover must be finite and non-negative.")
        if (
            self.min_weight is not None
            and self.max_weight is not None
            and self.min_weight > self.max_weight
        ):
            raise ConfigurationError("Constraints.min_weight cannot exceed max_weight.")
        for label, cap in self.group_caps.items():
            if not np.isfinite(cap) or cap < 0:
                raise ConfigurationError(
                    f"Constraints.group_caps[{label!r}] must be finite and non-negative."
                )

    @classmethod
    def long_only_fully_invested(cls) -> Constraints:
        """The default contract: long-only, gross (and net) exposure of 1.0."""
        return cls(long_only=True, leverage=1.0)


@dataclass(frozen=True)
class RebalanceSchedule:
    """When the backtester re-solves weights and how far it may look back.

    Attributes
    ----------
    frequency:
        Spacing of rebalance dates (see :data:`RebalanceFrequency`).
    lookback:
        Number of return periods each risk-model estimate may use. The engine
        must use only data strictly *before* the rebalance date (no look-ahead).
        ``None`` means "use all history available up to the rebalance date".
    min_lookback:
        Minimum number of return periods required before the first rebalance can
        occur. Defaults to :attr:`lookback` when set, else ``2``.
    start, end:
        Optional inclusive bounds on the backtest window. ``None`` defers to the
        supplied price data's own range.

    Raises
    ------
    ConfigurationError
        If ``lookback``/``min_lookback`` are non-positive or ``start > end``.
    """

    frequency: RebalanceFrequency = "monthly"
    lookback: int | None = None
    min_lookback: int | None = None
    start: date | None = None
    end: date | None = None

    def __post_init__(self) -> None:
        if self.lookback is not None and self.lookback < 1:
            raise ConfigurationError("RebalanceSchedule.lookback must be >= 1 when set.")
        if self.min_lookback is not None and self.min_lookback < 1:
            raise ConfigurationError("RebalanceSchedule.min_lookback must be >= 1 when set.")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ConfigurationError("RebalanceSchedule.start must not be after end.")

    @property
    def effective_min_lookback(self) -> int:
        """Resolved minimum history required before the first solve."""
        if self.min_lookback is not None:
            return self.min_lookback
        if self.lookback is not None:
            return self.lookback
        return 2


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DataSource(Protocol):
    """Supplies price panels for a set of assets over a date range."""

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a :class:`PriceData` panel for ``assets`` over ``[start, end]``.

        Implementations should raise :class:`~riskbudget.core.errors.DataError`
        when data cannot be sourced, parsed, or aligned across the requested
        assets.
        """
        ...


@runtime_checkable
class RiskModel(Protocol):
    """Estimates an asset-return covariance matrix."""

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return a symmetric PSD covariance matrix of shape ``(N, N)``.

        The covariance is ordered to match ``returns.assets``. Implementations
        should raise :class:`~riskbudget.core.errors.RiskModelError` if a usable
        estimate cannot be produced.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Solves for portfolio weights that match a target risk budget."""

    def solve(
        self,
        cov: np.ndarray,
        budget: RiskBudget,
        constraints: Constraints,
    ) -> Portfolio:
        """Return a :class:`Portfolio` whose risk contributions match ``budget``.

        ``cov`` must be ordered to match ``budget.assets``. Implementations
        should raise :class:`~riskbudget.core.errors.OptimizationError` when the
        problem is infeasible or the solver does not converge.
        """
        ...


@runtime_checkable
class Backtester(Protocol):
    """Runs a walk-forward backtest of a risk-budgeted strategy."""

    def run(
        self,
        data: PriceData,
        model: RiskModel,
        optimizer: Optimizer,
        budget: RiskBudget,
        schedule: RebalanceSchedule,
    ) -> BacktestResult:
        """Walk ``data`` forward, re-solving on ``schedule``, returning results.

        The engine must estimate the risk model using only data available before
        each rebalance date (no look-ahead). It should raise
        :class:`~riskbudget.core.errors.BacktestError` on unrecoverable failures.
        """
        ...


@runtime_checkable
class MeanModel(Protocol):
    """Estimates expected (annualized) asset returns ``μ``.

    The additive counterpart to :class:`RiskModel` (BUILD_PLAN §5.1): a
    :class:`RiskModel` produces a covariance and a :class:`MeanModel` produces
    the expected-returns vector the classical MSR / efficient-frontier
    constructors need. Implementations include historical-mean, EWMA, and the
    risk-based proxy ``μ`` of the Efficient-MSR benchmark (Agent 3,
    ``riskmodel/returns_model.py``).
    """

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return an :class:`ExpectedReturns` vector ordered to match ``returns.assets``.

        Implementations should raise
        :class:`~riskbudget.core.errors.RiskModelError` if a usable estimate
        cannot be produced.
        """
        ...


@runtime_checkable
class PortfolioConstructor(Protocol):
    """Constructs portfolio weights from a covariance and optional ``μ`` / budget.

    The umbrella interface (BUILD_PLAN §5.1) that ERC, GMV, MSR, and
    equal-weight all implement, so the backtester can run any method through one
    contract. Which keyword inputs are required depends on the method: GMV needs
    only ``cov``; MSR / efficient-frontier need ``mu``; the risk-budget (ERC)
    path needs ``budget``. The existing :class:`Optimizer` is retained for the
    pure risk-budget path; this generalizes it without changing that signature.
    """

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        """Return a :class:`Portfolio` for ``cov`` under ``constraints``.

        ``cov`` (and ``mu`` / ``budget`` when supplied) must share one asset
        ordering; alignment is reconciled via :meth:`ExpectedReturns.as_array`,
        :meth:`RiskBudget.as_array`, and :meth:`Portfolio.as_array`.
        Implementations should raise
        :class:`~riskbudget.core.errors.OptimizationError` when the problem is
        infeasible, the solver does not converge, or a required input
        (``mu``/``budget``) for the chosen method is missing.
        """
        ...


@runtime_checkable
class Allocator(Protocol):
    """Allocates a risk budget over *time* between a risky and a safe asset.

    The dynamic / temporal sense of risk budgeting (BUILD_PLAN §2, §5.1): CPPI
    and the drawdown/floor strategies (Agent 8, ``dynamic/``). ``safe`` may be
    an explicit return series or a constant rate; allocation behaviour is driven
    by :class:`AllocatorParams`.
    """

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Run the dynamic allocation, returning a :class:`BacktestResult`.

        ``safe`` is either a :class:`ReturnMatrix` of safe-asset returns aligned
        to ``risky`` or a scalar per-period/annualized rate (the allocator
        documents which); when omitted as a series it falls back to
        ``params.safe_rate``. Implementations should raise
        :class:`~riskbudget.core.errors.BacktestError` on unrecoverable
        failures.
        """
        ...


__all__ = [
    "Allocator",
    "Backtester",
    "Constraints",
    "DataSource",
    "MeanModel",
    "Optimizer",
    "PortfolioConstructor",
    "RebalanceFrequency",
    "RebalanceSchedule",
    "RiskModel",
]
