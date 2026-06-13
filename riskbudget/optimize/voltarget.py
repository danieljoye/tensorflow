"""Volatility-targeting overlay.

A :class:`VolatilityTargetConstructor` wraps *any* portfolio constructor (or
risk-budget optimizer) and scales its weights so the portfolio's **ex-ante
annualized volatility** hits a target. The scale (leverage) is recomputed at each
rebalance from the same trailing-window covariance the inner method used, so it is
strictly point-in-time (no look-ahead):

    sigma_annual = sqrt(wᵀ Σ w)              # Σ is the (annualized) window covariance
    leverage     = min(target / sigma_annual, max_leverage)   # 0 (all cash) if sigma == 0
    w_scaled     = leverage · w

Σ is the covariance the risk model produced for the window, annualized to the run
frequency (the spec's ``periods_per_year`` is propagated to the risk model), so
``sqrt(wᵀ Σ w)`` is already the annualized portfolio volatility.

When ``leverage > 1`` the book is levered (the residual is implicitly financed at
the risk-free rate); when ``leverage < 1`` the residual sits in cash. Because the
inner method is long-only, scaling preserves sign — no shorting is introduced.

BUILD_PLAN §2 (risk budgeting), §3.1 (conventions). Standard risk-parity practice
applies exactly this overlay to bring a low-vol budgeted book up to a usable risk
level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from riskbudget.core.errors import OptimizationError
from riskbudget.core.types import Portfolio

if TYPE_CHECKING:  # pragma: no cover - typing only
    from riskbudget.core.interfaces import Constraints
    from riskbudget.core.types import ExpectedReturns, RiskBudget


class VolatilityTargetConstructor:
    """Scale an inner constructor's weights to a target annualized volatility.

    Parameters
    ----------
    inner:
        Any ``PortfolioConstructor`` (has ``construct``) or risk-budget
        ``Optimizer`` (has ``solve``).
    target_volatility:
        Target **annualized** volatility (e.g. ``0.10`` for 10%). Interpreted in
        the same units as the (annualized) window covariance.
    max_leverage:
        Cap on the gross exposure (default 3.0) to bound leverage when realized
        vol is very low.
    """

    def __init__(
        self,
        inner: object,
        *,
        target_volatility: float,
        max_leverage: float = 3.0,
    ) -> None:
        if target_volatility <= 0:
            raise OptimizationError("target_volatility must be positive.")
        if max_leverage <= 0:
            raise OptimizationError("max_leverage must be positive.")
        self.inner = inner
        self.target_volatility = float(target_volatility)
        self.max_leverage = float(max_leverage)

    def _inner_portfolio(
        self,
        cov: np.ndarray,
        mu: ExpectedReturns | None,
        budget: RiskBudget | None,
        constraints: Constraints,
    ) -> Portfolio:
        inner = self.inner
        if hasattr(inner, "construct"):
            return cast(
                Portfolio, inner.construct(cov, mu=mu, budget=budget, constraints=constraints)
            )
        if hasattr(inner, "solve"):
            return cast(Portfolio, inner.solve(cov, budget, constraints))
        raise OptimizationError(
            "VolatilityTargetConstructor inner object implements neither construct() nor solve()."
        )

    def leverage_for(self, portfolio: Portfolio, cov: np.ndarray) -> float:
        """Ex-ante leverage that brings ``portfolio`` to the target annual vol."""
        sigma_annual = float(portfolio.volatility(np.asarray(cov, dtype=float)))
        if sigma_annual <= 0.0:
            return 0.0
        return min(self.target_volatility / sigma_annual, self.max_leverage)

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        portfolio = self._inner_portfolio(cov, mu, budget, constraints)
        k = self.leverage_for(portfolio, cov)
        return Portfolio({asset: w * k for asset, w in portfolio.weights.items()})
