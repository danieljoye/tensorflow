"""Expected-return estimators (BUILD_PLAN §4, §5.1, Agent 3).

Implementations of the :class:`~riskbudget.core.interfaces.MeanModel` protocol —
the ``μ`` vector the classical MSR / efficient-frontier / Efficient-MSR
constructors (Agent 4) need but the pure risk-budget path does not. All estimates
are **annualized** and returned as :class:`~riskbudget.core.types.ExpectedReturns`
aligned to ``returns.assets``.

Estimators
----------
- :class:`HistoricalMeanReturns` — sample mean, ``arithmetic`` (``mean·freq``) or
  ``geometric`` / CAGR (``(1+r).prod()**(freq/T) − 1``).
- :class:`EWMAMeanReturns` — exponentially-weighted mean (recent-heavy), annualized
  arithmetically.
- :class:`CAPMReturns` — market-implied returns ``μ = r_f + β·(E[R_m] − r_f)``
  with ``β`` from a market-augmented covariance; an equal-weight proxy market when
  no benchmark column is supplied.
- :class:`RiskBasedReturns` — Martellini (2008) risk-based proxy: expected return
  proportional to total volatility (or semi-deviation), scaled to a target mean.
  Avoids the unreliable sample mean and feeds the Efficient-MSR benchmark.

References: BUILD_PLAN §4; Martellini, "Toward the Design of Better Equity
Benchmarks," JPM 2008 (§11).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.core.types import ExpectedReturns, ReturnMatrix
from riskbudget.riskmodel._common import (
    DEFAULT_PERIODS_PER_YEAR,
    returns_array,
    to_expected_returns,
    validate_periods_per_year,
)

MeanMethod = Literal["arithmetic", "geometric"]
RiskProxy = Literal["volatility", "semideviation"]


class HistoricalMeanReturns:
    """Historical-mean expected returns (implements ``MeanModel``).

    Parameters
    ----------
    method:
        ``"arithmetic"`` (default) annualizes the per-period sample mean by
        ``× periods_per_year``; ``"geometric"`` compounds to a CAGR
        ``(∏(1+r))**(periods_per_year/T) − 1``.
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(
        self,
        *,
        method: MeanMethod = "arithmetic",
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if method not in ("arithmetic", "geometric"):
            raise RiskModelError("method must be 'arithmetic' or 'geometric'.")
        self.method = method
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return annualized historical-mean expected returns."""
        values = returns_array(returns, min_periods=1)
        if self.method == "arithmetic":
            mu = values.mean(axis=0) * self.periods_per_year
        else:
            t = values.shape[0]
            growth = np.prod(1.0 + values, axis=0)
            if np.any(growth <= 0):
                raise RiskModelError("Geometric mean undefined: cumulative growth is non-positive.")
            mu = growth ** (self.periods_per_year / t) - 1.0
        return to_expected_returns(mu, returns.assets)


class EWMAMeanReturns:
    """Exponentially-weighted mean returns (implements ``MeanModel``).

    Weights recent observations more heavily via ``α = 2/(span+1)``; the weighted
    per-period mean is annualized arithmetically.

    Parameters
    ----------
    span:
        Exponential span controlling the decay.
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(
        self,
        *,
        span: int | float = 180,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        span_f = float(span)
        if not np.isfinite(span_f) or span_f <= 0:
            raise RiskModelError("span must be finite and positive.")
        self.span = span_f
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return annualized EWMA-mean expected returns."""
        values = returns_array(returns, min_periods=1)
        n_periods = values.shape[0]
        alpha = 2.0 / (self.span + 1.0)
        ages = np.arange(n_periods - 1, -1, -1, dtype=float)
        weights = (1.0 - alpha) ** ages
        weights = weights / weights.sum()
        mu = (weights @ values) * self.periods_per_year
        return to_expected_returns(mu, returns.assets)


class CAPMReturns:
    """Market-implied (CAPM) expected returns (implements ``MeanModel``).

    ``μᵢ = r_f + βᵢ·(E[R_m] − r_f)`` with ``βᵢ = cov(rᵢ, r_m)/var(r_m)``. When no
    ``market`` asset id is supplied, an equal-weight portfolio of the universe is
    used as the market proxy (PyPortfolioOpt convention).

    Parameters
    ----------
    risk_free_rate:
        Annualized risk-free rate ``r_f`` (BUILD_PLAN §3.1).
    market:
        Optional asset id to treat as the market portfolio. If ``None`` an
        equal-weight proxy of all assets is used.
    periods_per_year:
        Annualization factor.
    """

    def __init__(
        self,
        *,
        risk_free_rate: float = 0.0,
        market: str | None = None,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if not np.isfinite(risk_free_rate):
            raise RiskModelError("risk_free_rate must be finite.")
        self.risk_free_rate = float(risk_free_rate)
        self.market = market
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return annualized CAPM-implied expected returns."""
        values = returns_array(returns, min_periods=2)
        assets = returns.assets

        if self.market is not None:
            if self.market not in assets:
                raise RiskModelError(f"Market asset {self.market!r} not in universe {assets}.")
            market_ret = values[:, assets.index(self.market)]
        else:
            market_ret = values.mean(axis=1)  # equal-weight proxy

        market_var = float(np.var(market_ret, ddof=1))
        if market_var <= 0:
            raise RiskModelError("Market proxy has zero variance; CAPM beta undefined.")

        x = values - values.mean(axis=0, keepdims=True)
        m = market_ret - market_ret.mean()
        betas = (x.T @ m) / ((values.shape[0] - 1) * market_var)

        market_excess = market_ret.mean() * self.periods_per_year - self.risk_free_rate
        mu = self.risk_free_rate + betas * market_excess
        return to_expected_returns(mu, assets)


class RiskBasedReturns:
    """Risk-based expected returns proxy (Martellini 2008; implements ``MeanModel``).

    Expected return is taken *proportional to* a risk measure — total volatility
    or downside semi-deviation — rather than the unreliable sample mean. The raw
    risk scores are rescaled so their cross-sectional mean equals
    ``target_return`` (annualized), giving a well-behaved ``μ`` for the
    Efficient-MSR / max-Sharpe paths.

    Parameters
    ----------
    proxy:
        ``"volatility"`` (default) or ``"semideviation"`` (downside-only).
    target_return:
        Annualized cross-sectional mean the proxy is scaled to. Defaults to
        ``0.10`` (10%). The *relative* ordering, not the level, is what matters
        downstream.
    periods_per_year:
        Annualization factor.
    """

    def __init__(
        self,
        *,
        proxy: RiskProxy = "volatility",
        target_return: float = 0.10,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if proxy not in ("volatility", "semideviation"):
            raise RiskModelError("proxy must be 'volatility' or 'semideviation'.")
        if not np.isfinite(target_return):
            raise RiskModelError("target_return must be finite.")
        self.proxy = proxy
        self.target_return = float(target_return)
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return annualized risk-based expected returns."""
        values = returns_array(returns, min_periods=2)
        if self.proxy == "volatility":
            risk = values.std(axis=0, ddof=1)
        else:
            drops = np.minimum(values, 0.0)
            # Semi-deviation about zero: sqrt(mean of squared negative returns).
            risk = np.sqrt((drops**2).sum(axis=0) / values.shape[0])
        risk = risk * np.sqrt(self.periods_per_year)

        mean_risk = float(risk.mean())
        if mean_risk <= 0:
            raise RiskModelError("Risk proxy is degenerate (zero average risk).")
        mu = risk / mean_risk * self.target_return
        return to_expected_returns(mu, returns.assets)


def mean_historical(
    *,
    method: MeanMethod = "arithmetic",
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> HistoricalMeanReturns:
    """Factory for the ``"mean_historical"`` mean model (BUILD_PLAN §5.2)."""
    return HistoricalMeanReturns(method=method, periods_per_year=periods_per_year)


def ewma_mean(
    *,
    span: int | float = 180,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> EWMAMeanReturns:
    """Factory for the ``"ewma_mean"`` mean model (BUILD_PLAN §5.2)."""
    return EWMAMeanReturns(span=span, periods_per_year=periods_per_year)


def capm(
    *,
    risk_free_rate: float = 0.0,
    market: str | None = None,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> CAPMReturns:
    """Factory for the ``"capm"`` mean model (BUILD_PLAN §5.2)."""
    return CAPMReturns(
        risk_free_rate=risk_free_rate, market=market, periods_per_year=periods_per_year
    )


def risk_based(
    *,
    proxy: RiskProxy = "volatility",
    target_return: float = 0.10,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> RiskBasedReturns:
    """Factory for the ``"risk_based"`` mean model (BUILD_PLAN §5.2)."""
    return RiskBasedReturns(
        proxy=proxy, target_return=target_return, periods_per_year=periods_per_year
    )


__all__ = [
    "CAPMReturns",
    "EWMAMeanReturns",
    "HistoricalMeanReturns",
    "RiskBasedReturns",
    "capm",
    "ewma_mean",
    "mean_historical",
    "risk_based",
]
