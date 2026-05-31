"""Exponentially-weighted covariance estimator (BUILD_PLAN §4, Agent 3).

``exp_cov``: weights recent observations more heavily via an exponential decay
parameterized by a ``span`` (default ≈ 180, EDHEC convention). The decay factor
is ``α = 2 / (span + 1)`` and observation ``t`` (0 = oldest) carries weight
``(1 − α)^(T − 1 − t)``. The weighted covariance is annualized and PSD-repaired.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.core.types import ReturnMatrix
from riskbudget.riskmodel._common import (
    DEFAULT_PERIODS_PER_YEAR,
    finalize_cov,
    returns_array,
    validate_periods_per_year,
)

# EDHEC / PyPortfolioOpt default span for exponential covariance.
DEFAULT_SPAN = 180


def _ewma_weights(n_periods: int, span: float) -> np.ndarray:
    """Normalized exponential weights, newest observation heaviest."""
    alpha = 2.0 / (span + 1.0)
    # Ages: oldest row has the largest exponent, newest has exponent 0.
    ages = np.arange(n_periods - 1, -1, -1, dtype=float)
    weights = (1.0 - alpha) ** ages
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        raise RiskModelError("EWMA weights degenerated; check span/period count.")
    normalized: np.ndarray = weights / total
    return normalized


class EWMACovariance:
    """Exponentially-weighted covariance estimator (implements ``RiskModel``).

    Parameters
    ----------
    span:
        Exponential span controlling the decay (``α = 2/(span+1)``). Larger spans
        approach the equally-weighted sample covariance.
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    demean:
        If ``True`` (default) subtract the weighted mean before forming the
        covariance; if ``False`` use raw second moments about zero.
    """

    def __init__(
        self,
        *,
        span: int | float = DEFAULT_SPAN,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
        demean: bool = True,
    ) -> None:
        span_f = float(span)
        if not np.isfinite(span_f) or span_f <= 0:
            raise RiskModelError("span must be finite and positive.")
        self.span = span_f
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        self.demean = demean

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized exponentially-weighted covariance."""
        values = returns_array(returns, min_periods=2)
        n_periods, n_assets = values.shape
        weights = _ewma_weights(n_periods, self.span)

        if self.demean:
            mean = weights @ values  # weighted mean per asset
            centered = values - mean
        else:
            centered = values

        # Weighted covariance: Σ_t w_t · x_t x_tᵀ (weights sum to 1).
        weighted = centered * weights[:, None]
        cov = centered.T @ weighted
        cov = cov * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


def ewma_covariance(
    *,
    span: int | float = DEFAULT_SPAN,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    demean: bool = True,
) -> EWMACovariance:
    """Factory for the ``"ewma"`` risk model (BUILD_PLAN §5.2)."""
    return EWMACovariance(span=span, periods_per_year=periods_per_year, demean=demean)


__all__ = ["DEFAULT_SPAN", "EWMACovariance", "ewma_covariance"]
