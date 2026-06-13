"""Downside semicovariance estimator (BUILD_PLAN §4, Agent 3).

Captures *downside* co-movement only. Returns are floored against a benchmark
``B`` (default 0): ``drops = min(r − B, 0)``; the semicovariance is
``S = (dropsᵀ drops) / T · frequency`` (annualized). This is the covariance of the
negative excursions and feeds downside-aware risk budgeting. The output is routed
through the §12.3 PSD fix like every estimator here.
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


class SemiCovariance:
    """Annualized downside semicovariance estimator (implements ``RiskModel``).

    Parameters
    ----------
    benchmark:
        Per-period threshold ``B`` below which returns count as a "drop"
        (BUILD_PLAN §4). Defaults to ``0.0``. Often set to the per-period
        risk-free rate or the asset mean (mean-semivariance).
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    ddof:
        Delta degrees of freedom for the ``1/(T − ddof)`` normalization. The
        downside literature divides by ``T`` (``ddof=0``, the default) so that
        the semicovariance of the full sample is comparable across assets.
    """

    def __init__(
        self,
        *,
        benchmark: float = 0.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
        ddof: int = 0,
    ) -> None:
        bench = float(benchmark)
        if not np.isfinite(bench):
            raise RiskModelError("benchmark must be finite.")
        self.benchmark = bench
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        if ddof not in (0, 1):
            raise RiskModelError("ddof must be 0 or 1.")
        self.ddof = ddof

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized downside semicovariance, ordered to ``returns.assets``."""
        values = returns_array(returns, min_periods=self.ddof + 1)
        n_periods, n_assets = values.shape
        drops = np.minimum(values - self.benchmark, 0.0)
        denom = n_periods - self.ddof
        if denom <= 0:
            raise RiskModelError("Not enough periods for the requested ddof.")
        cov = (drops.T @ drops) / denom
        cov = cov * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


def semi_covariance(
    *,
    benchmark: float = 0.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = 0,
) -> SemiCovariance:
    """Factory for the ``"semicov"`` risk model (BUILD_PLAN §5.2)."""
    return SemiCovariance(benchmark=benchmark, periods_per_year=periods_per_year, ddof=ddof)


__all__ = ["SemiCovariance", "semi_covariance"]
