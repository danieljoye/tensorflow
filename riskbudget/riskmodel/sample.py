"""Sample covariance estimator (BUILD_PLAN §4, Agent 3).

The maximum-likelihood / unbiased sample covariance of asset returns, annualized
by the periods-per-year frequency. The output is routed through the §12.3 PSD
fix like every estimator here.
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


class SampleCovariance:
    """Annualized sample covariance estimator (implements ``RiskModel``).

    Parameters
    ----------
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1); 252 for daily, 52 weekly, 12
        monthly. The covariance scales linearly with frequency.
    ddof:
        Delta degrees of freedom for the sample covariance (``1`` for the
        unbiased estimator, ``0`` for the MLE). Defaults to ``1``.
    """

    def __init__(
        self,
        *,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
        ddof: int = 1,
    ) -> None:
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        if ddof not in (0, 1):
            raise RiskModelError("ddof must be 0 or 1.")
        self.ddof = ddof

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized sample covariance, ordered to ``returns.assets``."""
        values = returns_array(returns, min_periods=self.ddof + 1)
        n_assets = values.shape[1]
        # rowvar=False => columns are variables (assets).
        cov = np.cov(values, rowvar=False, ddof=self.ddof)
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.shape != (n_assets, n_assets):
            # Single-asset edge case: np.cov returns a scalar.
            cov = cov.reshape(n_assets, n_assets)
        cov = cov * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


def sample_covariance(
    *,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = 1,
) -> SampleCovariance:
    """Factory for the ``"sample"`` risk model (BUILD_PLAN §5.2)."""
    return SampleCovariance(periods_per_year=periods_per_year, ddof=ddof)


__all__ = ["SampleCovariance", "sample_covariance"]
