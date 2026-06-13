"""Internal helpers shared across the risk-model estimators.

Not part of the public surface. Centralizes the return-matrix extraction,
annualization conventions (BUILD_PLAN §3.1), and the final PSD routing so every
covariance estimator behaves identically at its boundaries.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.core.types import ExpectedReturns, ReturnMatrix
from riskbudget.riskmodel.psd import nearest_psd

# Default annualization factor: daily data (BUILD_PLAN §3.1).
DEFAULT_PERIODS_PER_YEAR = 252


def validate_periods_per_year(periods_per_year: int | float) -> float:
    """Validate the annualization factor (must be finite and positive)."""
    ppy = float(periods_per_year)
    if not np.isfinite(ppy) or ppy <= 0:
        raise RiskModelError("periods_per_year must be finite and positive.")
    return ppy


def returns_array(returns: ReturnMatrix, *, min_periods: int = 2) -> np.ndarray:
    """Extract the ``(T, N)`` return array, validating it has enough rows.

    Parameters
    ----------
    returns:
        The validated return panel.
    min_periods:
        Minimum number of periods required (estimators need at least 2 for a
        covariance to be defined).

    Raises
    ------
    RiskModelError
        If fewer than ``min_periods`` periods are present.
    """
    if not isinstance(returns, ReturnMatrix):
        raise RiskModelError("Estimators require a ReturnMatrix input.")
    values = np.asarray(returns.values, dtype=float)
    n_periods = values.shape[0]
    if n_periods < min_periods:
        raise RiskModelError(f"Need at least {min_periods} periods to estimate; got {n_periods}.")
    return values


def finalize_cov(cov: np.ndarray, *, n_assets: int) -> np.ndarray:
    """Symmetrize, PSD-repair, and shape-check a covariance estimate.

    Every public covariance estimator passes its raw matrix through here so the
    BUILD_PLAN §12.3 PSD fix is applied uniformly to *every* output.
    """
    sym = 0.5 * (cov + cov.T)
    repaired = nearest_psd(sym)
    if repaired.shape != (n_assets, n_assets):
        raise RiskModelError(f"Covariance shape {repaired.shape} does not match {n_assets} assets.")
    if not np.isfinite(repaired).all():
        raise RiskModelError("Covariance estimate is not finite.")
    return repaired


def to_expected_returns(mu: np.ndarray, assets: list[str]) -> ExpectedReturns:
    """Wrap a ``μ`` vector as :class:`ExpectedReturns` aligned to ``assets``."""
    mu = np.asarray(mu, dtype=float).ravel()
    if mu.shape[0] != len(assets):
        raise RiskModelError(
            f"Expected-returns length {mu.shape[0]} does not match {len(assets)} assets."
        )
    if not np.isfinite(mu).all():
        raise RiskModelError("Expected-returns estimate is not finite.")
    return ExpectedReturns({asset: float(m) for asset, m in zip(assets, mu, strict=True)})


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "finalize_cov",
    "returns_array",
    "to_expected_returns",
    "validate_periods_per_year",
]
