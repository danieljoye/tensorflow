"""Statistical (PCA) factor covariance model (BUILD_PLAN §4, Agent 3).

Reduces estimation noise by approximating the covariance with a low-rank factor
structure plus idiosyncratic variance. The top-``k`` principal components of the
sample covariance are kept as statistical factors; the reconstruction is

``Σ_factor = B·F·Bᵀ + diag(d)``,

where ``B`` are the factor loadings, ``F`` the (diagonal) factor covariance, and
``d`` the idiosyncratic variances chosen so that the *diagonal* of the
reconstruction matches the sample variances exactly (no shrinkage of total
variance, only of the off-diagonal noise beyond the kept factors).

References: BUILD_PLAN §4. Spectral / PCA factor model; every output is routed
through the §12.3 PSD fix.
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


class PCAFactorModel:
    """PCA statistical factor covariance model (implements ``RiskModel``).

    Parameters
    ----------
    n_factors:
        Number of principal components (statistical factors) to retain. Clipped
        to ``[1, N]``. Fewer factors means more aggressive denoising.
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(
        self,
        *,
        n_factors: int = 1,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if n_factors < 1:
            raise RiskModelError("n_factors must be >= 1.")
        self.n_factors = int(n_factors)
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized PCA factor-model covariance."""
        values = returns_array(returns, min_periods=2)
        n_assets = values.shape[1]
        k = min(self.n_factors, n_assets)

        x = values - values.mean(axis=0, keepdims=True)
        sample = (x.T @ x) / (values.shape[0] - 1)

        # eigh: ascending eigenvalues; take the top-k.
        eigvals, eigvecs = np.linalg.eigh(0.5 * (sample + sample.T))
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        idx = np.argsort(eigvals)[::-1][:k]
        top_vals = eigvals[idx]
        top_vecs = eigvecs[:, idx]

        # Low-rank systematic part: V·diag(λ)·Vᵀ.
        systematic = (top_vecs * top_vals) @ top_vecs.T
        # Idiosyncratic variance restores the sample diagonal (non-negative).
        resid = np.diag(sample) - np.diag(systematic)
        resid = np.clip(resid, a_min=0.0, a_max=None)
        reconstructed = systematic + np.diag(resid)

        cov = reconstructed * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


def pca_factor_model(
    *,
    n_factors: int = 1,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> PCAFactorModel:
    """Factory for the ``"pca"`` risk model (BUILD_PLAN §5.2)."""
    return PCAFactorModel(n_factors=n_factors, periods_per_year=periods_per_year)


__all__ = ["PCAFactorModel", "pca_factor_model"]
