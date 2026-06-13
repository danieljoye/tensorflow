"""Nearest-PSD covariance repair (BUILD_PLAN §12.3).

Every covariance estimator in this package routes its output through
:func:`nearest_psd` so that anything which later gets inverted (GMV / MSR /
Black-Litterman) or fed to a QP receives a symmetric positive-semidefinite
matrix. The repair is a spectral eigenvalue clip:

``Σ_psd = V · diag(max(λ, 0)) · Vᵀ`` from ``eigh(Σ)``.

The PSD *test* is a Cholesky factorization of ``Σ + jitter·I`` (cheap, exact);
if it succeeds the matrix is already PSD and is returned untouched (after a
symmetry projection). Reference: BUILD_PLAN §12.3, derived from
``PyPortfolioOpt.risk_models.fix_nonpositive_semidefinite``.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import RiskModelError

# Jitter added to the diagonal for the Cholesky-based PSD test (BUILD_PLAN §12.3).
_PSD_TEST_JITTER = 1e-16


def is_psd(cov: np.ndarray, *, jitter: float = _PSD_TEST_JITTER) -> bool:
    """Return ``True`` if ``cov`` is positive semidefinite.

    The test is a Cholesky factorization of ``cov + jitter·I`` (BUILD_PLAN
    §12.3). A successful factorization certifies PSD-ness up to the jitter.

    Parameters
    ----------
    cov:
        A square, finite, (approximately) symmetric matrix.
    jitter:
        Tiny diagonal loading to make the boundary (singular but PSD) case pass.
    """
    arr = _as_square(cov)
    sym = 0.5 * (arr + arr.T)
    try:
        np.linalg.cholesky(sym + jitter * np.eye(sym.shape[0]))
    except np.linalg.LinAlgError:
        return False
    return True


def nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Return the nearest PSD matrix to ``cov`` via spectral eigenvalue clipping.

    Symmetrizes ``cov`` (averaging it with its transpose), eigen-decomposes the
    result with :func:`numpy.linalg.eigh`, clips negative eigenvalues to zero,
    and reconstructs ``V·diag(max(λ, 0))·Vᵀ`` (BUILD_PLAN §12.3). If the input is
    already PSD it is returned unchanged apart from an exact symmetry projection.

    Parameters
    ----------
    cov:
        A square covariance estimate, possibly indefinite from sampling noise.

    Returns
    -------
    numpy.ndarray
        A symmetric PSD matrix of the same shape, C-contiguous float64.

    Raises
    ------
    RiskModelError
        If ``cov`` is not a finite square 2-D array.
    """
    arr = _as_square(cov)
    sym = 0.5 * (arr + arr.T)

    if is_psd(sym):
        # Already PSD; just return the symmetry-projected copy.
        return np.ascontiguousarray(sym)

    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    repaired = (eigvecs * clipped) @ eigvecs.T
    # Re-symmetrize to wash out asymmetric floating-point error.
    repaired = 0.5 * (repaired + repaired.T)
    return np.ascontiguousarray(repaired)


def _as_square(cov: np.ndarray) -> np.ndarray:
    """Validate and coerce ``cov`` to a finite, square float ndarray."""
    try:
        arr = np.asarray(cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RiskModelError(f"Covariance is not numeric: {exc}") from exc
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise RiskModelError(f"Covariance must be a square 2-D matrix, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise RiskModelError("Covariance matrix is empty.")
    if not np.isfinite(arr).all():
        raise RiskModelError("Covariance contains NaN or infinite values.")
    return arr


__all__ = ["is_psd", "nearest_psd"]
