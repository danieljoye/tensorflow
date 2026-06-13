"""Minimum-torsion (and PCA) factor transforms (BUILD_PLAN §12.4).

The *minimum-torsion transform* turns a set of (generally correlated) assets into
a set of uncorrelated factors that are as close as possible — in a tracking-error
sense — to the original assets. It is the stable alternative to raw PCA factors,
whose sign/ordering "spin" as the covariance shifts.

Source
------
Meucci, Santangelo & Deguest, "Risk Budgeting and Diversification Based on
Optimized Uncorrelated Factors" (working paper, 2015), SSRN 2276632. The polar
fixed-point iteration here mirrors the reference ``reckziegel/uncorbets``
implementation: the matrix square root used is the **symmetric** (eigendecomposition)
root of the correlation matrix, *not* the Cholesky factor.

Definitions
-----------
For a covariance ``Σ`` with volatilities ``s = √diag(Σ)`` and correlation ``C``:

- ``pca`` torsion: ``t = diag(s) · E · diag(1/s)`` where ``E`` are the (sorted)
  eigenvectors of ``C`` — i.e. the principal-component factors rescaled back to
  return space.
- ``minimum-torsion`` transform: the polar fixed-point iteration on ``c = sqrtm(C)``
  (symmetric root) yielding ``t = diag(s) · (π · c⁻¹) · diag(1/s)``.
- ``approximate`` one-shot minimum torsion: ``t = diag(s) · C^{-1/2} · diag(1/s)``.

In every case the factor returns are ``f = t · r`` (for asset returns ``r``), and
the factor covariance ``t · Σ · tᵀ`` is diagonal (uncorrelated factors).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import eigh

from riskbudget.core.errors import RiskModelError

TorsionMethod = Literal["minimum-torsion", "pca", "approximate"]

# Convergence / numerical tolerances (BUILD_PLAN §12.4).
_CONVERGENCE_TOL = 1e-8
_MAX_ITER = 10_000
_EIG_FLOOR = 1e-15


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    """Coerce, square-check, symmetrize-check a covariance matrix."""
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise RiskModelError(f"Covariance must be a square matrix, got shape {sigma.shape}.")
    if not np.isfinite(sigma).all():
        raise RiskModelError("Covariance contains NaN or infinite values.")
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-10):
        raise RiskModelError("Covariance matrix is not symmetric.")
    return sigma


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a covariance into volatilities ``s`` and correlation ``C``.

    Returns ``(s, C)`` with ``s = √diag(Σ)`` and ``C = diag(1/s)·Σ·diag(1/s)``.
    """
    sigma = _validate_cov(cov)
    variances = np.diag(sigma)
    if np.any(variances <= 0):
        raise RiskModelError("Covariance has a non-positive diagonal; cannot form correlation.")
    s = np.sqrt(variances)
    inv_s = 1.0 / s
    corr = inv_s[:, None] * sigma * inv_s[None, :]
    # Symmetrize away tiny float drift and pin the unit diagonal.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return s, corr


def sqrtm_sym(matrix: np.ndarray) -> np.ndarray:
    """Symmetric (eigendecomposition) matrix square root of a symmetric PSD matrix.

    This is the root mandated by BUILD_PLAN §12.4 — **not** the Cholesky factor.
    Negative eigenvalues (numerical noise on a borderline-PSD matrix) are floored
    at zero before taking the root.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = eigh(sym)
    eigvals = np.clip(eigvals, 0.0, None)
    return np.asarray((eigvecs * np.sqrt(eigvals)) @ eigvecs.T, dtype=float)


def _inv_sqrtm_sym(matrix: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root ``M^{-1/2}`` of a symmetric PD matrix."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = eigh(sym)
    if np.any(eigvals <= _EIG_FLOOR):
        raise RiskModelError("Correlation matrix is singular; cannot form inverse square root.")
    return np.asarray((eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T, dtype=float)


def pca_torsion(cov: np.ndarray) -> np.ndarray:
    """PCA torsion ``t`` whose factors are the (rescaled) principal components.

    Factors ``f = t·r`` are the eigenvector portfolios of the correlation matrix,
    rescaled back to return space by the asset volatilities. They are uncorrelated
    but unstable (sign/order flip as ``Σ`` shifts) — hence the minimum-torsion
    alternative.
    """
    s, corr = cov_to_corr(cov)
    eigvals, eigvecs = eigh(corr)
    # Sort by descending eigenvalue for a deterministic factor order.
    order = np.argsort(eigvals)[::-1]
    e = eigvecs[:, order]
    return np.asarray(np.diag(s) @ e.T @ np.diag(1.0 / s), dtype=float)


def approximate_torsion(cov: np.ndarray) -> np.ndarray:
    """One-shot approximate minimum torsion ``t = diag(s)·C^{-1/2}·diag(1/s)``."""
    s, corr = cov_to_corr(cov)
    inv_sqrt_c = _inv_sqrtm_sym(corr)
    return np.asarray(np.diag(s) @ inv_sqrt_c @ np.diag(1.0 / s), dtype=float)


def minimum_torsion(
    cov: np.ndarray,
    *,
    max_iter: int = _MAX_ITER,
    tol: float = _CONVERGENCE_TOL,
) -> np.ndarray:
    """Minimum-torsion transform via the polar fixed-point iteration (§12.4).

    Let ``s = √diag(Σ)``, ``C`` the correlation, and ``c = sqrtm(C)`` the
    **symmetric** root. Initialise ``d = 1`` and iterate::

        U = diag(d)·C·diag(d)
        u = sqrtm(U)
        q = u⁻¹·diag(d)·c
        d = diag(q·c)
        π = diag(d)·q

    stopping when ``|Δ‖c−π‖_F| / ‖c−π‖_F / n ≤ tol``. The torsion matrix is
    ``t = diag(s)·(π·c⁻¹)·diag(1/s)``.

    Parameters
    ----------
    cov:
        Symmetric PSD covariance matrix ``(N, N)``.
    max_iter:
        Maximum fixed-point iterations before raising ``RiskModelError``.
    tol:
        Convergence tolerance on the normalised Frobenius-norm change.

    Returns
    -------
    np.ndarray
        The torsion matrix ``t`` of shape ``(N, N)``; factors ``f = t·r`` are
        uncorrelated (``t·Σ·tᵀ`` diagonal) and minimally distant from the assets.
    """
    s, corr = cov_to_corr(cov)
    n = corr.shape[0]

    c = sqrtm_sym(corr)
    try:
        c_inv = np.linalg.inv(c)
    except np.linalg.LinAlgError as exc:
        raise RiskModelError("Correlation square root is singular.") from exc

    d = np.ones(n)
    f_prev = np.inf
    pi = c.copy()

    for _ in range(max_iter):
        u_mat = (d[:, None] * corr) * d[None, :]
        u = sqrtm_sym(u_mat)
        try:
            u_inv = np.linalg.inv(u)
        except np.linalg.LinAlgError as exc:
            raise RiskModelError("Polar iteration produced a singular factor.") from exc
        q = u_inv @ (d[:, None] * c)
        d = np.diag(q @ c).copy()
        pi = d[:, None] * q

        f_norm = float(np.linalg.norm(c - pi, "fro"))
        if abs(f_norm - f_prev) / max(f_norm, _EIG_FLOOR) / n <= tol:
            f_prev = f_norm
            break
        f_prev = f_norm
    else:
        raise RiskModelError(
            f"Minimum-torsion polar iteration did not converge in {max_iter} iterations."
        )

    return np.asarray(np.diag(s) @ (pi @ c_inv) @ np.diag(1.0 / s), dtype=float)


def torsion(cov: np.ndarray, method: TorsionMethod = "minimum-torsion") -> np.ndarray:
    """Dispatch to the requested torsion transform.

    Parameters
    ----------
    cov:
        Symmetric PSD covariance matrix.
    method:
        One of ``"minimum-torsion"`` (default), ``"pca"``, or ``"approximate"``.
    """
    if method == "minimum-torsion":
        return minimum_torsion(cov)
    if method == "pca":
        return pca_torsion(cov)
    if method == "approximate":
        return approximate_torsion(cov)
    raise RiskModelError(
        f"Unknown torsion method {method!r}; expected 'minimum-torsion', 'pca', or 'approximate'."
    )


__all__ = [
    "TorsionMethod",
    "approximate_torsion",
    "cov_to_corr",
    "minimum_torsion",
    "pca_torsion",
    "sqrtm_sym",
    "torsion",
]
