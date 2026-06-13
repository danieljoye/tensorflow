"""Diversification metrics: Diversification Ratio and Effective Number of Bets.

These are pure functions over ``weights`` and a covariance ``Σ`` (BUILD_PLAN §12.5).

Sources
-------
- **Diversification Ratio** ``DR(w) = (wᵀσ) / √(wᵀΣw)`` — Choueifaty & Coignard,
  "Toward Maximum Diversification" (*Journal of Portfolio Management*, 2008).
- **Effective Number of Bets** ``ENB = exp(−Σ pᵢ ln pᵢ)`` over the diversification
  distribution ``p`` on an uncorrelated factor basis — Meucci, "Managing
  Diversification" (*Risk*, 2009); factor basis from Meucci, Santangelo & Deguest
  (min-torsion, SSRN 2276632).

The diversification distribution uses the torsion matrix ``t`` (the asset→factor
transform, see :mod:`riskbudget.diversification.torsion`)::

    p = (tᵀ)⁻¹ b ⊙ (t · Σ · b) / (bᵀ Σ b)

with ``b`` the portfolio weights. By construction ``Σ pᵢ = 1``, so ``p`` is a
probability distribution and ``ENB = exp(entropy(p))`` lies in ``[1, N]``.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.diversification.torsion import TorsionMethod, torsion

# Guard threshold from BUILD_PLAN §12.5: only take ln(pᵢ) for pᵢ above this.
_GUARD_THRESHOLD = 1e-5


def _as_vectors(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce ``weights`` and ``cov`` to aligned float arrays."""
    w = np.asarray(weights, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise RiskModelError(f"Covariance must be square, got shape {sigma.shape}.")
    if w.size != sigma.shape[0]:
        raise RiskModelError(
            f"Weights length {w.size} does not match covariance dimension {sigma.shape[0]}."
        )
    if not np.isfinite(w).all():
        raise RiskModelError("Weights contain NaN or infinite values.")
    if not np.isfinite(sigma).all():
        raise RiskModelError("Covariance contains NaN or infinite values.")
    return w, sigma


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Choueifaty–Coignard Diversification Ratio ``DR(w) = (wᵀσ) / √(wᵀΣw)``.

    ``σ`` is the vector of asset volatilities (``√diag(Σ)``). ``DR ≥ 1`` always
    (equality iff the portfolio is a single asset or all assets are perfectly
    correlated); a higher ratio means more diversification benefit.

    Parameters
    ----------
    weights:
        Portfolio weights ``w`` of length ``N`` (need not be normalised — ``DR``
        is scale-invariant in ``w``).
    cov:
        Symmetric PSD covariance matrix ``(N, N)``.
    """
    w, sigma = _as_vectors(weights, cov)
    asset_vol = np.sqrt(np.diag(sigma))
    weighted_vol = float(np.abs(w) @ asset_vol)
    var = float(w @ sigma @ w)
    if var <= 0:
        raise RiskModelError("Portfolio variance is non-positive; cannot form DR.")
    return weighted_vol / float(np.sqrt(var))


def diversification_distribution(
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    method: TorsionMethod = "minimum-torsion",
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Meucci diversification distribution ``p`` on an uncorrelated factor basis.

    ``p = (tᵀ)⁻¹ b ⊙ (t · Σ · b) / (bᵀ Σ b)`` where ``t`` is the torsion matrix
    (asset→factor) and ``b`` the weights. Sums to one by construction.

    Parameters
    ----------
    weights:
        Portfolio weights ``b``.
    cov:
        Covariance ``Σ``.
    method:
        Torsion basis used when ``t`` is not supplied: ``"minimum-torsion"``
        (default), ``"pca"``, or ``"approximate"``.
    t:
        Optional pre-computed torsion matrix (overrides ``method``); useful to
        avoid recomputing the basis across repeated metric evaluations.
    """
    b, sigma = _as_vectors(weights, cov)
    t_mat = torsion(sigma, method) if t is None else np.asarray(t, dtype=float)
    if t_mat.shape != sigma.shape:
        raise RiskModelError("Torsion matrix shape does not match covariance.")

    portfolio_var = float(b @ sigma @ b)
    if portfolio_var <= 0:
        raise RiskModelError("Portfolio variance is non-positive; cannot form ENB distribution.")

    # (tᵀ)⁻¹ b — solve tᵀ x = b rather than inverting.
    try:
        t_inv_t_b = np.linalg.solve(t_mat.T, b)
    except np.linalg.LinAlgError as exc:
        raise RiskModelError("Torsion matrix is singular; cannot form ENB distribution.") from exc

    p = t_inv_t_b * (t_mat @ sigma @ b) / portfolio_var
    return np.asarray(p, dtype=float)


def _entropy_guarded(p: np.ndarray) -> float:
    """Shannon entropy with the §12.5 guard ``pᵢ ln(1 + (pᵢ−1)·[pᵢ>1e-5])``.

    The indicator ``[pᵢ > 1e-5]`` zeroes the log term for tiny/negative ``pᵢ``,
    avoiding ``ln(0)`` and ``ln(<0)`` while leaving the dominant mass untouched.
    """
    indicator = (p > _GUARD_THRESHOLD).astype(float)
    return -float(np.sum(p * np.log(1.0 + (p - 1.0) * indicator)))


def effective_number_of_bets(
    weights: np.ndarray,
    cov: np.ndarray,
    *,
    method: TorsionMethod = "minimum-torsion",
    t: np.ndarray | None = None,
) -> float:
    """Effective Number of Bets ``ENB = exp(−Σ pᵢ ln pᵢ)`` (Meucci).

    Computes the diversification distribution ``p`` on the chosen uncorrelated
    factor basis and returns ``exp`` of its (guarded) Shannon entropy. ``ENB``
    ranges in ``[1, N]``: ``≈ N`` for ``1/N`` over ``N`` uncorrelated unit-variance
    factors, ``≈ 1`` for a fully concentrated (or fully correlated) book.

    Parameters
    ----------
    weights:
        Portfolio weights ``b``.
    cov:
        Covariance ``Σ``.
    method:
        Factor basis (default ``"minimum-torsion"``); see
        :func:`diversification_distribution`.
    t:
        Optional pre-computed torsion matrix.
    """
    p = diversification_distribution(weights, cov, method=method, t=t)
    return float(np.exp(_entropy_guarded(p)))


__all__ = [
    "diversification_distribution",
    "diversification_ratio",
    "effective_number_of_bets",
]
