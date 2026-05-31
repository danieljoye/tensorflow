"""Risk-contribution decomposition (BUILD_PLAN §2).

For a weight vector ``w`` and covariance ``Σ`` the portfolio volatility is
``σ(w) = sqrt(wᵀ Σ w)``. Euler's theorem decomposes it additively:

- Marginal risk contribution: ``MRCᵢ = (Σ w)ᵢ / σ(w)``
- Total risk contribution:    ``TRCᵢ = wᵢ · MRCᵢ``  with  ``Σᵢ TRCᵢ = σ(w)``
- Percentage contribution:    ``PCRᵢ = TRCᵢ / σ(w)``  with  ``Σᵢ PCRᵢ = 1``

These match :meth:`riskbudget.core.types.Portfolio.risk_contributions` but operate
directly on arrays so the solvers can use them without building a ``Portfolio``.

Source: Maillard, Roncalli & Teïletche (JPM 2010); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskbudget.core.errors import ValidationError

# Treat tiny weights/vols as zero (BUILD_PLAN §3.1).
_ZERO_TOL = 1e-12


def _as_vectors(weights: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and validate ``(w, Σ)`` shapes for contribution math."""
    w = np.asarray(weights, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValidationError(f"Covariance must be square, got shape {sigma.shape}.")
    if sigma.shape[0] != w.size:
        raise ValidationError(f"Covariance shape {sigma.shape} does not match {w.size} weights.")
    if not np.isfinite(w).all():
        raise ValidationError("Weights contain NaN or infinite values.")
    if not np.isfinite(sigma).all():
        raise ValidationError("Covariance contains NaN or infinite values.")
    return w, sigma


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility ``σ(w) = sqrt(wᵀ Σ w)``."""
    w, sigma = _as_vectors(weights, cov)
    var = float(w @ sigma @ w)
    if var < 0:
        raise ValidationError("Negative portfolio variance; covariance is not PSD.")
    return float(np.sqrt(var))


def marginal_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contributions ``MRCᵢ = (Σ w)ᵢ / σ(w)``.

    Returns a zero vector when the portfolio volatility is (numerically) zero.
    """
    w, sigma = _as_vectors(weights, cov)
    vol = portfolio_volatility(w, sigma)
    if vol <= _ZERO_TOL:
        return np.zeros_like(w)
    mrc: np.ndarray = (sigma @ w) / vol
    return mrc


def total_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Total risk contributions ``TRCᵢ = wᵢ · MRCᵢ`` (sums to ``σ(w)``)."""
    w, sigma = _as_vectors(weights, cov)
    trc: np.ndarray = w * marginal_risk_contributions(w, sigma)
    return trc


def percentage_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Percentage contributions ``PCRᵢ = TRCᵢ / σ(w)`` (sums to 1).

    Returns a zero vector when the portfolio volatility is (numerically) zero.
    """
    w, sigma = _as_vectors(weights, cov)
    vol = portfolio_volatility(w, sigma)
    if vol <= _ZERO_TOL:
        return np.zeros_like(w)
    return total_risk_contributions(w, sigma) / vol


@dataclass(frozen=True)
class RiskDecomposition:
    """A full additive risk decomposition of a portfolio.

    Attributes
    ----------
    volatility:
        Portfolio volatility ``σ(w)``.
    marginal:
        Marginal risk contributions ``MRCᵢ``.
    total:
        Total risk contributions ``TRCᵢ`` (sum to :attr:`volatility`).
    percentage:
        Percentage contributions ``PCRᵢ`` (sum to 1).
    """

    volatility: float
    marginal: np.ndarray
    total: np.ndarray
    percentage: np.ndarray

    def verify(self, atol: float = 1e-10) -> bool:
        """Check the additive identity ``Σᵢ TRCᵢ == σ(w)`` within ``atol``."""
        return bool(abs(float(self.total.sum()) - self.volatility) <= atol)


def decompose_risk(weights: np.ndarray, cov: np.ndarray) -> RiskDecomposition:
    """Return the full :class:`RiskDecomposition` for ``(w, Σ)``.

    Verifies the Euler identity ``Σᵢ TRCᵢ = σ(w)`` (the additive decomposition).
    """
    w, sigma = _as_vectors(weights, cov)
    vol = portfolio_volatility(w, sigma)
    mrc = marginal_risk_contributions(w, sigma)
    trc = w * mrc
    pcr = trc / vol if vol > _ZERO_TOL else np.zeros_like(w)
    return RiskDecomposition(volatility=vol, marginal=mrc, total=trc, percentage=pcr)


__all__ = [
    "RiskDecomposition",
    "decompose_risk",
    "marginal_risk_contributions",
    "percentage_risk_contributions",
    "portfolio_volatility",
    "total_risk_contributions",
]
