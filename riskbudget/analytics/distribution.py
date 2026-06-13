"""Distribution diagnostics: skewness, kurtosis, Jarque-Bera (EDHEC, §2, §11).

These reproduce the EDHEC course ``edhec_risk_kit`` definitions, which use the
**population** (biased, ``ddof=0``) standard deviation in the denominator:

- :func:`skewness` — ``E[(r − μ)³] / σ³``.
- :func:`kurtosis` — ``E[(r − μ)⁴] / σ⁴``; ``excess=True`` subtracts 3 so a
  normal distribution returns ~0 (the form the Cornish-Fisher VaR expects).
- :func:`is_normal` — the Jarque-Bera normality test (``scipy.stats.jarque_bera``);
  returns ``True`` when the null of normality is *not* rejected at ``level``.

Source: EDHEC *Introduction to Portfolio Construction and Analysis with Python*
(Martellini & Vaidyanathan); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

from riskbudget.core.errors import ValidationError


def _as_returns(returns: pd.Series | np.ndarray | object, *, name: str = "returns") -> np.ndarray:
    """Coerce ``returns`` to a 1-D finite float ndarray."""
    if isinstance(returns, pd.Series):
        arr = returns.to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        raise ValidationError(f"{name} is empty.")
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} contains NaN or infinite values.")
    return arr


def _standardized_moment(returns: pd.Series | np.ndarray, order: int) -> float:
    """``E[(r − μ)^order] / σ^order`` with population σ (EDHEC convention)."""
    r = _as_returns(returns)
    demeaned = r - r.mean()
    sigma = float(np.std(r))  # population (ddof=0)
    if sigma == 0.0:
        return float("nan")
    return float(np.mean(demeaned**order) / sigma**order)


def skewness(returns: pd.Series | np.ndarray) -> float:
    """Sample skewness ``E[(r − μ)³] / σ³`` (population σ). EDHEC ``skewness``."""
    return _standardized_moment(returns, 3)


def kurtosis(returns: pd.Series | np.ndarray, *, excess: bool = False) -> float:
    """Sample kurtosis ``E[(r − μ)⁴] / σ⁴`` (population σ). EDHEC ``kurtosis``.

    Parameters
    ----------
    excess:
        When True, subtract 3 to return *excess* kurtosis (0 for a normal
        distribution). The Cornish-Fisher VaR adjustment expects the excess
        form. Default False (raw kurtosis, ~3 for a normal).
    """
    raw = _standardized_moment(returns, 4)
    return raw - 3.0 if excess else raw


def is_normal(returns: pd.Series | np.ndarray, level: float = 0.01) -> bool:
    """Jarque-Bera normality test: True if normality is not rejected at ``level``.

    Uses ``scipy.stats.jarque_bera``; the null hypothesis (the data are normally
    distributed) is accepted when the p-value exceeds ``level``. Matches EDHEC
    ``is_normal`` (default ``level=0.01``).
    """
    if not np.isfinite(level) or not (0.0 < level < 1.0):
        raise ValidationError("level must be in (0, 1).")
    r = _as_returns(returns)
    result = scipy.stats.jarque_bera(r)
    p_value = float(result.pvalue)
    return p_value > level


__all__ = [
    "is_normal",
    "kurtosis",
    "skewness",
]
