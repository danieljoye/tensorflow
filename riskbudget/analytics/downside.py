"""Downside-risk metrics (EDHEC formulations, BUILD_PLAN §2, §11).

All functions take **simple per-period returns** as a :class:`pandas.Series` (or
anything coercible to a 1-D float ndarray). They reproduce the EDHEC course
``edhec_risk_kit`` formulations:

- :func:`drawdown` — wealth index, running peak, and the drawdown series.
- :func:`max_drawdown` — the worst (most negative) drawdown.
- :func:`semideviation` — volatility computed over negative returns only.
- :func:`var_historic` — empirical-percentile Value at Risk.
- :func:`var_gaussian` — parametric Gaussian VaR, optionally **Cornish-Fisher**
  modified for skewness and excess kurtosis.
- :func:`cvar_historic` — historic Conditional VaR / expected shortfall.

VaR/CVaR are returned as **positive numbers** (a loss magnitude), following the
EDHEC sign convention: a 5% VaR of ``0.03`` means "a 3% loss is the worst we
expect to be exceeded only 5% of the time".

Source: EDHEC *Introduction to Portfolio Construction and Analysis with Python*
(Martellini & Vaidyanathan); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskbudget.analytics.distribution import kurtosis, skewness
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


def _check_level(level: float) -> float:
    if not np.isfinite(level) or not (0.0 < level < 1.0):
        raise ValidationError("Confidence level must be in (0, 1).")
    return float(level)


def drawdown(returns: pd.Series | np.ndarray, *, start_value: float = 1.0) -> pd.DataFrame:
    """Wealth index, running peak, and drawdown series (EDHEC ``drawdown``).

    Builds the compounded wealth index ``start_value · ∏(1 + r)``, its running
    peak, and the drawdown ``(wealth − peak) / peak`` (≤ 0). Returns a DataFrame
    with columns ``["wealth", "peak", "drawdown"]``; the index matches the input
    Series index when one is provided, otherwise a default ``RangeIndex``.

    Parameters
    ----------
    returns:
        Per-period simple returns.
    start_value:
        Initial wealth (``> 0``); defaults to ``1.0`` (growth-of-$1).
    """
    if not np.isfinite(start_value) or start_value <= 0:
        raise ValidationError("start_value must be finite and positive.")
    r = _as_returns(returns)
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(r.size)
    wealth = start_value * np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return pd.DataFrame(
        {"wealth": wealth, "peak": peak, "drawdown": dd},
        index=index,
    )


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """The worst (most negative) drawdown over the series.

    Returns a non-positive float; ``0.0`` if the wealth index never falls below
    a prior peak.
    """
    return float(drawdown(returns)["drawdown"].min())


def semideviation(returns: pd.Series | np.ndarray) -> float:
    """Semi-deviation: population std of the negative-return periods only.

    Matches EDHEC ``semideviation`` (``returns[returns < 0].std(ddof=0)``).
    Returns ``0.0`` when there are no negative returns.
    """
    r = _as_returns(returns)
    negatives = r[r < 0.0]
    if negatives.size == 0:
        return 0.0
    return float(np.std(negatives))


def var_historic(returns: pd.Series | np.ndarray, level: float = 0.05) -> float:
    """Historic (empirical-percentile) VaR at confidence ``level``.

    Returns the loss magnitude (a non-negative number): the negative of the
    ``level`` empirical quantile of the returns, matching EDHEC ``var_historic``.
    """
    r = _as_returns(returns)
    lvl = _check_level(level)
    return float(-np.percentile(r, lvl * 100.0))


def var_gaussian(
    returns: pd.Series | np.ndarray,
    level: float = 0.05,
    *,
    modified: bool = True,
) -> float:
    """Parametric Gaussian VaR, optionally Cornish-Fisher modified.

    When ``modified`` is True (the default), the standard-normal quantile
    ``z`` is adjusted for the sample skewness ``s`` and excess kurtosis ``k``
    via the Cornish-Fisher expansion (EDHEC ``var_gaussian``)::

        z ← z + (z²−1)·s/6 + (z³−3z)·(k)/24 − (2z³−5z)·s²/36

    where ``k`` is *excess* kurtosis (so a normal sample leaves ``z``
    unchanged). VaR is then ``−(μ + z·σ)`` (population σ), returned as a loss
    magnitude. On a left-skewed / fat-tailed sample the modified VaR is ``≥`` the
    Gaussian one.
    """
    r = _as_returns(returns)
    lvl = _check_level(level)
    z = float(norm.ppf(lvl))
    if modified:
        s = skewness(r)
        k = kurtosis(r, excess=True)
        z = (
            z
            + (z**2 - 1) * s / 6.0
            + (z**3 - 3 * z) * k / 24.0
            - (2 * z**3 - 5 * z) * (s**2) / 36.0
        )
    return float(-(np.mean(r) + z * np.std(r)))


def cvar_historic(returns: pd.Series | np.ndarray, level: float = 0.05) -> float:
    """Historic Conditional VaR (expected shortfall) at confidence ``level``.

    The mean loss over the periods whose return breaches the historic VaR,
    returned as a positive loss magnitude (EDHEC ``cvar_historic``). Falls back
    to the historic VaR itself if no observation lies strictly beyond it.
    """
    r = _as_returns(returns)
    lvl = _check_level(level)
    threshold = -var_historic(r, lvl)  # return-space cutoff (negative)
    beyond = r[r <= threshold]
    if beyond.size == 0:
        return var_historic(r, lvl)
    return float(-np.mean(beyond))


__all__ = [
    "cvar_historic",
    "drawdown",
    "max_drawdown",
    "semideviation",
    "var_gaussian",
    "var_historic",
]
