"""Performance metrics on a periodic-return series (EDHEC formulations).

These follow the EDHEC course ``edhec_risk_kit`` conventions (BUILD_PLAN §3.1,
§11). Inputs are **simple (arithmetic) per-period returns** as a
:class:`pandas.Series` (or anything coercible to a float ndarray), and
``periods_per_year`` defaults to **252** (daily) per §3.1; pass ``52`` for
weekly or ``12`` for monthly data.

Conventions
-----------
- Annualized return is **compound** (geometric): ``(1 + r).prod()`` grossed up
  by ``periods_per_year / n_periods``.
- Annualized volatility scales by ``√periods_per_year``.
- The risk-free rate is annualized and de-annualized per period when forming
  excess returns for Sharpe / Sortino.

Source: EDHEC *Introduction to Portfolio Construction and Analysis with Python*
(Martellini & Vaidyanathan); BUILD_PLAN §3.1, §11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbudget.analytics.downside import max_drawdown, semideviation
from riskbudget.core.errors import ValidationError

# Default annualization factor (daily) — BUILD_PLAN §3.1.
DEFAULT_PERIODS_PER_YEAR = 252


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


def _check_ppy(periods_per_year: int) -> int:
    if periods_per_year <= 0:
        raise ValidationError("periods_per_year must be a positive integer.")
    return int(periods_per_year)


def annualized_return(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Compound (geometric) annualized return.

    ``(∏(1 + rₜ))^(periods_per_year / n) − 1`` for ``n`` periods. This matches
    EDHEC's ``annualize_rets``.
    """
    r = _as_returns(returns)
    ppy = _check_ppy(periods_per_year)
    n = r.size
    growth = float(np.prod(1.0 + r))
    return float(growth ** (ppy / n) - 1.0)


def annualized_volatility(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized volatility ``std(r) · √periods_per_year`` (population std)."""
    r = _as_returns(returns)
    ppy = _check_ppy(periods_per_year)
    return float(np.std(r) * np.sqrt(ppy))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of excess returns (EDHEC ``sharpe_ratio``).

    ``risk_free_rate`` is **annualized**; it is de-annualized to a per-period
    rate ``(1 + rf)^(1/ppy) − 1`` before forming excess returns. The excess
    series is then annualized (return compounded, vol scaled by ``√ppy``).
    Returns ``nan`` when annualized volatility is zero.
    """
    r = _as_returns(returns)
    ppy = _check_ppy(periods_per_year)
    if not np.isfinite(risk_free_rate):
        raise ValidationError("risk_free_rate must be finite.")
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
    excess = r - rf_per_period
    ann_ex_ret = annualized_return(excess, ppy)
    ann_vol = annualized_volatility(r, ppy)
    if ann_vol == 0.0:
        return float("nan")
    return float(ann_ex_ret / ann_vol)


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio: excess return over annualized semideviation.

    Downside deviation uses negative-return periods only (see
    :func:`riskbudget.analytics.downside.semideviation`) and is annualized by
    ``√periods_per_year``. Returns ``nan`` when there is no downside.
    """
    r = _as_returns(returns)
    ppy = _check_ppy(periods_per_year)
    if not np.isfinite(risk_free_rate):
        raise ValidationError("risk_free_rate must be finite.")
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
    excess = r - rf_per_period
    ann_ex_ret = annualized_return(excess, ppy)
    semi = semideviation(r)
    if semi == 0.0:
        return float("nan")
    return float(ann_ex_ret / (semi * np.sqrt(ppy)))


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Calmar ratio: annualized return divided by the absolute max drawdown.

    Returns ``nan`` when the max drawdown is zero (no drawdown observed).
    """
    r = _as_returns(returns)
    ppy = _check_ppy(periods_per_year)
    ann_ret = annualized_return(r, ppy)
    mdd = abs(max_drawdown(r))
    if mdd == 0.0:
        return float("nan")
    return float(ann_ret / mdd)


def hit_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of periods with a strictly positive return."""
    r = _as_returns(returns)
    return float(np.mean(r > 0.0))


def average_turnover(weights: pd.DataFrame) -> float:
    """Average one-way turnover across consecutive rebalances.

    Turnover at a rebalance is ``½ · Σᵢ |wᵢ,ₜ − wᵢ,ₜ₋₁|`` (one-way, the fraction
    of the book traded). The result is the mean over all rebalance-to-rebalance
    transitions; a single-row weights frame has no transitions and returns
    ``0.0``.

    Parameters
    ----------
    weights:
        DataFrame indexed by rebalance date, one column per asset (matching
        :attr:`riskbudget.core.types.BacktestResult.weights`). Missing assets in
        a row are treated as zero weight.
    """
    if not isinstance(weights, pd.DataFrame):
        raise ValidationError("weights must be a pandas DataFrame.")
    if weights.shape[0] < 2:
        return 0.0
    w = weights.fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ValidationError("weights contains NaN or infinite values.")
    diffs = np.abs(np.diff(w, axis=0)).sum(axis=1) * 0.5
    return float(np.mean(diffs))


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "annualized_return",
    "annualized_volatility",
    "average_turnover",
    "calmar_ratio",
    "hit_rate",
    "sharpe_ratio",
    "sortino_ratio",
]
