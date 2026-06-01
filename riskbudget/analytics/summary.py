"""Canonical per-strategy summary table (agent-6 §5, BUILD_PLAN §7).

``summary_stats`` collapses a return series (or a panel of strategies) into the
canonical one-row-per-strategy metrics table used to compare strategies side by
side and to drive the comparison report. The columns are the agreed set:

    annualized return, annualized volatility, Sharpe, max drawdown, skewness,
    kurtosis, Cornish-Fisher VaR(5%), and historic CVaR(5%)

plus Sortino, Calmar, and hit rate for completeness. Every metric delegates to
:mod:`riskbudget.analytics.performance` /
:mod:`riskbudget.analytics.downside` /
:mod:`riskbudget.analytics.distribution`, so this module never recomputes the
underlying statistics — it only assembles them.

Source: EDHEC ``edhec_risk_kit`` ``summary_stats``; BUILD_PLAN §3.1, §7, §11.
"""

from __future__ import annotations

import pandas as pd

from riskbudget.analytics.distribution import kurtosis, skewness
from riskbudget.analytics.downside import cvar_historic, max_drawdown, var_gaussian
from riskbudget.analytics.performance import (
    DEFAULT_PERIODS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from riskbudget.core.errors import ValidationError

# Canonical column order for the summary table.
SUMMARY_COLUMNS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "skewness",
    "kurtosis",
    "cornish_fisher_var_5",
    "historic_cvar_5",
    "hit_rate",
]


def summary_stats(
    returns: pd.Series | pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    var_level: float = 0.05,
) -> pd.DataFrame:
    """One-row-per-strategy table of headline performance and risk metrics.

    Parameters
    ----------
    returns:
        Either a single strategy's per-period simple returns (:class:`pandas.Series`,
        named or not) or a :class:`pandas.DataFrame` with one column per strategy.
        Each column is summarized independently.
    risk_free_rate:
        Annualized risk-free rate for Sharpe / Sortino (de-annualized internally).
    periods_per_year:
        Annualization factor (default 252, BUILD_PLAN §3.1).
    var_level:
        Tail confidence for the VaR/CVaR columns (default 0.05 ⇒ 5%).

    Returns
    -------
    pandas.DataFrame
        Index = strategy name(s); columns = :data:`SUMMARY_COLUMNS`. Metrics are
        Cornish-Fisher modified VaR and historic CVaR at ``var_level``, plus
        skewness and *excess* kurtosis (so a normal sample ⇒ ~0).
    """
    if isinstance(returns, pd.Series):
        name = returns.name if returns.name is not None else "strategy"
        frame = returns.to_frame(name=str(name))
    elif isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 0:
            raise ValidationError("summary_stats received a DataFrame with no columns.")
        frame = returns
    else:
        raise ValidationError("returns must be a pandas Series or DataFrame.")

    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        series = frame[col]
        rows[str(col)] = {
            "annualized_return": annualized_return(series, periods_per_year),
            "annualized_volatility": annualized_volatility(series, periods_per_year),
            "sharpe_ratio": sharpe_ratio(series, risk_free_rate, periods_per_year),
            "sortino_ratio": sortino_ratio(series, risk_free_rate, periods_per_year),
            "calmar_ratio": calmar_ratio(series, periods_per_year),
            "max_drawdown": max_drawdown(series),
            "skewness": skewness(series),
            "kurtosis": kurtosis(series, excess=True),
            "cornish_fisher_var_5": var_gaussian(series, var_level, modified=True),
            "historic_cvar_5": cvar_historic(series, var_level),
            "hit_rate": hit_rate(series),
        }

    table = pd.DataFrame.from_dict(rows, orient="index")
    return table[SUMMARY_COLUMNS]


__all__ = [
    "SUMMARY_COLUMNS",
    "summary_stats",
]
