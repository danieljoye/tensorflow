"""Analytics layer: performance, downside risk, distribution, attribution, summary.

This package turns a :class:`~riskbudget.core.types.BacktestResult` (and
point-in-time portfolios) into numbers — it never recomputes strategy logic.
EDHEC ``edhec_risk_kit`` formulations throughout (BUILD_PLAN §2, §3.1, §11).

Public surface:

- performance: :func:`annualized_return`, :func:`annualized_volatility`,
  :func:`sharpe_ratio`, :func:`sortino_ratio`, :func:`calmar_ratio`,
  :func:`hit_rate`, :func:`average_turnover`.
- downside: :func:`drawdown`, :func:`max_drawdown`, :func:`semideviation`,
  :func:`var_historic`, :func:`var_gaussian`, :func:`cvar_historic`.
- distribution: :func:`skewness`, :func:`kurtosis`, :func:`is_normal`.
- attribution: :func:`risk_contribution_history`, :func:`budget_drift`,
  :func:`diversification_history`.
- summary: :func:`summary_stats`.
"""

from __future__ import annotations

from riskbudget.analytics.attribution import (
    budget_drift,
    diversification_history,
    risk_contribution_history,
)
from riskbudget.analytics.distribution import is_normal, kurtosis, skewness
from riskbudget.analytics.downside import (
    cvar_historic,
    drawdown,
    max_drawdown,
    semideviation,
    var_gaussian,
    var_historic,
)
from riskbudget.analytics.performance import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from riskbudget.analytics.summary import SUMMARY_COLUMNS, summary_stats

__all__ = [
    "SUMMARY_COLUMNS",
    "annualized_return",
    "annualized_volatility",
    "average_turnover",
    "budget_drift",
    "calmar_ratio",
    "cvar_historic",
    "diversification_history",
    "drawdown",
    "hit_rate",
    "is_normal",
    "kurtosis",
    "max_drawdown",
    "risk_contribution_history",
    "semideviation",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "summary_stats",
    "var_gaussian",
    "var_historic",
]
