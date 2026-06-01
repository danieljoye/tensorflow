"""Run a panel of strategies on one dataset and compare them head-to-head.

:func:`compare` takes several :class:`~riskbudget.spec.StrategySpec` objects, runs
each through the walk-forward backtester (Agent 5) on its configured data, and
returns a :func:`~riskbudget.analytics.summary.summary_stats` table — one row per
strategy (BUILD_PLAN §5.2). This is the canonical way to put a risk-budgeted
portfolio next to its classical benchmarks.

The heavy lifting (data → construct → backtest) lives in :mod:`riskbudget.run`,
which the public ``construct``/``backtest`` helpers also use, so every surface
shares one wiring path.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from riskbudget.analytics import summary_stats
from riskbudget.core.errors import BacktestError, ConfigurationError
from riskbudget.core.types import BacktestResult
from riskbudget.run import backtest
from riskbudget.spec import StrategySpec


def compare(
    specs: Sequence[StrategySpec],
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
    var_level: float = 0.05,
) -> pd.DataFrame:
    """Run each spec through the backtester and return a comparison table.

    Parameters
    ----------
    specs:
        One or more :class:`StrategySpec`. Each is backtested independently on its
        own configured data source; their per-period return series are pooled into
        a single :func:`summary_stats` table (one row per strategy ``name``).
    risk_free_rate, periods_per_year:
        Override the annualization / Sharpe inputs for *all* rows. When ``None``
        they default to the first spec's own settings (specs in a panel normally
        share these).
    var_level:
        Tail confidence for the VaR/CVaR columns (default 5%).

    Returns
    -------
    pandas.DataFrame
        Index = strategy names; columns = the canonical summary metrics. Strategy
        names must be unique across the panel.

    Raises
    ------
    ConfigurationError
        If ``specs`` is empty or two specs share a ``name``.
    BacktestError
        If a backtest produces no usable return series.
    """
    if not specs:
        raise ConfigurationError("compare() needs at least one StrategySpec.")

    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ConfigurationError(f"StrategySpec names must be unique in a panel; got {names}.")

    rf = specs[0].risk_free_rate if risk_free_rate is None else risk_free_rate
    ppy = specs[0].periods_per_year if periods_per_year is None else periods_per_year

    results: dict[str, BacktestResult] = {s.name: backtest(s) for s in specs}

    returns_frame = _collect_returns(results)
    return summary_stats(
        returns_frame,
        risk_free_rate=rf,
        periods_per_year=ppy,
        var_level=var_level,
    )


def _collect_returns(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Pool each result's per-period return series into one aligned DataFrame."""
    series: dict[str, pd.Series] = {}
    for name, result in results.items():
        if result.returns is None:
            raise BacktestError(f"Backtest for {name!r} produced no return series to compare.")
        series[name] = result.returns.rename(name)
    frame = pd.DataFrame(series)
    return frame


__all__ = ["compare"]
