"""Multi-strategy comparison report (overlay equity curves + drawdowns + a
stacked summary-stats table) for head-to-head method comparison.

Where :func:`riskbudget.reporting.build_report` documents a *single* backtest,
:func:`build_comparison_report` overlays several :class:`BacktestResult` objects
on shared charts and stacks their :func:`~riskbudget.analytics.summary.summary_stats`
rows into one table. It reuses the :class:`~riskbudget.reporting.report.Report`
container, so ``.to_html()`` renders the same self-contained document.

Pure rendering — no strategy recomputation (BUILD_PLAN §3.1). Plotly is imported
lazily via :func:`riskbudget.reporting.report._require_plotly`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from riskbudget.analytics.downside import drawdown
from riskbudget.analytics.summary import summary_stats
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import BacktestResult
from riskbudget.reporting.report import Report, _require_plotly


def _aligned_returns(results: Mapping[str, BacktestResult]) -> pd.DataFrame:
    """Stack each result's portfolio returns into one aligned DataFrame."""
    if not results:
        raise ValidationError("build_comparison_report requires at least one result.")
    cols: dict[str, pd.Series] = {}
    for name, res in results.items():
        r = res.returns
        if r is None or len(r) == 0:
            raise ValidationError(f"Strategy '{name}' has no returns to compare.")
        cols[name] = pd.Series(r).astype(float)
    return pd.DataFrame(cols).sort_index()


def _overlay_figure(frame: pd.DataFrame, title: str, ytitle: str, *, log_y: bool) -> Any:
    go = _require_plotly()
    fig = go.Figure()
    for col in frame.columns:
        s = frame[col].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.to_numpy(), mode="lines", name=str(col)))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ytitle,
        template="plotly_white",
        legend_title="Strategy",
        height=420,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def build_comparison_report(
    results: Mapping[str, BacktestResult],
    *,
    title: str = "Strategy comparison",
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    var_level: float = 0.05,
    log_equity: bool = True,
) -> Report:
    """Assemble a head-to-head comparison :class:`Report` from several backtests.

    Parameters
    ----------
    results:
        Mapping of ``strategy name -> BacktestResult``. Returns are aligned on a
        shared date index before charting and scoring.
    title:
        Report heading.
    periods_per_year, risk_free_rate, var_level:
        Forwarded to :func:`~riskbudget.analytics.summary.summary_stats`.
    log_equity:
        Plot the growth-of-$1 overlay on a log y-axis (sensible for long histories).

    Returns
    -------
    Report
        With a stacked summary table (one row per strategy) and two overlay
        figures: ``equity_curve`` (growth of $1) and ``drawdown``.
    """
    rets = _aligned_returns(results)
    # Score every strategy over the common (overlapping) period so the table is a
    # fair head-to-head; fall back to per-strategy own-history if there is no
    # shared window.
    common = rets.dropna(how="any")
    if len(common) >= 2:
        summary = summary_stats(
            common,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            var_level=var_level,
        )
    else:
        rows = [
            summary_stats(
                rets[name].dropna(),
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                var_level=var_level,
            ).rename(index={rets[name].dropna().name or 0: name})
            for name in rets.columns
        ]
        summary = pd.concat(rows)

    # Charts plot each strategy over its own available dates (overlay drops NaN).
    equity = pd.DataFrame({name: (1.0 + rets[name].dropna()).cumprod() for name in rets.columns})
    dd = pd.DataFrame({name: drawdown(rets[name].dropna())["drawdown"] for name in rets.columns})

    figures = {
        "equity_curve": _overlay_figure(
            equity, "Growth of $1 (rebased)", "Growth of $1", log_y=log_equity
        ),
        "drawdown": _overlay_figure(dd, "Drawdown", "Drawdown", log_y=False),
    }
    metadata: dict[str, Any] = {
        "strategies": list(results.keys()),
        "n_periods": len(rets),
        "periods_per_year": periods_per_year,
    }
    return Report(title=title, summary=summary, figures=figures, metadata=metadata)
