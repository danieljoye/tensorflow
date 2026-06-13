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


def _composition_figure(results: Mapping[str, BacktestResult]) -> Any:
    """Stacked-bar of each strategy's *average* composition mix (% of gross).

    Each rebalance date's weights are normalized by that date's **gross
    exposure** (sum of absolute weights) before averaging, so the bars show the
    true allocation mix summing to 100% regardless of leverage. Leverage is
    reported separately: each bar carries a per-strategy annotation of the mean
    gross exposure (e.g. ``1.8x gross``). Missing assets render as 0%.
    """
    go = _require_plotly()
    comp_cols: dict[str, pd.Series] = {}
    gross_by_strategy: dict[str, float] = {}
    for name, res in results.items():
        if res.weights is None:
            continue
        w = res.weights.astype(float)
        gross = w.abs().sum(axis=1)
        # Composition mix: normalize each rebalance row by its gross exposure
        # (rows with zero gross — an all-cash book — contribute nothing).
        mix = w.div(gross.where(gross > 0.0), axis=0)
        comp_cols[name] = mix.mean()
        gross_by_strategy[name] = float(gross.mean())
    comp = pd.DataFrame(comp_cols).T.fillna(0.0)  # rows = strategies, cols = assets
    fig = go.Figure()
    for asset in comp.columns:
        values = comp[asset].fillna(0.0)
        fig.add_trace(
            go.Bar(
                x=comp.index.astype(str),
                y=values.to_numpy(),
                name=str(asset),
                text=[f"{v:.0%}" for v in values],
                textposition="inside",
            )
        )
    # Mean gross leverage per strategy, annotated above each bar.
    for name in comp.index:
        fig.add_annotation(
            x=str(name),
            y=1.0,
            yshift=14,
            text=f"{gross_by_strategy.get(str(name), 0.0):.1f}x gross",
            showarrow=False,
        )
    fig.update_layout(
        title="Average portfolio composition (% of gross; leverage annotated)",
        barmode="stack",
        xaxis_title="Strategy",
        yaxis_title="Share of gross exposure",
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend_title="Asset",
        height=420,
    )
    return fig


def _unique_in_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _rebalance_subtitle(results: Mapping[str, BacktestResult], periods_per_year: int) -> str:
    """Build an explicit rebalancing/period caption from ALL results' metadata.

    When every strategy shares a frequency / lookback / rebalance count the
    caption reads as a single setting; when they differ it reports the mix
    (e.g. ``Mixed rebalancing (monthly, quarterly) · lookbacks 60-252``) rather
    than misattributing the first strategy's settings to all of them.
    """
    freqs = _unique_in_order(
        [str((res.metadata or {}).get("frequency", "?")) for res in results.values()]
    )
    lookbacks = _unique_in_order(
        [
            (res.metadata or {}).get("lookback")
            for res in results.values()
            if (res.metadata or {}).get("lookback") is not None
        ]
    )
    n_rebs = _unique_in_order(
        [
            (res.diagnostics or {}).get("n_rebalances")
            for res in results.values()
            if (res.diagnostics or {}).get("n_rebalances") is not None
        ]
    )
    unit = "day" if periods_per_year >= 252 else "month" if periods_per_year == 12 else "period"

    if len(freqs) == 1:
        parts = [f"{freqs[0].capitalize()} rebalancing"]
    else:
        parts = [f"Mixed rebalancing ({', '.join(freqs)})"]
    if len(lookbacks) == 1:
        parts.append(f"{lookbacks[0]}-{unit} lookback (covariance & vol)")
    elif lookbacks:
        parts.append(f"lookbacks {min(lookbacks)}-{max(lookbacks)}")
    if len(n_rebs) == 1:
        parts.append(f"{n_rebs[0]} rebalances")
    elif n_rebs:
        parts.append(f"{min(n_rebs)}-{max(n_rebs)} rebalances")

    starts = [
        pd.Timestamp(res.weights.index[0])
        for res in results.values()
        if res.weights is not None and len(res.weights)
    ]
    ends = [
        pd.Timestamp(res.weights.index[-1])
        for res in results.values()
        if res.weights is not None and len(res.weights)
    ]
    span = f" · {min(starts):%Y-%m} → {max(ends):%Y-%m}" if starts else ""
    return " · ".join(parts) + span


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
        "composition": _composition_figure(results),
        "equity_curve": _overlay_figure(
            equity, "Growth of $1 (rebased)", "Growth of $1", log_y=log_equity
        ),
        "drawdown": _overlay_figure(dd, "Drawdown", "Drawdown", log_y=False),
    }
    subtitle = _rebalance_subtitle(results, periods_per_year)
    metadata: dict[str, Any] = {
        "strategies": list(results.keys()),
        "n_periods": len(rets),
        "periods_per_year": periods_per_year,
        "rebalance": subtitle,
    }
    return Report(
        title=title, subtitle=subtitle, summary=summary, figures=figures, metadata=metadata
    )
