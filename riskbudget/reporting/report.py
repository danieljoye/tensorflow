"""Report assembly: summary table + Plotly charts (agent-6 §6).

Pure rendering — this module recomputes *no* strategy logic. It consumes a
:class:`~riskbudget.core.types.BacktestResult` (already produced by Agent 5's
engine) plus a covariance for the risk-contribution view, and assembles:

- a :func:`summary table <riskbudget.analytics.summary.summary_stats>`,
- an **equity-curve** line chart,
- an **underwater / drawdown** area chart,
- a **weights-through-time** stacked-area chart, and
- a **risk-contribution** stacked-area chart (percentage contributions).

:func:`build_report` returns a :class:`Report` dataclass holding the summary table
and the Plotly figures; :meth:`Report.to_html` renders a single self-contained
HTML document. Plotly is imported lazily so that simply importing the analytics
package does not require it.

Source: charts follow the EDHEC course dashboards; metrics from
:mod:`riskbudget.analytics`. BUILD_PLAN §3.1, §11.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from riskbudget.analytics.attribution import risk_contribution_history
from riskbudget.analytics.downside import drawdown
from riskbudget.analytics.summary import summary_stats
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import BacktestResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go


def _require_plotly() -> Any:
    """Import plotly.graph_objects lazily, with a clear error if missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValidationError(
            "plotly is required for reporting; install the 'reporting' extra."
        ) from exc
    return go


@dataclass
class Report:
    """A rendered analytics report for a single backtest.

    Attributes
    ----------
    title:
        Report heading.
    summary:
        The :func:`~riskbudget.analytics.summary.summary_stats` table (one row).
    figures:
        Ordered mapping ``name -> plotly Figure`` for each chart.
    metadata:
        Free-form provenance copied from the source
        :class:`~riskbudget.core.types.BacktestResult`.
    """

    title: str
    summary: pd.DataFrame
    figures: dict[str, go.Figure] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_html(self, *, include_plotlyjs: str | bool = "cdn", full_html: bool = True) -> str:
        """Render the report to a single HTML string.

        Parameters
        ----------
        include_plotlyjs:
            Forwarded to Plotly's ``to_html``. ``"cdn"`` (default) keeps the
            document small by loading plotly.js from a CDN; pass ``True`` to
            inline the library for fully offline viewing.
        full_html:
            When True wrap the output in a complete ``<html>`` document; when
            False emit only the report fragment (for embedding).
        """
        summary_html = self.summary.to_html(
            classes="summary-stats", float_format=lambda v: f"{v:.4f}", border=0
        )

        chart_blocks: list[str] = []
        first = True
        for name, fig in self.figures.items():
            # Load plotly.js only once (with the first figure).
            js: str | bool = include_plotlyjs if first else False
            fragment = fig.to_html(include_plotlyjs=js, full_html=False)
            chart_blocks.append(f'<section class="chart" id="chart-{name}">{fragment}</section>')
            first = False

        body = (
            f"<h1>{self.title}</h1>\n"
            f'<section class="summary"><h2>Summary statistics</h2>{summary_html}</section>\n'
            + "\n".join(chart_blocks)
        )

        if not full_html:
            return body

        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='utf-8'>\n"
            f"<title>{self.title}</title>\n"
            "<style>"
            "body{font-family:system-ui,Arial,sans-serif;margin:2rem;}"
            "table.summary-stats{border-collapse:collapse;}"
            "table.summary-stats th,table.summary-stats td"
            "{padding:4px 10px;text-align:right;border-bottom:1px solid #ddd;}"
            ".chart{margin:1.5rem 0;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
        )


def equity_curve_figure(result: BacktestResult) -> go.Figure:
    """Line chart of the cumulative equity curve."""
    go = _require_plotly()
    curve = result.equity_curve
    fig = go.Figure(go.Scatter(x=list(curve.index), y=curve.to_numpy(dtype=float), mode="lines"))
    fig.update_layout(title="Equity curve", xaxis_title="Date", yaxis_title="Growth of $1")
    return fig


def drawdown_figure(result: BacktestResult) -> go.Figure:
    """Underwater (drawdown) area chart derived from the equity curve.

    The drawdown is computed from the equity curve's period-over-period returns so
    it reuses :func:`riskbudget.analytics.downside.drawdown` rather than
    recomputing wealth from scratch.
    """
    go = _require_plotly()
    curve = result.equity_curve.astype(float)
    if curve.size < 2:
        dd = pd.Series([0.0] * curve.size, index=curve.index)
    else:
        rets = curve.pct_change().iloc[1:]
        dd = drawdown(rets)["drawdown"]
    fig = go.Figure(
        go.Scatter(
            x=list(dd.index),
            y=dd.to_numpy(dtype=float),
            mode="lines",
            fill="tozeroy",
            line={"color": "crimson"},
        )
    )
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def weights_figure(result: BacktestResult) -> go.Figure:
    """Stacked-area chart of weights through time."""
    go = _require_plotly()
    weights = result.weights.fillna(0.0)
    fig = go.Figure()
    for asset in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=list(weights.index),
                y=weights[asset].to_numpy(dtype=float),
                mode="lines",
                name=str(asset),
                stackgroup="weights",
            )
        )
    fig.update_layout(title="Weights through time", xaxis_title="Date", yaxis_title="Weight")
    return fig


def risk_contribution_figure(
    result: BacktestResult,
    cov: np.ndarray,
) -> go.Figure:
    """Stacked-area chart of percentage risk contributions through time.

    Reuses :func:`riskbudget.analytics.attribution.risk_contribution_history`
    (which in turn uses Agent 4's contribution math). ``cov`` must be ordered to
    match the weights-frame columns.
    """
    go = _require_plotly()
    pcr = risk_contribution_history(result.weights, cov, percentage=True, verify=False)
    fig = go.Figure()
    for asset in pcr.columns:
        fig.add_trace(
            go.Scatter(
                x=list(pcr.index),
                y=pcr[asset].to_numpy(dtype=float),
                mode="lines",
                name=str(asset),
                stackgroup="risk",
            )
        )
    fig.update_layout(
        title="Risk contributions through time",
        xaxis_title="Date",
        yaxis_title="Percentage risk contribution",
    )
    return fig


def build_report(
    result: BacktestResult,
    *,
    cov: np.ndarray | None = None,
    title: str = "Backtest report",
    strategy_name: str | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Report:
    """Assemble a :class:`Report` from a backtest result (pure rendering).

    Parameters
    ----------
    result:
        The backtest output to report on. Its ``returns`` series drives the
        summary table; if absent, it is derived from the equity curve.
    cov:
        Optional covariance (ordered to match ``result.weights`` columns) used for
        the risk-contribution chart. When ``None`` that chart is omitted.
    title:
        Report heading.
    strategy_name:
        Row label for the summary table; defaults to the returns series name or
        ``"strategy"``.
    risk_free_rate, periods_per_year:
        Forwarded to :func:`riskbudget.analytics.summary.summary_stats`.

    Returns
    -------
    Report
        Holding the summary table and the Plotly figures.
    """
    if not isinstance(result, BacktestResult):
        raise ValidationError("build_report expects a BacktestResult.")

    if result.returns is not None:
        rets = result.returns.astype(float)
    else:
        curve = result.equity_curve.astype(float)
        if curve.size < 2:
            raise ValidationError(
                "Cannot derive returns: equity curve has fewer than two points and "
                "no returns series was supplied."
            )
        rets = curve.pct_change().iloc[1:]

    if strategy_name is not None:
        rets = rets.rename(strategy_name)
    elif rets.name is None:
        rets = rets.rename("strategy")

    summary = summary_stats(rets, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

    figures: dict[str, go.Figure] = {
        "equity": equity_curve_figure(result),
        "drawdown": drawdown_figure(result),
        "weights": weights_figure(result),
    }
    if cov is not None:
        figures["risk_contribution"] = risk_contribution_figure(result, cov)

    return Report(
        title=title,
        summary=summary,
        figures=figures,
        metadata=dict(result.metadata),
    )


__all__ = [
    "Report",
    "build_report",
    "drawdown_figure",
    "equity_curve_figure",
    "risk_contribution_figure",
    "weights_figure",
]
