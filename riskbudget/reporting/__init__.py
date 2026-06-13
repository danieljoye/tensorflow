"""Reporting layer: assemble summary tables + Plotly charts into a report.

Pure rendering over a :class:`~riskbudget.core.types.BacktestResult` — no strategy
recomputation (agent-6 §6, BUILD_PLAN §3.1). Plotly is imported lazily inside the
figure builders, so importing this package does not require plotly.

Public surface: :class:`Report`, :func:`build_report`, and the individual figure
builders (:func:`equity_curve_figure`, :func:`drawdown_figure`,
:func:`weights_figure`, :func:`risk_contribution_figure`).
"""

from __future__ import annotations

from riskbudget.reporting.comparison import build_comparison_report
from riskbudget.reporting.report import (
    Report,
    build_report,
    drawdown_figure,
    equity_curve_figure,
    risk_contribution_figure,
    weights_figure,
)

__all__ = [
    "Report",
    "build_comparison_report",
    "build_report",
    "drawdown_figure",
    "equity_curve_figure",
    "risk_contribution_figure",
    "weights_figure",
]
