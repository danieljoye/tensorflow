"""Tests for reporting/report.py: report assembly and HTML rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.analytics.summary import SUMMARY_COLUMNS
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import BacktestResult
from riskbudget.reporting.report import (
    Report,
    build_report,
    drawdown_figure,
    equity_curve_figure,
    risk_contribution_figure,
    weights_figure,
)


def _backtest_result(assets: list[str], cov: np.ndarray) -> BacktestResult:
    rng = np.random.default_rng(42)
    n = 250
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    rets = pd.Series(rng.standard_normal(n) * 0.01 + 0.0004, index=dates, name="erc")
    equity = (1.0 + rets).cumprod()
    reb_dates = dates[::50]
    weights = pd.DataFrame(
        np.tile([0.4, 0.35, 0.25], (len(reb_dates), 1)),
        index=reb_dates,
        columns=assets,
    )
    return BacktestResult(
        equity_curve=equity,
        weights=weights,
        returns=rets,
        metadata={"method": "erc"},
    )


def test_build_report_returns_report(assets: list[str], cov_3: np.ndarray) -> None:
    result = _backtest_result(assets, cov_3)
    report = build_report(result, cov=cov_3, strategy_name="erc")
    assert isinstance(report, Report)
    assert list(report.summary.columns) == SUMMARY_COLUMNS
    assert list(report.summary.index) == ["erc"]
    # All four charts present when cov supplied.
    assert set(report.figures) == {"equity", "drawdown", "weights", "risk_contribution"}
    assert report.metadata["method"] == "erc"


def test_report_omits_risk_chart_without_cov(assets: list[str], cov_3: np.ndarray) -> None:
    result = _backtest_result(assets, cov_3)
    report = build_report(result)
    assert "risk_contribution" not in report.figures
    assert "equity" in report.figures


def test_report_to_html_renders(assets: list[str], cov_3: np.ndarray) -> None:
    result = _backtest_result(assets, cov_3)
    report = build_report(result, cov=cov_3)
    html = report.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "Summary statistics" in html
    assert "plotly" in html.lower()
    # Each chart section is present.
    for name in report.figures:
        assert f"chart-{name}" in html


def test_report_to_html_fragment(assets: list[str], cov_3: np.ndarray) -> None:
    result = _backtest_result(assets, cov_3)
    report = build_report(result, cov=cov_3)
    fragment = report.to_html(full_html=False)
    assert not fragment.startswith("<!DOCTYPE")
    assert "<h1>" in fragment


def test_individual_figure_builders(assets: list[str], cov_3: np.ndarray) -> None:
    import plotly.graph_objects as go

    result = _backtest_result(assets, cov_3)
    assert isinstance(equity_curve_figure(result), go.Figure)
    assert isinstance(drawdown_figure(result), go.Figure)
    assert isinstance(weights_figure(result), go.Figure)
    assert isinstance(risk_contribution_figure(result, cov_3), go.Figure)


def test_build_report_derives_returns_from_equity(assets: list[str], cov_3: np.ndarray) -> None:
    result = _backtest_result(assets, cov_3)
    result.returns = None  # force derivation from equity curve
    report = build_report(result, cov=cov_3, strategy_name="derived")
    assert list(report.summary.index) == ["derived"]


def test_build_report_rejects_non_backtest_result() -> None:
    with pytest.raises(ValidationError):
        build_report(object())  # type: ignore[arg-type]


def test_build_report_single_point_without_returns_raises(assets: list[str]) -> None:
    equity = pd.Series([1.0], index=[pd.Timestamp("2021-01-01")])
    weights = pd.DataFrame([[0.4, 0.35, 0.25]], index=[pd.Timestamp("2021-01-01")], columns=assets)
    result = BacktestResult(equity_curve=equity, weights=weights, returns=None)
    with pytest.raises(ValidationError):
        build_report(result)
