"""Tests for the multi-strategy comparison report (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import BacktestResult
from riskbudget.reporting import build_comparison_report


def _make_result(
    name: str, mu: float, sigma: float, *, seed: int, periods: int = 240
) -> BacktestResult:
    """Construct a BacktestResult directly from a synthetic monthly return series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=periods, freq="ME")
    rets = pd.Series(rng.normal(mu, sigma, len(idx)), index=idx)
    equity = (1.0 + rets).cumprod()
    w = rng.dirichlet([3.0, 2.0], len(idx))  # two assets summing to 1 each row
    weights = pd.DataFrame(w, index=idx, columns=["STOCKS", "BONDS"])
    return BacktestResult(
        equity_curve=equity,
        weights=weights,
        returns=rets,
        metadata={"name": name, "frequency": "monthly", "lookback": 60},
        diagnostics={"n_rebalances": len(idx)},
    )


@pytest.fixture
def results() -> dict[str, BacktestResult]:
    return {
        "ERC": _make_result("ERC", 0.006, 0.020, seed=1),
        "GMV": _make_result("GMV", 0.004, 0.013, seed=2),
        "EqualWeight": _make_result("EqualWeight", 0.007, 0.030, seed=3),
    }


def test_comparison_report_one_row_per_strategy(results: dict[str, BacktestResult]) -> None:
    rep = build_comparison_report(results, periods_per_year=12)
    assert list(rep.summary.index) == ["ERC", "GMV", "EqualWeight"]
    assert "sharpe_ratio" in rep.summary.columns
    assert rep.metadata["strategies"] == ["ERC", "GMV", "EqualWeight"]
    assert rep.metadata["n_periods"] == 240


def test_comparison_report_has_overlay_figures(results: dict[str, BacktestResult]) -> None:
    rep = build_comparison_report(results, periods_per_year=12)
    assert set(rep.figures) == {"composition", "equity_curve", "drawdown"}
    # one trace per strategy on each overlay
    for name in ("equity_curve", "drawdown"):
        assert len(rep.figures[name].data) == 3
    # composition: one bar trace per asset (STOCKS, BONDS)
    assert len(rep.figures["composition"].data) == 2


def test_comparison_report_marks_composition_and_rebalance(
    results: dict[str, BacktestResult],
) -> None:
    rep = build_comparison_report(results, periods_per_year=12)
    # composition figure is a stacked bar of mean weights, one trace per asset
    comp = rep.figures["composition"]
    assert comp.layout.barmode == "stack"
    assert {tr.name for tr in comp.data} == {"STOCKS", "BONDS"}
    # rebalancing cadence is made explicit in the subtitle + metadata
    assert "Monthly rebalancing" in rep.subtitle
    assert "60-period lookback" in rep.subtitle
    assert rep.metadata["rebalance"] == rep.subtitle
    html = rep.to_html(include_plotlyjs=False)
    assert "Monthly rebalancing" in html


def test_comparison_report_renders_self_contained_html(
    results: dict[str, BacktestResult],
) -> None:
    rep = build_comparison_report(results, periods_per_year=12, title="Methods")
    html = rep.to_html(include_plotlyjs=True)
    assert "<html" in html.lower()
    assert "Methods" in html
    assert "ERC" in html and "GMV" in html
    assert "cdn.plot.ly" not in html or len(html) > 3_000_000  # inline bundle


def test_comparison_report_empty_raises() -> None:
    with pytest.raises(ValidationError):
        build_comparison_report({})


def test_comparison_report_aligns_misaligned_indices() -> None:
    a = _make_result("A", 0.005, 0.02, seed=4)
    b = _make_result("B", 0.005, 0.02, seed=5)
    # truncate B so the two have different lengths; report must still align/score
    b.returns = b.returns.iloc[12:]
    rep = build_comparison_report({"A": a, "B": b}, periods_per_year=12)
    assert list(rep.summary.index) == ["A", "B"]
    assert rep.metadata["n_periods"] == 240  # union index
