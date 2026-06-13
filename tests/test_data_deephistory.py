"""Offline tests for the deep-history providers (Shiller S&P 500 + gold).

These tests run entirely against the committed trimmed CSV fixtures under
``riskbudget/data/providers/fixtures/{shiller,gold}/`` — they never touch the
network, which is mandatory for CI. The live GitHub transport is documented and
guarded but exercised only at its construction guard rails; the recorded
fixtures exercise the same derivation/alignment path that runs live.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import riskbudget as rb
from riskbudget.core.errors import DataError
from riskbudget.core.interfaces import DataSource
from riskbudget.core.types import PriceData
from riskbudget.data.providers.gold import (
    GoldDataSource,
    gold_data_source,
)
from riskbudget.data.providers.gold import (
    _parse_table as _parse_gold,
)
from riskbudget.data.providers.shiller import (
    ShillerDataSource,
    shiller_data_source,
)
from riskbudget.data.providers.shiller import (
    _parse_table as _parse_shiller,
)
from riskbudget.registry import REGISTRY

START = date(1971, 1, 1)
END = date(2023, 1, 1)

SHILLER_ASSETS = ["SP500_NOMINAL", "SP500_REAL", "SP500_TR", "US10Y_TR"]


@pytest.fixture
def shiller() -> ShillerDataSource:
    """The offline fixture-backed Shiller adapter (package default)."""
    return ShillerDataSource.from_fixtures()


@pytest.fixture
def gold() -> GoldDataSource:
    """The offline fixture-backed gold adapter (package default)."""
    return GoldDataSource.from_fixtures()


# ---------------------------------------------------------------------------
# Protocol conformance + happy path
# ---------------------------------------------------------------------------


def test_shiller_satisfies_datasource_protocol(shiller: ShillerDataSource) -> None:
    assert isinstance(shiller, DataSource)


def test_gold_satisfies_datasource_protocol(gold: GoldDataSource) -> None:
    assert isinstance(gold, DataSource)


def test_shiller_get_prices_returns_pricedata(shiller: ShillerDataSource) -> None:
    prices = shiller.get_prices(SHILLER_ASSETS, START, END)
    assert isinstance(prices, PriceData)
    assert prices.assets == SHILLER_ASSETS
    assert prices.shape[0] > 100
    assert (prices.values > 0).all()


def test_gold_get_prices_returns_pricedata(gold: GoldDataSource) -> None:
    prices = gold.get_prices(["GOLD"], START, END)
    assert isinstance(prices, PriceData)
    assert prices.assets == ["GOLD"]
    assert (prices.values > 0).all()


def test_factory_offline_defaults_are_working_sources() -> None:
    s = shiller_data_source()
    g = gold_data_source()
    assert isinstance(s, DataSource)
    assert isinstance(g, DataSource)
    assert s.get_prices(["SP500_TR"], START, END).assets == ["SP500_TR"]
    assert g.get_prices(["GOLD"], START, END).assets == ["GOLD"]


def test_available_assets(shiller: ShillerDataSource, gold: GoldDataSource) -> None:
    assert list(shiller.available_assets()) == SHILLER_ASSETS
    assert list(gold.available_assets()) == ["GOLD"]


def test_requested_asset_order_is_preserved(shiller: ShillerDataSource) -> None:
    prices = shiller.get_prices(["US10Y_TR", "SP500_TR"], START, END)
    assert prices.assets == ["US10Y_TR", "SP500_TR"]


def test_dates_are_ascending_and_unique(shiller: ShillerDataSource) -> None:
    idx = shiller.get_prices(["SP500_TR"], START, END).dates
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates


# ---------------------------------------------------------------------------
# Derived-series properties: TR indices positive + monotone-cumulated
# ---------------------------------------------------------------------------


def test_sp500_tr_index_is_positive_and_cumulated(shiller: ShillerDataSource) -> None:
    tr = shiller.get_prices(["SP500_TR"], START, END).frame["SP500_TR"]
    assert (tr > 0).all()
    # An equity TR index grows over a 50-year window.
    assert tr.iloc[-1] > tr.iloc[0]
    # It is a cumulative product => recoverable monthly gross returns are positive.
    gross = tr / tr.shift(1)
    assert (gross.dropna() > 0).all()


def test_us10y_tr_index_is_positive_and_cumulated(shiller: ShillerDataSource) -> None:
    bond = shiller.get_prices(["US10Y_TR"], START, END).frame["US10Y_TR"]
    assert (bond > 0).all()
    assert bond.iloc[-1] > bond.iloc[0]
    gross = bond / bond.shift(1)
    assert (gross.dropna() > 0).all()


def test_us10y_tr_matches_documented_duration_proxy() -> None:
    # Reconstruct the proxy from a tiny synthetic table and compare to the parser.
    text = (
        "Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,"
        "Real Price,Real Dividend,Real Earnings,PE10\n"
        "1971-01-01,100,4.0,5,40,6.0,100,4,5,17\n"
        "1971-02-01,101,4.0,5,40,5.0,101,4,5,17\n"
    )
    frame = _parse_shiller(text)
    # ModDur=8: r = y0/1200 - 8*(y1-y0)/100 = 6/1200 - 8*(-0.01) = 0.005 + 0.08
    expected = 1.0 * (1.0 + (0.06 / 12.0 - 8.0 * (0.05 - 0.06)))
    assert frame["US10Y_TR"].iloc[1] == pytest.approx(expected)


def test_sp500_tr_matches_documented_dividend_formula() -> None:
    text = (
        "Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,"
        "Real Price,Real Dividend,Real Earnings,PE10\n"
        "1971-01-01,100,12.0,5,40,6.0,100,4,5,17\n"
        "1971-02-01,110,12.0,5,40,6.0,110,4,5,17\n"
    )
    frame = _parse_shiller(text)
    # r = (P1 + D1/12)/P0 - 1 = (110 + 1)/100 - 1 = 0.11
    assert frame["SP500_TR"].iloc[1] == pytest.approx(1.11)


# ---------------------------------------------------------------------------
# Date-window restriction
# ---------------------------------------------------------------------------


def test_shiller_window_restriction_is_inclusive(shiller: ShillerDataSource) -> None:
    sub = shiller.get_prices(["SP500_TR"], date(1980, 1, 1), date(1980, 6, 30))
    assert sub.dates.min() >= pd.Timestamp(1980, 1, 1)
    assert sub.dates.max() <= pd.Timestamp(1980, 6, 30)
    assert sub.shape[0] <= 6


def test_gold_window_restriction(gold: GoldDataSource) -> None:
    full = gold.get_prices(["GOLD"], START, END).shape[0]
    sub = gold.get_prices(["GOLD"], date(1980, 1, 1), date(1985, 1, 1)).shape[0]
    assert sub < full


def test_shiller_window_with_no_data_raises(shiller: ShillerDataSource) -> None:
    with pytest.raises(DataError, match="No Shiller data"):
        shiller.get_prices(["SP500_TR"], date(1800, 1, 1), date(1801, 1, 1))


def test_gold_window_with_no_data_raises(gold: GoldDataSource) -> None:
    with pytest.raises(DataError, match="No Gold data"):
        gold.get_prices(["GOLD"], date(2200, 1, 1), date(2201, 1, 1))


# ---------------------------------------------------------------------------
# Multi-asset alignment
# ---------------------------------------------------------------------------


def test_shiller_multi_asset_panel_is_dense_and_aligned(shiller: ShillerDataSource) -> None:
    frame = shiller.get_prices(["SP500_TR", "US10Y_TR"], START, END).frame
    assert not frame.isna().to_numpy().any()
    assert list(frame.index) == sorted(frame.index)


def test_cross_provider_alignment(shiller: ShillerDataSource, gold: GoldDataSource) -> None:
    eq = shiller.get_prices(["SP500_TR", "US10Y_TR"], START, END).frame
    au = gold.get_prices(["GOLD"], START, END).frame
    combined = pd.concat([eq, au], axis=1, join="inner").dropna(how="any")
    panel = PriceData(combined)
    assert panel.assets == ["SP500_TR", "US10Y_TR", "GOLD"]
    assert panel.shape[0] > 100


# ---------------------------------------------------------------------------
# Error handling / taxonomy
# ---------------------------------------------------------------------------


def test_shiller_unknown_asset_raises(shiller: ShillerDataSource) -> None:
    with pytest.raises(DataError, match="Unknown Shiller asset"):
        shiller.get_prices(["NOPE"], START, END)


def test_gold_unknown_asset_raises(gold: GoldDataSource) -> None:
    with pytest.raises(DataError, match="Unknown Gold asset"):
        gold.get_prices(["SILVER"], START, END)


def test_empty_assets_raise(shiller: ShillerDataSource, gold: GoldDataSource) -> None:
    with pytest.raises(DataError, match="at least one asset"):
        shiller.get_prices([], START, END)
    with pytest.raises(DataError, match="at least one asset"):
        gold.get_prices([], START, END)


def test_duplicate_assets_raise(shiller: ShillerDataSource) -> None:
    with pytest.raises(DataError, match="Duplicate assets"):
        shiller.get_prices(["SP500_TR", "SP500_TR"], START, END)


def test_start_after_end_raises(shiller: ShillerDataSource, gold: GoldDataSource) -> None:
    with pytest.raises(DataError, match="must not be after"):
        shiller.get_prices(["SP500_TR"], END, START)
    with pytest.raises(DataError, match="must not be after"):
        gold.get_prices(["GOLD"], END, START)


def test_missing_fixture_raises() -> None:
    with pytest.raises(DataError, match="Shiller fixture not found"):
        ShillerDataSource.from_fixtures("/nonexistent/path/shiller.csv")
    with pytest.raises(DataError, match="Gold fixture not found"):
        GoldDataSource.from_fixtures("/nonexistent/path/gold.csv")


def test_constructor_requires_transport() -> None:
    with pytest.raises(DataError, match="needs a fixture_path"):
        ShillerDataSource()
    with pytest.raises(DataError, match="needs a fixture_path"):
        GoldDataSource()


def test_live_constructors_build_without_network() -> None:
    # Building the live adapter must not hit the network; only get_prices would.
    assert isinstance(ShillerDataSource.from_github(), DataSource)
    assert isinstance(GoldDataSource.from_github(), DataSource)
    assert isinstance(shiller_data_source(live=True), DataSource)
    assert isinstance(gold_data_source(live=True), DataSource)


# ---------------------------------------------------------------------------
# Parser unit tests (the path shared by both transports)
# ---------------------------------------------------------------------------


def test_shiller_parser_missing_column_raises() -> None:
    with pytest.raises(DataError, match="missing required columns"):
        _parse_shiller("Date,SP500\n1971-01-01,100\n")


def test_shiller_parser_empty_raises() -> None:
    with pytest.raises(DataError, match="empty"):
        _parse_shiller("")


def test_gold_parser_missing_column_raises() -> None:
    with pytest.raises(DataError, match="missing required columns"):
        _parse_gold("Date\n1971-01\n")


def test_gold_parser_bad_value_raises() -> None:
    with pytest.raises(DataError, match="could not be parsed"):
        _parse_gold("Date,Price\n1971-01,oops\n")


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_exposes_deep_history_sources() -> None:
    assert "shiller" in REGISTRY.data_sources()
    assert "gold" in REGISTRY.data_sources()
    s = REGISTRY.data_source("shiller")()
    g = REGISTRY.data_source("gold")()
    assert isinstance(s, DataSource)
    assert isinstance(g, DataSource)
    assert s.get_prices(["SP500_TR"], START, END).assets == ["SP500_TR"]
    assert g.get_prices(["GOLD"], START, END).assets == ["GOLD"]


# ---------------------------------------------------------------------------
# Example smoke test: builds the panel and runs ERC offline
# ---------------------------------------------------------------------------


def _load_example() -> ModuleType:
    """Load examples/real_data_demo.py by file path (``examples`` is not a
    package on sys.path under pytest)."""
    path = Path(__file__).resolve().parents[1] / "examples" / "real_data_demo.py"
    spec = importlib.util.spec_from_file_location("real_data_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_builds_panel_and_runs_erc_offline() -> None:
    example = _load_example()
    build_panel, make_spec = example.build_panel, example.make_spec

    panel = build_panel(live=False)
    assert isinstance(panel, PriceData)
    assert panel.assets == ["STOCKS", "BONDS", "GOLD"]
    assert panel.shape[0] > 100
    assert panel.dates.min() >= pd.Timestamp(1971, 1, 1)

    result = rb.backtest(make_spec("ERC", "erc"), prices=panel)
    assert result.assets == ["STOCKS", "BONDS", "GOLD"]
    assert np.isfinite(float(result.equity_curve.iloc[-1]))
    assert float(result.equity_curve.iloc[-1]) > 0


def test_example_report_builds_offline(tmp_path: Path) -> None:
    from riskbudget.reporting import build_report

    example = _load_example()
    build_panel, make_spec = example.build_panel, example.make_spec

    panel = build_panel(live=False)
    result = rb.backtest(make_spec("ERC", "erc"), prices=panel)
    report = build_report(result, strategy_name="ERC", periods_per_year=12)
    html = report.to_html(include_plotlyjs=True)
    assert "<html" in html.lower()
    out = tmp_path / "report.html"
    out.write_text(html)
    assert out.stat().st_size > 0
