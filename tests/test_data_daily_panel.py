"""Offline tests for the long daily multi-asset panel provider.

These run entirely against the committed pre-assembled panel CSV under
``examples/data/daily_panel_long.csv`` -- they never touch the network, which is
mandatory for CI. The live GitHub transport is documented and guarded but only
its construction guard rails are exercised here; :func:`assemble_panel` (the
derivation shared with the live transport) is unit-tested directly against
recorded raw-source snippets.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from riskbudget.core.errors import DataError
from riskbudget.core.interfaces import DataSource
from riskbudget.core.types import PriceData
from riskbudget.data.providers.daily import (
    DailyPanelDataSource,
    assemble_panel,
    daily_panel_data_source,
)
from riskbudget.registry import REGISTRY

START = date(1968, 1, 1)
END = date(2024, 1, 1)
ASSETS = ["STOCKS", "BONDS", "GOLD"]


@pytest.fixture
def panel() -> DailyPanelDataSource:
    """The offline fixture-backed daily panel adapter (package default)."""
    return DailyPanelDataSource.from_fixtures()


# ---------------------------------------------------------------------------
# Protocol conformance + happy path
# ---------------------------------------------------------------------------


def test_satisfies_datasource_protocol(panel: DailyPanelDataSource) -> None:
    assert isinstance(panel, DataSource)


def test_get_prices_returns_pricedata(panel: DailyPanelDataSource) -> None:
    prices = panel.get_prices(ASSETS, START, END)
    assert isinstance(prices, PriceData)
    assert prices.assets == ASSETS
    # ~13.7k daily rows across the full window.
    assert prices.shape[0] > 10_000
    assert (prices.values > 0).all()


def test_available_assets(panel: DailyPanelDataSource) -> None:
    assert list(panel.available_assets()) == ASSETS


def test_panel_is_daily_and_deep(panel: DailyPanelDataSource) -> None:
    prices = panel.get_prices(ASSETS, START, END)
    # The aligned panel starts at the gold series (1968-01-02).
    assert prices.dates.min() <= pd.Timestamp(1968, 1, 3)
    assert prices.dates.max() >= pd.Timestamp(2023, 1, 1)
    # Daily cadence: a single calendar year holds far more than 12 rows.
    one_year = panel.get_prices(ASSETS, date(2000, 1, 1), date(2000, 12, 31))
    assert one_year.shape[0] > 200


def test_requested_asset_order_is_preserved(panel: DailyPanelDataSource) -> None:
    prices = panel.get_prices(["GOLD", "STOCKS"], START, END)
    assert prices.assets == ["GOLD", "STOCKS"]


def test_dates_are_ascending_and_unique(panel: DailyPanelDataSource) -> None:
    idx = panel.get_prices(["STOCKS"], START, END).dates
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates


def test_multi_asset_panel_is_dense_and_aligned(panel: DailyPanelDataSource) -> None:
    frame = panel.get_prices(ASSETS, START, END).frame
    assert not frame.isna().to_numpy().any()
    assert list(frame.index) == sorted(frame.index)


def test_factory_offline_default_is_working_source() -> None:
    src = daily_panel_data_source()
    assert isinstance(src, DataSource)
    assert src.get_prices(["STOCKS"], START, END).assets == ["STOCKS"]


# ---------------------------------------------------------------------------
# Derived-series properties
# ---------------------------------------------------------------------------


def test_bonds_index_is_positive_and_rebased(panel: DailyPanelDataSource) -> None:
    bonds = panel.get_prices(["BONDS"], START, END).frame["BONDS"]
    assert (bonds > 0).all()
    # Rebased to 1.0 on the first aligned row.
    assert bonds.iloc[0] == pytest.approx(1.0)
    # Recoverable daily gross returns are positive.
    gross = bonds / bonds.shift(1)
    assert (gross.dropna() > 0).all()


def test_stocks_grow_over_the_full_window(panel: DailyPanelDataSource) -> None:
    stocks = panel.get_prices(["STOCKS"], START, END).frame["STOCKS"]
    assert stocks.iloc[-1] > stocks.iloc[0]


# ---------------------------------------------------------------------------
# Date-window restriction
# ---------------------------------------------------------------------------


def test_window_restriction_is_inclusive(panel: DailyPanelDataSource) -> None:
    sub = panel.get_prices(["STOCKS"], date(1980, 1, 1), date(1980, 1, 31))
    assert sub.dates.min() >= pd.Timestamp(1980, 1, 1)
    assert sub.dates.max() <= pd.Timestamp(1980, 1, 31)
    assert sub.shape[0] <= 23  # business days in a month


def test_window_with_no_data_raises(panel: DailyPanelDataSource) -> None:
    with pytest.raises(DataError, match="No daily-panel data"):
        panel.get_prices(["STOCKS"], date(1900, 1, 1), date(1901, 1, 1))


# ---------------------------------------------------------------------------
# Error handling / taxonomy
# ---------------------------------------------------------------------------


def test_unknown_asset_raises(panel: DailyPanelDataSource) -> None:
    with pytest.raises(DataError, match="Unknown daily-panel asset"):
        panel.get_prices(["SILVER"], START, END)


def test_empty_assets_raise(panel: DailyPanelDataSource) -> None:
    with pytest.raises(DataError, match="at least one asset"):
        panel.get_prices([], START, END)


def test_duplicate_assets_raise(panel: DailyPanelDataSource) -> None:
    with pytest.raises(DataError, match="Duplicate assets"):
        panel.get_prices(["STOCKS", "STOCKS"], START, END)


def test_start_after_end_raises(panel: DailyPanelDataSource) -> None:
    with pytest.raises(DataError, match="must not be after"):
        panel.get_prices(["STOCKS"], END, START)


def test_missing_fixture_raises() -> None:
    with pytest.raises(DataError, match="Daily panel fixture not found"):
        DailyPanelDataSource.from_fixtures("/nonexistent/path/panel.csv")


def test_constructor_requires_transport() -> None:
    with pytest.raises(DataError, match="needs a fixture_path"):
        DailyPanelDataSource()


def test_live_constructors_build_without_network() -> None:
    # Building the live adapter must not hit the network; only get_prices would.
    assert isinstance(DailyPanelDataSource.from_github(), DataSource)
    assert isinstance(daily_panel_data_source(live=True), DataSource)


# ---------------------------------------------------------------------------
# assemble_panel: the derivation shared with the live transport
# ---------------------------------------------------------------------------

# Minimal recorded raw-source snippets in each mirror's exact on-disk shape.
_STOCKS_RAW = (
    "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen\n"
    "1800-01-01,1,1,1,1\n"  # pre-1885 backfill row -> dropped
    "1968-01-02,96,96,95,100,0\n"
    "1968-01-03,100,101,99,110,0\n"
)
_DGS10_RAW = "observation_date,DGS10\n1968-01-02,6.0\n1968-01-03,5.0\n"
_GOLD_RAW = "date,Gold Price\n1968-01-02,35.0\n1968-01-03,36.0\n"


def test_assemble_panel_aligns_and_derives() -> None:
    frame = assemble_panel(_STOCKS_RAW, _DGS10_RAW, _GOLD_RAW)
    assert list(frame.columns) == ASSETS
    # Pre-1885 stock backfill row dropped; gold/yield define the 2-row overlap.
    assert list(frame.index) == [pd.Timestamp(1968, 1, 2), pd.Timestamp(1968, 1, 3)]
    assert frame["STOCKS"].tolist() == [100.0, 110.0]
    assert frame["GOLD"].tolist() == [35.0, 36.0]
    # BONDS rebased to 1.0; day-2 gross uses the documented ModDur=8 proxy:
    # r = y0/252 - 8*(y1-y0) = 0.06/252 - 8*(0.05-0.06) = 0.0002381 + 0.08
    assert frame["BONDS"].iloc[0] == pytest.approx(1.0)
    expected_day2 = 1.0 + (0.06 / 252.0 - 8.0 * (0.05 - 0.06))
    assert frame["BONDS"].iloc[1] == pytest.approx(expected_day2)


def test_assemble_panel_skips_blank_and_dot_yields() -> None:
    dgs10 = "observation_date,DGS10\n1968-01-02,6.0\n1968-01-03,.\n1968-01-04,5.5\n"
    gold = "date,Gold Price\n1968-01-02,35.0\n1968-01-03,36.0\n1968-01-04,37.0\n"
    stocks = (
        "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen\n"
        "1968-01-02,1,1,1,100,0\n1968-01-03,1,1,1,110,0\n1968-01-04,1,1,1,120,0\n"
    )
    frame = assemble_panel(stocks, dgs10, gold)
    # The ``.`` yield row is dropped, so 1968-01-03 falls out of the inner join.
    assert list(frame.index) == [pd.Timestamp(1968, 1, 2), pd.Timestamp(1968, 1, 4)]


def test_assemble_panel_empty_overlap_raises() -> None:
    stocks = "Data,O,H,L,Zamkniecie,V\n1968-01-02,1,1,1,100,0\n"
    dgs10 = "observation_date,DGS10\n1990-01-02,8.0\n"
    gold = "date,Gold Price\n2000-01-02,280.0\n"
    with pytest.raises(DataError, match="empty after aligning"):
        assemble_panel(stocks, dgs10, gold)


# ---------------------------------------------------------------------------
# Registry wiring + cross-provider parity
# ---------------------------------------------------------------------------


def test_registry_exposes_daily_panel_source() -> None:
    assert "daily_panel" in REGISTRY.data_sources()
    src = REGISTRY.data_source("daily_panel")()
    assert isinstance(src, DataSource)
    assert src.get_prices(["STOCKS"], START, END).assets == ["STOCKS"]
