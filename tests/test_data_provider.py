"""Offline tests for the Tiingo prototype provider adapter (Agent 1, BUILD_PLAN §8).

These tests run entirely against the committed JSON fixture under
``riskbudget/data/providers/fixtures/tiingo/`` — they never touch the network,
which is mandatory because every market-data host is blocked by this
environment's allowlist (see ``docs/data-sources.md``). The live HTTPS transport
is documented and unit-covered only at its guard rails; the recorded fixture
exercises the same parsing/alignment path that runs in production.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from riskbudget.core.errors import DataError
from riskbudget.core.interfaces import DataSource
from riskbudget.core.types import PriceData
from riskbudget.data.providers.tiingo import (
    TiingoDataSource,
    _parse_records,
    tiingo_data_source,
)

START = date(2023, 1, 3)
END = date(2023, 2, 27)


@pytest.fixture
def source() -> TiingoDataSource:
    """The offline fixture-backed adapter (package default cache)."""
    return TiingoDataSource.from_fixtures()


# ---------------------------------------------------------------------------
# Protocol conformance + happy path
# ---------------------------------------------------------------------------


def test_satisfies_datasource_protocol(source: TiingoDataSource) -> None:
    assert isinstance(source, DataSource)


def test_get_prices_returns_pricedata(source: TiingoDataSource) -> None:
    prices = source.get_prices(["AAPL", "MSFT"], START, END)
    assert isinstance(prices, PriceData)
    assert prices.assets == ["AAPL", "MSFT"]
    assert prices.shape == (40, 2)
    # Adjusted close is used: the first AAPL adjClose in the fixture is 62.5087,
    # well below the raw close (125.0173), confirming we read adjClose not close.
    assert prices.frame.iloc[0]["AAPL"] == pytest.approx(62.5087)
    assert (prices.values > 0).all()


def test_factory_offline_default_is_working_source() -> None:
    ds = tiingo_data_source()
    assert isinstance(ds, DataSource)
    prices = ds.get_prices(["AAPL"], START, END)
    assert prices.assets == ["AAPL"]


def test_requested_asset_order_is_preserved(source: TiingoDataSource) -> None:
    prices = source.get_prices(["MSFT", "AAPL"], START, END)
    assert prices.assets == ["MSFT", "AAPL"]


def test_prices_convert_to_returns(source: TiingoDataSource) -> None:
    prices = source.get_prices(["AAPL", "MSFT"], START, END)
    returns = prices.to_returns("simple")
    assert returns.shape == (39, 2)
    assert returns.assets == ["AAPL", "MSFT"]


def test_dates_are_ascending_and_unique(source: TiingoDataSource) -> None:
    prices = source.get_prices(["AAPL", "MSFT"], START, END)
    idx = prices.dates
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates


def test_available_symbols(source: TiingoDataSource) -> None:
    assert source.available_symbols() == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Date-window restriction
# ---------------------------------------------------------------------------


def test_window_restriction_is_inclusive(source: TiingoDataSource) -> None:
    sub = source.get_prices(["AAPL"], date(2023, 1, 3), date(2023, 1, 10))
    assert sub.dates.min() == pd.Timestamp(2023, 1, 3)
    assert sub.dates.max() <= pd.Timestamp(2023, 1, 10)
    assert sub.shape[0] < 40


def test_window_with_no_data_raises(source: TiingoDataSource) -> None:
    with pytest.raises(DataError, match="No Tiingo prices"):
        source.get_prices(["AAPL"], date(2030, 1, 1), date(2030, 12, 31))


# ---------------------------------------------------------------------------
# Multi-asset alignment
# ---------------------------------------------------------------------------


def test_multi_asset_panel_is_dense_and_aligned(source: TiingoDataSource) -> None:
    prices = source.get_prices(["AAPL", "MSFT"], START, END)
    frame = prices.frame
    assert not frame.isna().to_numpy().any()
    # Both columns share the exact same date index.
    assert list(frame.index) == sorted(frame.index)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_symbol_raises(source: TiingoDataSource) -> None:
    with pytest.raises(DataError, match="No recorded Tiingo fixture"):
        source.get_prices(["NOPE"], START, END)


def test_empty_assets_raises(source: TiingoDataSource) -> None:
    with pytest.raises(DataError, match="at least one asset"):
        source.get_prices([], START, END)


def test_duplicate_assets_raise(source: TiingoDataSource) -> None:
    with pytest.raises(DataError, match="Duplicate assets"):
        source.get_prices(["AAPL", "AAPL"], START, END)


def test_start_after_end_raises(source: TiingoDataSource) -> None:
    with pytest.raises(DataError, match="must not be after"):
        source.get_prices(["AAPL"], END, START)


def test_missing_fixture_dir_raises() -> None:
    with pytest.raises(DataError, match="fixture directory not found"):
        TiingoDataSource.from_fixtures("/nonexistent/path/xyz")


# ---------------------------------------------------------------------------
# Constructor / transport selection
# ---------------------------------------------------------------------------


def test_constructor_requires_key_or_fixtures() -> None:
    with pytest.raises(DataError, match="either an api_key"):
        TiingoDataSource()


def test_live_requires_api_key() -> None:
    with pytest.raises(DataError, match="non-empty api_key"):
        TiingoDataSource.live("")


def test_live_constructor_builds_without_network() -> None:
    # Building the live adapter must not hit the network; only get_prices would.
    ds = TiingoDataSource.live("fake-token")
    assert isinstance(ds, DataSource)


def test_factory_live_path_selected_with_key() -> None:
    ds = tiingo_data_source(api_key="fake-token")
    assert isinstance(ds, TiingoDataSource)
    # No fixture root => fixture reads are unavailable on the live transport.
    with pytest.raises(DataError, match="only defined for the fixture transport"):
        ds.available_symbols()


def test_available_symbols_requires_fixtures() -> None:
    ds = TiingoDataSource.live("fake-token")
    with pytest.raises(DataError, match="only defined for the fixture transport"):
        ds.available_symbols()


# ---------------------------------------------------------------------------
# Parser unit tests (the path shared by both transports)
# ---------------------------------------------------------------------------


def test_parse_records_uses_adjclose() -> None:
    records = [
        {"date": "2023-01-03T00:00:00.000Z", "close": 100.0, "adjClose": 50.0},
        {"date": "2023-01-04T00:00:00.000Z", "close": 101.0, "adjClose": 50.5},
    ]
    series = _parse_records("X", records)
    assert list(series.values) == [50.0, 50.5]
    assert series.index[0] == pd.Timestamp(2023, 1, 3)


def test_parse_records_normalizes_and_sorts() -> None:
    records = [
        {"date": "2023-01-04T00:00:00.000Z", "adjClose": 2.0},
        {"date": "2023-01-03T00:00:00.000Z", "adjClose": 1.0},
    ]
    series = _parse_records("X", records)
    assert series.index.is_monotonic_increasing
    assert list(series.values) == [1.0, 2.0]
    # midnight-UTC stamps collapse to a tz-naive calendar date
    assert series.index.tz is None


def test_parse_records_dedupes_keeping_last() -> None:
    records = [
        {"date": "2023-01-03T00:00:00.000Z", "adjClose": 1.0},
        {"date": "2023-01-03T00:00:00.000Z", "adjClose": 9.0},
    ]
    series = _parse_records("X", records)
    assert len(series) == 1
    assert series.iloc[0] == 9.0


def test_parse_records_empty_raises() -> None:
    with pytest.raises(DataError, match="no price records"):
        _parse_records("X", [])


def test_parse_records_missing_field_raises() -> None:
    with pytest.raises(DataError, match="missing 'date' or"):
        _parse_records("X", [{"date": "2023-01-03T00:00:00.000Z"}])


def test_parse_records_bad_value_raises() -> None:
    with pytest.raises(DataError, match="could not be parsed"):
        _parse_records("X", [{"date": "not-a-date", "adjClose": "oops"}])


def test_parse_records_non_object_raises() -> None:
    with pytest.raises(DataError, match="is not an object"):
        _parse_records("X", [["not", "a", "mapping"]])


# ---------------------------------------------------------------------------
# Fixture integrity: the committed recording is valid Tiingo JSON
# ---------------------------------------------------------------------------


def test_fixtures_match_tiingo_schema() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "riskbudget"
        / "data"
        / "providers"
        / "fixtures"
        / "tiingo"
    )
    files = sorted(root.glob("*.json"))
    assert files, "expected committed Tiingo fixtures"
    required = {"date", "close", "adjClose"}
    for path in files:
        records = json.loads(path.read_text())
        assert isinstance(records, list) and records
        for row in records:
            assert required.issubset(row.keys())
