"""Tests for the CSV / parquet price-panel loader (Agent 2).

Covers: CSV and parquet round-trips, date/asset filtering, the documented
forward-fill missing-data policy (and ``on_missing`` variants), the
``DataSource`` protocol, and error handling.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import DataError
from riskbudget.core.interfaces import DataSource
from riskbudget.core.types import PriceData
from riskbudget.data.csvsource import CsvDataSource, csv_data_source


def _panel(n: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "SPY": 100.0 + np.arange(n, dtype=float),
            "AGG": 50.0 + 0.5 * np.arange(n, dtype=float),
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "prices.csv"
    _panel().to_csv(p)
    return p


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "prices.parquet"
    _panel().to_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Round-trips & protocol
# ---------------------------------------------------------------------------


def test_satisfies_datasource_protocol(csv_path: Path) -> None:
    assert isinstance(CsvDataSource(csv_path), DataSource)


def test_csv_roundtrip(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    out = src.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2021, 1, 12))
    assert isinstance(out, PriceData)
    expected = _panel()
    assert np.allclose(out.frame["SPY"].to_numpy(), expected["SPY"].to_numpy())
    assert out.assets == ["SPY", "AGG"]


def test_parquet_roundtrip(parquet_path: Path) -> None:
    src = CsvDataSource(parquet_path)
    out = src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))
    expected = _panel()
    assert np.allclose(out.frame["SPY"].to_numpy(), expected["SPY"].to_numpy())


def test_date_filtering(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    out = src.get_prices(["SPY"], date(2021, 1, 3), date(2021, 1, 6))
    assert out.shape[0] == 4
    assert out.dates.min() == pd.Timestamp("2021-01-03")
    assert out.dates.max() == pd.Timestamp("2021-01-06")


def test_asset_selection_and_order(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    out = src.get_prices(["AGG", "SPY"], date(2021, 1, 1), date(2021, 1, 12))
    assert out.assets == ["AGG", "SPY"]


def test_explicit_fmt_override(tmp_path: Path) -> None:
    # A parquet file with a non-standard suffix, format forced.
    p = tmp_path / "prices.data"
    _panel().to_parquet(p)
    src = CsvDataSource(p, fmt="parquet")
    out = src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))
    assert out.shape[1] == 1


def test_factory_callable(csv_path: Path) -> None:
    assert isinstance(csv_data_source(csv_path), CsvDataSource)


def test_assets_and_dates_properties(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    assert set(src.assets) == {"SPY", "AGG"}
    assert isinstance(src.dates, pd.DatetimeIndex)
    assert len(src.dates) == 12


# ---------------------------------------------------------------------------
# Missing-data policy
# ---------------------------------------------------------------------------


def test_forward_fill_interior_gap(tmp_path: Path) -> None:
    frame = _panel()
    gap_date = frame.index[4]
    frame.loc[gap_date, "SPY"] = np.nan
    p = tmp_path / "gap.csv"
    frame.to_csv(p)

    src = CsvDataSource(p, forward_fill=True)
    out = src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))
    # The gap inherits the previous day's price (index 3 -> 103.0).
    assert out.frame["SPY"].iloc[4] == pytest.approx(103.0)


def test_forward_fill_from_outside_window(tmp_path: Path) -> None:
    # A gap on the first day of the window is filled from a prior (out-of-window)
    # observation because the fill runs on the full panel before slicing.
    frame = _panel()
    frame.loc[frame.index[5], "AGG"] = np.nan
    p = tmp_path / "gap2.csv"
    frame.to_csv(p)
    src = CsvDataSource(p, forward_fill=True)
    out = src.get_prices(["AGG"], date(2021, 1, 6), date(2021, 1, 12))
    # Window starts at index 5 (the gap); filled from index 4 value 52.0.
    assert out.frame["AGG"].iloc[0] == pytest.approx(52.0)


def test_leading_nan_errors_by_default(tmp_path: Path) -> None:
    frame = _panel()
    frame.loc[frame.index[0], "SPY"] = np.nan
    frame.loc[frame.index[1], "SPY"] = np.nan
    p = tmp_path / "lead.csv"
    frame.to_csv(p)
    src = CsvDataSource(p, on_missing="error")
    with pytest.raises(DataError, match="missing prices"):
        src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))


def test_leading_nan_drop_policy(tmp_path: Path) -> None:
    frame = _panel()
    frame.loc[frame.index[0], "SPY"] = np.nan
    frame.loc[frame.index[1], "SPY"] = np.nan
    p = tmp_path / "lead2.csv"
    frame.to_csv(p)
    src = CsvDataSource(p, on_missing="drop")
    out = src.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2021, 1, 12))
    assert out.assets == ["AGG"]


def test_all_dropped_errors(tmp_path: Path) -> None:
    frame = _panel()
    frame.iloc[0:2, :] = np.nan
    p = tmp_path / "alldrop.csv"
    frame.to_csv(p)
    src = CsvDataSource(p, on_missing="drop")
    with pytest.raises(DataError, match="dropped"):
        src.get_prices(["SPY", "AGG"], date(2021, 1, 1), date(2021, 1, 12))


def test_no_forward_fill_leaves_interior_gap(tmp_path: Path) -> None:
    frame = _panel()
    frame.loc[frame.index[4], "SPY"] = np.nan
    p = tmp_path / "nofill.csv"
    frame.to_csv(p)
    src = CsvDataSource(p, forward_fill=False, on_missing="error")
    with pytest.raises(DataError, match="missing prices"):
        src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        CsvDataSource(tmp_path / "nope.csv")


def test_unknown_suffix(tmp_path: Path) -> None:
    p = tmp_path / "prices.xlsx"
    p.write_text("x")
    with pytest.raises(DataError, match="infer format"):
        CsvDataSource(p)


def test_bad_on_missing(csv_path: Path) -> None:
    with pytest.raises(DataError, match="on_missing"):
        CsvDataSource(csv_path, on_missing="bogus")  # type: ignore[arg-type]


def test_unknown_asset(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    with pytest.raises(DataError, match="Unknown assets"):
        src.get_prices(["TLT"], date(2021, 1, 1), date(2021, 1, 12))


def test_empty_assets(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    with pytest.raises(DataError, match="non-empty"):
        src.get_prices([], date(2021, 1, 1), date(2021, 1, 12))


def test_start_after_end(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    with pytest.raises(DataError, match="must not be after"):
        src.get_prices(["SPY"], date(2021, 1, 12), date(2021, 1, 1))


def test_empty_window(csv_path: Path) -> None:
    src = CsvDataSource(csv_path)
    with pytest.raises(DataError, match="No rows"):
        src.get_prices(["SPY"], date(2030, 1, 1), date(2030, 1, 12))


def test_non_positive_price_rejected(tmp_path: Path) -> None:
    frame = _panel()
    frame.loc[frame.index[3], "SPY"] = -5.0
    p = tmp_path / "neg.csv"
    frame.to_csv(p)
    src = CsvDataSource(p)
    with pytest.raises(DataError, match="validation"):
        src.get_prices(["SPY"], date(2021, 1, 1), date(2021, 1, 12))


def test_duplicate_dates_rejected(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"])
    frame = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=dates)
    frame.index.name = "date"
    p = tmp_path / "dup.csv"
    frame.to_csv(p)
    with pytest.raises(DataError, match="Duplicate dates"):
        CsvDataSource(p)
