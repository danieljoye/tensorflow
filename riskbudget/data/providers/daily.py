"""Long-span **daily** multi-asset deep-history adapter (BUILD_PLAN §8).

Where :mod:`riskbudget.data.providers.shiller` / :mod:`riskbudget.data.providers.gold`
serve *monthly* deep history, this adapter serves the longest **daily** multi-asset
panel reachable from the GitHub-only allowlist of this build environment. It mirrors
the same two-transport pattern as the Tiingo / Shiller / gold providers:

- :meth:`DailyPanelDataSource.from_fixtures` — **offline**, reads a committed,
  pre-assembled daily panel CSV under ``examples/data/daily_panel_long.csv``
  (``Date`` index + ``STOCKS`` / ``BONDS`` / ``GOLD`` columns). This is what the
  test-suite and the offline example use; it needs no network.
- :meth:`DailyPanelDataSource.from_github` — **live**, downloads three daily
  source CSVs from GitHub raw mirrors (``requests`` imported lazily, falling back
  to ``urllib``) and re-assembles the identical panel via :func:`assemble_panel`.

Both transports converge on the same in-memory frame; the offline tests therefore
exercise the real alignment/derivation logic, only the *transport* differs.

Sources (all verified fetchable via ``raw.githubusercontent.com`` in this sandbox)
---------------------------------------------------------------------------------
- ``STOCKS`` — S&P 500 / SPX **daily close**, true daily granularity from
  **1885-01-01** (Stooq ``^spx`` reconstruction mirrored at
  ``ai357060/flower``: ``Data/spx_d.csv``). Pre-1885 rows in that file are
  monthly-spaced backfill and are dropped.
- ``BONDS`` — a 10-year Treasury **total-return PROXY** built from the daily
  ``DGS10`` constant-maturity yield (FRED series, daily from **1962-01-02**,
  mirrored at ``juanfp02/commodities_and_sovereigns``: ``data/DGS10.csv``). The
  daily return is the carry minus a duration-scaled price move::

      r_t = y_{prev}/252 - ModDur * (y_t - y_{prev})      (ModDur = 8.0)

  cumulated to a growth index based at 1.0. This is the **same** constant-duration
  (~8y) proxy methodology the Shiller provider documents — it ignores convexity,
  roll-down and the exact bond cash-flow schedule. Use it for illustrative
  multi-asset risk-budgeting demos, not production bond analytics.
- ``GOLD`` — daily gold spot price (USD/oz), daily from **1968-01-02**
  (LBMA/London-fix daily series mirrored at ``UtaHagen/PortfolioProject``:
  ``Data/economic_indicators_daily.csv``).

The aligned (inner-join) daily panel spans **1968-01-02 -> 2023-12-28**
(~13.7k rows); the start date is bounded by the gold series.

References
----------
- ``DataSource`` protocol: :mod:`riskbudget.core.interfaces`.
- Ranked source findings + caveats: :doc:`docs/daily-data-sources.md`.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData

# Default committed fixture (offline development / test path).
_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "examples" / "data" / "daily_panel_long.csv"

# Live GitHub raw mirrors of the three daily source series.
_STOCKS_URL = "https://raw.githubusercontent.com/ai357060/flower/master/Data/spx_d.csv"
_DGS10_URL = (
    "https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/DGS10.csv"
)
_GOLD_URL = (
    "https://raw.githubusercontent.com/UtaHagen/PortfolioProject/main/Data/"
    "economic_indicators_daily.csv"
)

# Modified duration used by the BONDS total-return proxy (years). Documented above.
_BOND_MOD_DUR = 8.0

# True-daily granularity in the SPX reconstruction begins here; earlier rows in the
# source file are monthly-spaced backfill and are dropped before assembly.
_STOCKS_DAILY_START = pd.Timestamp("1885-01-01")

# The assets this source serves (panel column order).
_ASSETS = ("STOCKS", "BONDS", "GOLD")


def _parse_fixture(text: str) -> pd.DataFrame:
    """Parse the committed pre-assembled daily panel CSV.

    Parameters
    ----------
    text:
        Full CSV payload: a ``Date`` header column followed by the
        :data:`_ASSETS` columns, one daily row each.

    Returns
    -------
    pandas.DataFrame
        Date-indexed (tz-naive), ascending, de-duplicated, with the
        :data:`_ASSETS` columns as strictly-positive level series.

    Raises
    ------
    DataError
        If the payload is empty, missing a required column, has unparseable
        values, or yields an empty frame.
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise DataError("Daily panel CSV is empty (no header row).")
    required = {"Date", *_ASSETS}
    missing = required.difference(reader.fieldnames)
    if missing:
        raise DataError(
            f"Daily panel CSV is missing required columns: {sorted(missing)}. "
            f"Found: {reader.fieldnames!r}."
        )

    dates: list[pd.Timestamp] = []
    cols: dict[str, list[float]] = {a: [] for a in _ASSETS}
    for row in reader:
        raw_date = row.get("Date")
        if not raw_date:
            continue
        try:
            ts = pd.Timestamp(raw_date).normalize()
            values = {a: float(row[a]) for a in _ASSETS}
        except (TypeError, ValueError) as exc:
            raise DataError(
                f"Daily panel CSV row could not be parsed ({raw_date!r}): {exc}"
            ) from exc
        dates.append(ts)
        for a in _ASSETS:
            cols[a].append(values[a])

    if not dates:
        raise DataError("Daily panel CSV contained no usable rows.")

    frame = pd.DataFrame(cols, index=pd.DatetimeIndex(dates))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise DataError("Daily panel is empty after parsing.")
    return frame.loc[:, list(_ASSETS)]


def _parse_two_col(text: str, value_name: str) -> pd.Series:
    """Parse a generic two-column ``date,value`` daily CSV into a Series.

    The first column is always treated as the date; the *second* column is the
    value (so this tolerates the different header names the three mirrors use,
    e.g. ``Close``/``DGS10``/``Gold Price``). Rows whose value is blank or the
    FRED ``.`` placeholder are skipped.
    """
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    if not rows:
        raise DataError(f"{value_name} CSV is empty (no header row).")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows[1:]:
        if len(row) < 2 or not row[0]:
            continue
        raw = row[1].strip()
        if raw in ("", "."):
            continue
        try:
            ts = pd.Timestamp(row[0]).normalize()
            val = float(raw)
        except (TypeError, ValueError):
            continue
        dates.append(ts)
        values.append(val)
    if not dates:
        raise DataError(f"{value_name} CSV contained no usable rows.")
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=value_name)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _stooq_spx_close(text: str) -> pd.Series:
    """Extract the daily SPX close from the Stooq ``spx_d.csv`` mirror.

    The file's columns are ``Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen``
    (Polish OHLCV); the close is the 5th column (index 4).
    """
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    if not rows:
        raise DataError("SPX CSV is empty (no header row).")
    dates: list[pd.Timestamp] = []
    closes: list[float] = []
    for row in rows[1:]:
        if len(row) < 5 or not row[0]:
            continue
        try:
            ts = pd.Timestamp(row[0]).normalize()
            close = float(row[4])
        except (TypeError, ValueError):
            continue
        dates.append(ts)
        closes.append(close)
    if not dates:
        raise DataError("SPX CSV contained no usable rows.")
    series = pd.Series(closes, index=pd.DatetimeIndex(dates), name="STOCKS")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series[series.index >= _STOCKS_DAILY_START]


def _bond_tr_from_yield(yield_pct: pd.Series) -> pd.Series:
    """Build the constant-duration 10Y total-return proxy from a daily yield (%).

    See the module docstring for the formula; ``ModDur`` is :data:`_BOND_MOD_DUR`.
    """
    y = yield_pct / 100.0
    daily_ret = (y.shift(1) / 252.0 - _BOND_MOD_DUR * (y - y.shift(1))).fillna(0.0)
    return (1.0 + daily_ret).cumprod().rename("BONDS")


def assemble_panel(stocks_text: str, dgs10_text: str, gold_text: str) -> pd.DataFrame:
    """Assemble the aligned daily panel from the three raw source payloads.

    This is the shared derivation used by the **live** transport; the offline
    transport reads the already-assembled result. The ``BONDS`` proxy is rebased
    to 1.0 on the first row of the aligned (inner-join) panel.

    Raises
    ------
    DataError
        If any source parses empty or the inner join leaves no overlapping dates.
    """
    stocks = _stooq_spx_close(stocks_text)
    bonds = _bond_tr_from_yield(_parse_two_col(dgs10_text, "DGS10"))
    gold = _parse_two_col(gold_text, "GOLD")

    panel = pd.concat([stocks, bonds, gold], axis=1, join="inner").dropna(how="any")
    if panel.empty:
        raise DataError("Daily panel is empty after aligning STOCKS/BONDS/GOLD.")
    panel = panel.loc[:, list(_ASSETS)].copy()
    panel["BONDS"] = panel["BONDS"] / panel["BONDS"].iloc[0]
    panel.index.name = "Date"
    return panel


class DailyPanelDataSource:
    """A :class:`~riskbudget.core.interfaces.DataSource` over the long daily panel.

    Two transports, one logical panel:

    - **fixture** (offline): reads the committed pre-assembled CSV.
    - **github** (network): downloads the three daily sources and re-assembles.

    Parameters
    ----------
    fixture_path:
        Path to a committed panel CSV. When set, the adapter reads from disk and
        never touches the network.
    urls:
        ``(stocks, dgs10, gold)`` live CSV URLs (github transport). Ignored when
        ``fixture_path`` is set.
    timeout:
        Per-request timeout in seconds for the live transport.
    """

    def __init__(
        self,
        *,
        fixture_path: Path | str | None = None,
        urls: tuple[str, str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._fixture_path = Path(fixture_path) if fixture_path is not None else None
        self._urls = urls
        self._timeout = timeout
        if self._fixture_path is None and self._urls is None:
            raise DataError(
                "DailyPanelDataSource needs a fixture_path (offline) or urls (live). "
                "Use DailyPanelDataSource.from_fixtures() for the offline cache."
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_fixtures(cls, fixture_path: Path | str | None = None) -> DailyPanelDataSource:
        """Build an **offline** adapter reading the committed panel CSV."""
        path = Path(fixture_path) if fixture_path is not None else _FIXTURE_PATH
        if not path.is_file():
            raise DataError(f"Daily panel fixture not found: {path}")
        return cls(fixture_path=path)

    @classmethod
    def from_github(
        cls,
        *,
        urls: tuple[str, str, str] | None = None,
        timeout: float = 30.0,
    ) -> DailyPanelDataSource:
        """Build a **live** adapter that downloads + re-assembles from GitHub."""
        return cls(urls=urls or (_STOCKS_URL, _DGS10_URL, _GOLD_URL), timeout=timeout)

    # -- DataSource protocol -----------------------------------------------

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a :class:`PriceData` panel of ``assets`` over ``[start, end]``.

        Raises
        ------
        DataError
            If ``assets`` is empty/duplicated/unknown, ``start > end``, or the
            window leaves no overlapping dates.
        """
        if not assets:
            raise DataError("get_prices requires at least one asset.")
        requested = [str(a) for a in assets]
        if len(set(requested)) != len(requested):
            raise DataError(f"Duplicate assets requested: {requested}")
        unknown = [a for a in requested if a not in _ASSETS]
        if unknown:
            raise DataError(f"Unknown daily-panel asset(s): {unknown}. Available: {list(_ASSETS)}.")
        if start > end:
            raise DataError(f"start ({start}) must not be after end ({end}).")

        frame = self._load_panel()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts), requested]
        window = window.dropna(how="any")
        if window.empty:
            raise DataError(
                f"No daily-panel data for {requested} in [{start}, {end}] "
                f"(available {frame.index.min().date()}..{frame.index.max().date()})."
            )
        return PriceData(window.loc[:, requested])

    def available_assets(self) -> Sequence[str]:
        """List the assets this source can serve."""
        return list(_ASSETS)

    # -- transports ---------------------------------------------------------

    def _load_panel(self) -> pd.DataFrame:
        """Return the assembled panel from the fixture or the live URLs."""
        if self._fixture_path is not None:
            return _parse_fixture(self._read_fixture())
        return self._fetch_live()

    def _read_fixture(self) -> str:
        """Load the committed panel CSV from disk."""
        assert self._fixture_path is not None  # narrowed by caller
        if not self._fixture_path.is_file():
            raise DataError(f"Daily panel fixture not found: {self._fixture_path}")
        try:
            return self._fixture_path.read_text()
        except OSError as exc:
            raise DataError(
                f"Could not read daily panel fixture {self._fixture_path}: {exc}"
            ) from exc

    def _fetch_live(self) -> pd.DataFrame:
        """Download the three sources and re-assemble. Network path only."""
        assert self._urls is not None  # narrowed by caller
        stocks_text, dgs10_text, gold_text = (self._download(u) for u in self._urls)
        return assemble_panel(stocks_text, dgs10_text, gold_text)

    def _download(self, url: str) -> str:
        """Download one CSV. ``requests`` is imported lazily (urllib fallback)."""
        try:  # pragma: no cover - network path unreachable in tests
            import requests

            response = requests.get(url, timeout=self._timeout)
            response.raise_for_status()
            return str(response.text)
        except ImportError:  # pragma: no cover - exercised only live
            import urllib.request

            try:
                with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                    return str(resp.read().decode("utf-8"))
            except OSError as exc:
                raise DataError(f"Daily panel live download failed ({url}): {exc}") from exc
        except Exception as exc:  # pragma: no cover - network path
            raise DataError(f"Daily panel live download failed ({url}): {exc}") from exc


def daily_panel_data_source(
    *,
    fixture_path: Path | str | None = None,
    live: bool = False,
    urls: tuple[str, str, str] | None = None,
) -> DailyPanelDataSource:
    """Registry-friendly factory (BUILD_PLAN §5.2) for :class:`DailyPanelDataSource`.

    With no arguments it returns the committed offline fixture adapter. Pass
    ``live=True`` (or ``urls``) for the GitHub transport.
    """
    if live or urls is not None:
        return DailyPanelDataSource.from_github(urls=urls)
    return DailyPanelDataSource.from_fixtures(fixture_path)


__all__ = [
    "DailyPanelDataSource",
    "assemble_panel",
    "daily_panel_data_source",
]
