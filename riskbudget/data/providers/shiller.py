"""Robert Shiller / S&P 500 deep-history adapter (BUILD_PLAN §8; deep-history backbone).

This adapter exposes the monthly Shiller S&P 500 dataset (1871-present) as a
first-class :class:`~riskbudget.core.interfaces.DataSource`. It mirrors the
:mod:`riskbudget.data.providers.tiingo` two-transport pattern exactly:

- :meth:`ShillerDataSource.from_fixtures` — **offline**, reads a committed
  trimmed slice of the real CSV under ``providers/fixtures/shiller/``. This is
  what the test-suite and the offline example use; it needs no network.
- :meth:`ShillerDataSource.from_github` — **live**, downloads the full CSV from
  the datahub GitHub mirror (``requests`` imported lazily, falling back to
  ``urllib``).

Both transports feed one parser (:func:`_parse_table`), so the offline tests
exercise the identical derivation code that runs live; only the *transport*
differs.

Source CSV
----------
``https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv``
with monthly columns::

    Date, SP500, Dividend, Earnings, Consumer Price Index, Long Interest Rate,
    Real Price, Real Dividend, Real Earnings, PE10

Derived "assets" (selectable via :meth:`get_prices`)
----------------------------------------------------
- ``SP500_NOMINAL`` — the nominal S&P 500 index level (``SP500`` column).
- ``SP500_REAL`` — the CPI-adjusted (real) S&P 500 level (``Real Price``).
- ``SP500_TR`` — a **total-return** index built from price *and* dividends:
  the monthly simple return is ``(P_t + D_t/12) / P_{t-1} - 1`` (the annualized
  ``Dividend`` figure is paid in twelfths each month), cumulated to a growth
  index based at 1.0 on the first available month.
- ``US10Y_TR`` — a 10-year Treasury total-return **PROXY** derived from the
  ``Long Interest Rate`` column ``y`` (a long government-bond yield, in
  percent). The monthly return is the carry minus a duration-scaled price move::

      r_t = y_{prev}/100/12 - ModDur * (y_t - y_{prev})/100      (ModDur = 8.0)

  cumulated to a growth index based at 1.0. **This is a constant-duration
  (≈8y) proxy, NOT an actual bond-index total return**: it ignores convexity,
  roll-down, the exact bond cash-flow schedule, and the difference between the
  reported long yield and an on-the-run 10Y. Use it for illustrative
  multi-asset risk-budgeting demos, not for production bond analytics.

References
----------
- Shiller data (datahub mirror): https://github.com/datasets/s-and-p-500
- ``DataSource`` protocol: :mod:`riskbudget.core.interfaces`.
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
_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "shiller" / "sp500.csv"

# Live datahub mirror of the Shiller monthly dataset.
_GITHUB_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

# Modified duration used by the US10Y_TR proxy (years). Documented above.
_US10Y_MOD_DUR = 8.0

# Raw CSV columns we depend on.
_COL_DATE = "Date"
_COL_SP500 = "SP500"
_COL_REAL = "Real Price"
_COL_DIVIDEND = "Dividend"
_COL_RATE = "Long Interest Rate"

# Derived assets this source can serve.
_DERIVED_ASSETS = ("SP500_NOMINAL", "SP500_REAL", "SP500_TR", "US10Y_TR")


def _parse_table(text: str) -> pd.DataFrame:
    """Parse the raw Shiller CSV into a date-indexed frame of derived series.

    Parameters
    ----------
    text:
        The full CSV payload (header + monthly rows).

    Returns
    -------
    pandas.DataFrame
        Indexed by (tz-naive, month-start) date, ascending and de-duplicated,
        with one column per entry in :data:`_DERIVED_ASSETS`. All columns are
        strictly positive price/level series.

    Raises
    ------
    DataError
        If the payload is empty, missing a required column, has unparseable
        values, or yields an empty series.
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise DataError("Shiller CSV is empty (no header row).")
    required = {_COL_DATE, _COL_SP500, _COL_REAL, _COL_DIVIDEND, _COL_RATE}
    missing = required.difference(reader.fieldnames)
    if missing:
        raise DataError(
            f"Shiller CSV is missing required columns: {sorted(missing)}. "
            f"Found: {reader.fieldnames!r}."
        )

    dates: list[pd.Timestamp] = []
    sp500: list[float] = []
    real: list[float] = []
    dividend: list[float] = []
    rate: list[float] = []
    for row in reader:
        raw_date = row.get(_COL_DATE)
        if not raw_date:
            continue
        try:
            ts = pd.Timestamp(raw_date).normalize()
            sp = float(row[_COL_SP500])
            rp = float(row[_COL_REAL])
            dv = float(row[_COL_DIVIDEND])
            ir = float(row[_COL_RATE])
        except (TypeError, ValueError) as exc:
            raise DataError(f"Shiller CSV row could not be parsed ({raw_date!r}): {exc}") from exc
        dates.append(ts)
        sp500.append(sp)
        real.append(rp)
        dividend.append(dv)
        rate.append(ir)

    if not dates:
        raise DataError("Shiller CSV contained no usable monthly rows.")

    raw = pd.DataFrame(
        {
            _COL_SP500: sp500,
            _COL_REAL: real,
            _COL_DIVIDEND: dividend,
            _COL_RATE: rate,
        },
        index=pd.DatetimeIndex(dates),
    )
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    price = raw[_COL_SP500]
    prev_price = price.shift(1)
    # Monthly total return: price change plus 1/12 of the annualized dividend.
    tr_ret = (price + raw[_COL_DIVIDEND] / 12.0) / prev_price - 1.0
    sp500_tr = (1.0 + tr_ret.fillna(0.0)).cumprod()

    # US10Y total-return proxy (constant ModDur). See module docstring.
    yld = raw[_COL_RATE] / 100.0
    carry = yld.shift(1) / 12.0
    price_move = -_US10Y_MOD_DUR * (yld - yld.shift(1))
    bond_ret = carry + price_move
    us10y_tr = (1.0 + bond_ret.fillna(0.0)).cumprod()

    frame = pd.DataFrame(
        {
            "SP500_NOMINAL": price,
            "SP500_REAL": raw[_COL_REAL],
            "SP500_TR": sp500_tr,
            "US10Y_TR": us10y_tr,
        }
    )
    if frame.empty:
        raise DataError("Shiller derived panel is empty after parsing.")
    return frame


class ShillerDataSource:
    """A :class:`~riskbudget.core.interfaces.DataSource` over Shiller S&P 500 data.

    Two transports, one parser:

    - **fixture** (offline): reads the committed trimmed CSV.
    - **github** (network): downloads the full CSV from the datahub mirror.

    The transport is selected by the constructor used; everything after the raw
    CSV text is shared, so the offline tests cover the live derivation logic.

    Parameters
    ----------
    fixture_path:
        Path to a committed CSV. When set, the adapter reads from disk and never
        touches the network.
    url:
        Live CSV URL (github transport). Ignored when ``fixture_path`` is set.
    timeout:
        Per-request timeout in seconds for the live transport.
    """

    def __init__(
        self,
        *,
        fixture_path: Path | str | None = None,
        url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._fixture_path = Path(fixture_path) if fixture_path is not None else None
        self._url = url
        self._timeout = timeout
        if self._fixture_path is None and not self._url:
            raise DataError(
                "ShillerDataSource needs a fixture_path (offline) or a url (live). "
                "Use ShillerDataSource.from_fixtures() for the offline cache."
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_fixtures(cls, fixture_path: Path | str | None = None) -> ShillerDataSource:
        """Build an **offline** adapter reading the committed trimmed CSV.

        Defaults to the package's committed fixture so it works with no
        arguments, no network, and no key.
        """
        path = Path(fixture_path) if fixture_path is not None else _FIXTURE_PATH
        if not path.is_file():
            raise DataError(f"Shiller fixture not found: {path}")
        return cls(fixture_path=path)

    @classmethod
    def from_github(cls, *, url: str | None = None, timeout: float = 30.0) -> ShillerDataSource:
        """Build a **live** adapter that downloads the full CSV from GitHub.

        Requires a network path to ``raw.githubusercontent.com``.
        """
        return cls(url=url or _GITHUB_URL, timeout=timeout)

    # -- DataSource protocol -----------------------------------------------

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a :class:`PriceData` panel of derived series for ``assets``.

        Restricts to ``[start, end]`` inclusive and aligns all requested series
        on the common monthly index.

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
        unknown = [a for a in requested if a not in _DERIVED_ASSETS]
        if unknown:
            raise DataError(
                f"Unknown Shiller asset(s): {unknown}. Available: {list(_DERIVED_ASSETS)}."
            )
        if start > end:
            raise DataError(f"start ({start}) must not be after end ({end}).")

        frame = _parse_table(self._fetch_text())
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts), requested]
        window = window.dropna(how="any")
        if window.empty:
            raise DataError(
                f"No Shiller data for {requested} in [{start}, {end}] "
                f"(available {frame.index.min().date()}..{frame.index.max().date()})."
            )
        return PriceData(window.loc[:, requested])

    # -- transports ---------------------------------------------------------

    def _fetch_text(self) -> str:
        """Return the raw CSV text from the fixture or the live URL."""
        if self._fixture_path is not None:
            return self._read_fixture()
        return self._fetch_live()

    def _read_fixture(self) -> str:
        """Load the committed CSV from disk."""
        assert self._fixture_path is not None  # narrowed by caller
        if not self._fixture_path.is_file():
            raise DataError(f"Shiller fixture not found: {self._fixture_path}")
        try:
            return self._fixture_path.read_text()
        except OSError as exc:
            raise DataError(f"Could not read Shiller fixture {self._fixture_path}: {exc}") from exc

    def _fetch_live(self) -> str:
        """Download the live CSV. ``requests`` is imported lazily (urllib fallback)."""
        assert self._url is not None  # narrowed by caller
        try:  # pragma: no cover - network path unreachable in tests
            import requests

            response = requests.get(self._url, timeout=self._timeout)
            response.raise_for_status()
            return str(response.text)
        except ImportError:  # pragma: no cover - exercised only live
            import urllib.request

            try:
                with urllib.request.urlopen(self._url, timeout=self._timeout) as resp:
                    return str(resp.read().decode("utf-8"))
            except OSError as exc:
                raise DataError(f"Shiller live download failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network path
            raise DataError(f"Shiller live download failed: {exc}") from exc

    def available_assets(self) -> Sequence[str]:
        """List the derived series this source can serve."""
        return list(_DERIVED_ASSETS)


def shiller_data_source(
    *,
    fixture_path: Path | str | None = None,
    live: bool = False,
    url: str | None = None,
) -> ShillerDataSource:
    """Registry-friendly factory (BUILD_PLAN §5.2) for :class:`ShillerDataSource`.

    With no arguments it returns the committed offline fixture adapter, so
    ``shiller_data_source()`` always yields a working (offline) ``DataSource``.
    Pass ``live=True`` (or a ``url``) for the GitHub transport.
    """
    if live or url is not None:
        return ShillerDataSource.from_github(url=url)
    return ShillerDataSource.from_fixtures(fixture_path)


__all__ = [
    "ShillerDataSource",
    "shiller_data_source",
]
