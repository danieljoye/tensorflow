"""Gold monthly spot-price deep-history adapter (BUILD_PLAN §8; deep-history backbone).

Exposes the datahub monthly gold-price series (1833-present) as a first-class
:class:`~riskbudget.core.interfaces.DataSource`, mirroring the
:mod:`riskbudget.data.providers.tiingo` / :mod:`riskbudget.data.providers.shiller`
two-transport pattern:

- :meth:`GoldDataSource.from_fixtures` — **offline**, reads a committed trimmed
  slice of the real CSV under ``providers/fixtures/gold/``.
- :meth:`GoldDataSource.from_github` — **live**, downloads the full CSV from the
  datahub GitHub mirror (``requests`` imported lazily, falling back to
  ``urllib``).

Both transports feed one parser (:func:`_parse_table`), so the offline tests
exercise the identical code that runs live.

Source CSV
----------
``https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv``
with columns ``Date`` (``YYYY-MM``) and ``Price`` (USD/oz, monthly average).

Asset
-----
- ``GOLD`` — the monthly gold spot price (USD per troy ounce).

References
----------
- Gold prices (datahub mirror): https://github.com/datasets/gold-prices
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
_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "gold" / "monthly.csv"

# Live datahub mirror of the monthly gold series.
_GITHUB_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"

_COL_DATE = "Date"
_COL_PRICE = "Price"

# The single asset this source serves.
_ASSET = "GOLD"
_ASSETS = (_ASSET,)


def _parse_table(text: str) -> pd.DataFrame:
    """Parse the raw gold CSV into a date-indexed single-column price frame.

    Parameters
    ----------
    text:
        The full CSV payload (header + monthly rows).

    Returns
    -------
    pandas.DataFrame
        Indexed by (tz-naive, month-start) date, ascending and de-duplicated,
        with a single strictly-positive ``GOLD`` column.

    Raises
    ------
    DataError
        If the payload is empty, missing a required column, has unparseable
        values, or yields an empty series.
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise DataError("Gold CSV is empty (no header row).")
    missing = {_COL_DATE, _COL_PRICE}.difference(reader.fieldnames)
    if missing:
        raise DataError(
            f"Gold CSV is missing required columns: {sorted(missing)}. "
            f"Found: {reader.fieldnames!r}."
        )

    dates: list[pd.Timestamp] = []
    prices: list[float] = []
    for row in reader:
        raw_date = row.get(_COL_DATE)
        if not raw_date:
            continue
        try:
            ts = pd.Timestamp(raw_date).normalize()
            px = float(row[_COL_PRICE])
        except (TypeError, ValueError) as exc:
            raise DataError(f"Gold CSV row could not be parsed ({raw_date!r}): {exc}") from exc
        dates.append(ts)
        prices.append(px)

    if not dates:
        raise DataError("Gold CSV contained no usable monthly rows.")

    frame = pd.DataFrame({_ASSET: prices}, index=pd.DatetimeIndex(dates))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise DataError("Gold price panel is empty after parsing.")
    return frame


class GoldDataSource:
    """A :class:`~riskbudget.core.interfaces.DataSource` over monthly gold spot prices.

    Two transports, one parser:

    - **fixture** (offline): reads the committed trimmed CSV.
    - **github** (network): downloads the full CSV from the datahub mirror.

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
                "GoldDataSource needs a fixture_path (offline) or a url (live). "
                "Use GoldDataSource.from_fixtures() for the offline cache."
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_fixtures(cls, fixture_path: Path | str | None = None) -> GoldDataSource:
        """Build an **offline** adapter reading the committed trimmed CSV."""
        path = Path(fixture_path) if fixture_path is not None else _FIXTURE_PATH
        if not path.is_file():
            raise DataError(f"Gold fixture not found: {path}")
        return cls(fixture_path=path)

    @classmethod
    def from_github(cls, *, url: str | None = None, timeout: float = 30.0) -> GoldDataSource:
        """Build a **live** adapter that downloads the full CSV from GitHub."""
        return cls(url=url or _GITHUB_URL, timeout=timeout)

    # -- DataSource protocol -----------------------------------------------

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a :class:`PriceData` panel of ``GOLD`` prices over ``[start, end]``.

        Raises
        ------
        DataError
            If ``assets`` is empty/duplicated/unknown, ``start > end``, or the
            window leaves no rows.
        """
        if not assets:
            raise DataError("get_prices requires at least one asset.")
        requested = [str(a) for a in assets]
        if len(set(requested)) != len(requested):
            raise DataError(f"Duplicate assets requested: {requested}")
        unknown = [a for a in requested if a not in _ASSETS]
        if unknown:
            raise DataError(f"Unknown Gold asset(s): {unknown}. Available: {list(_ASSETS)}.")
        if start > end:
            raise DataError(f"start ({start}) must not be after end ({end}).")

        frame = _parse_table(self._fetch_text())
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts), requested]
        window = window.dropna(how="any")
        if window.empty:
            raise DataError(
                f"No Gold data for {requested} in [{start}, {end}] "
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
            raise DataError(f"Gold fixture not found: {self._fixture_path}")
        try:
            return self._fixture_path.read_text()
        except OSError as exc:
            raise DataError(f"Could not read Gold fixture {self._fixture_path}: {exc}") from exc

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
                raise DataError(f"Gold live download failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - network path
            raise DataError(f"Gold live download failed: {exc}") from exc

    def available_assets(self) -> Sequence[str]:
        """List the asset(s) this source can serve."""
        return list(_ASSETS)


def gold_data_source(
    *,
    fixture_path: Path | str | None = None,
    live: bool = False,
    url: str | None = None,
) -> GoldDataSource:
    """Registry-friendly factory (BUILD_PLAN §5.2) for :class:`GoldDataSource`.

    With no arguments it returns the committed offline fixture adapter. Pass
    ``live=True`` (or a ``url``) for the GitHub transport.
    """
    if live or url is not None:
        return GoldDataSource.from_github(url=url)
    return GoldDataSource.from_fixtures(fixture_path)


__all__ = [
    "GoldDataSource",
    "gold_data_source",
]
