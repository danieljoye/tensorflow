"""Tiingo end-of-day price adapter implementing the :class:`DataSource` protocol.

Tiingo (https://www.tiingo.com) is the recommended first *real* market-data
provider for this system (see ``docs/data-sources.md``). Its end-of-day REST API
returns clean JSON with both raw and **split/dividend-adjusted** OHLCV fields, a
generous free tier, deep history (decades for US equities/ETFs), and an explicit
API-key auth model — so we never scrape an undocumented endpoint the way the
yfinance fallback does.

Network reality in this build environment
-----------------------------------------
Outbound network here is an **allowlist** that resolves only PyPI, PythonHosted
and GitHub. Every market-data host — Tiingo included — returns
``403 Host not in allowlist`` (confirmed empirically; see
``docs/data-sources.md`` §"Network probe"). The adapter is therefore developed
and tested against a small, committed *recorded* fixture under
``providers/fixtures/tiingo/`` that mirrors the exact JSON shape Tiingo returns
live. Construct it with:

- :meth:`TiingoDataSource.from_fixtures` — offline, reads the recorded cache.
  This is what the test-suite uses and it requires no network or API key.
- :meth:`TiingoDataSource.live` (or the default constructor with an
  ``api_key``) — wires the same parsing path to the real HTTPS endpoint via
  ``requests``; it works wherever Tiingo's host is reachable.

The two paths share one parser (:func:`_parse_records`), so the offline tests
exercise the identical code that runs live; only the *transport* differs.

Live request shape (documented for reproducibility)
---------------------------------------------------
For each ``symbol`` the adapter issues::

    GET https://api.tiingo.com/tiingo/daily/{symbol}/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&format=json
        &columns=date,close,adjClose
    Authorization: Token <api_key>

and reads the JSON array of daily records, taking the **adjusted close**
(``adjClose``) as the price series so downstream returns are corporate-action
consistent (BUILD_PLAN §3.1, "corporate-action / adjusted-close handling").

References
----------
- Tiingo End-of-Day API docs: https://www.tiingo.com/documentation/end-of-day
- ``DataSource`` protocol: :mod:`riskbudget.core.interfaces`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData

# Default fixture cache shipped with the package (offline development path).
_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "tiingo"

# Live endpoint template + the price field we treat as the canonical "price".
_BASE_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
_PRICE_FIELD = "adjClose"


def _parse_records(
    symbol: str,
    records: Iterable[Mapping[str, Any]],
) -> pd.Series:
    """Turn a Tiingo daily-price JSON array into a date-indexed price Series.

    Parameters
    ----------
    symbol:
        Asset id the records belong to (used only for error messages / naming).
    records:
        Iterable of per-day mappings as returned by Tiingo, each carrying at
        least a ``date`` and an :data:`_PRICE_FIELD` (``adjClose``) key.

    Returns
    -------
    pandas.Series
        Adjusted close indexed by (tz-naive, normalized) date, ascending and
        de-duplicated, named ``symbol``.

    Raises
    ------
    DataError
        If the payload is not a list of records, a record is missing the
        ``date``/``adjClose`` fields, a value is unparseable, or the series ends
        up empty.
    """
    dates: list[pd.Timestamp] = []
    prices: list[float] = []
    record_list = list(records)
    if not record_list:
        raise DataError(f"Tiingo returned no price records for {symbol!r}.")

    for row in record_list:
        if not isinstance(row, Mapping):
            raise DataError(f"Tiingo record for {symbol!r} is not an object: {row!r}.")
        if "date" not in row or _PRICE_FIELD not in row:
            raise DataError(
                f"Tiingo record for {symbol!r} is missing 'date' or {_PRICE_FIELD!r}: "
                f"keys={sorted(row.keys())!r}."
            )
        try:
            ts = pd.Timestamp(row["date"])
            price = float(row[_PRICE_FIELD])
        except (TypeError, ValueError) as exc:
            raise DataError(f"Tiingo record for {symbol!r} could not be parsed: {exc}") from exc
        # Tiingo stamps midnight-UTC; normalize to a tz-naive calendar date so
        # panels from different assets align and PriceData stays comparable.
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        dates.append(ts.normalize())
        prices.append(price)

    series = pd.Series(prices, index=pd.DatetimeIndex(dates), name=symbol)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise DataError(f"Tiingo price series for {symbol!r} is empty after parsing.")
    return series


class TiingoDataSource:
    """A :class:`~riskbudget.core.interfaces.DataSource` backed by Tiingo EOD prices.

    Two transports, one parser:

    - **fixture** (offline): reads ``{fixture_root}/{symbol}.json`` files — the
      committed recording used by the test-suite and offline development.
    - **live** (network): issues HTTPS GETs against Tiingo with an API key.

    The transport is selected by which constructor you use; everything after the
    raw JSON is shared, so the offline tests cover the live parsing logic.

    Parameters
    ----------
    api_key:
        Tiingo API token. Required for the live transport, ignored for fixtures.
    fixture_root:
        Directory of recorded ``{symbol}.json`` files. When set, the adapter
        reads from disk and never touches the network.
    session:
        Optional pre-built ``requests.Session`` (live transport only); a fresh
        one is created lazily if omitted.
    timeout:
        Per-request timeout in seconds for the live transport.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        fixture_root: Path | str | None = None,
        session: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._fixture_root = Path(fixture_root) if fixture_root is not None else None
        self._session = session
        self._timeout = timeout
        if self._fixture_root is None and not self._api_key:
            raise DataError(
                "TiingoDataSource needs either an api_key (live) or a fixture_root "
                "(offline). Use TiingoDataSource.from_fixtures() for the offline cache."
            )

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_fixtures(cls, fixture_root: Path | str | None = None) -> TiingoDataSource:
        """Build an **offline** adapter reading recorded ``{symbol}.json`` files.

        Defaults to the package's committed cache (``providers/fixtures/tiingo``)
        so it works with no arguments, no network, and no API key.
        """
        root = Path(fixture_root) if fixture_root is not None else _FIXTURE_ROOT
        if not root.is_dir():
            raise DataError(f"Tiingo fixture directory not found: {root}")
        return cls(api_key=None, fixture_root=root)

    @classmethod
    def live(cls, api_key: str, *, timeout: float = 30.0) -> TiingoDataSource:
        """Build a **live** adapter that calls the real Tiingo HTTPS endpoint.

        Requires a network path to ``api.tiingo.com`` (blocked in this build
        environment — see the module docstring) and a valid ``api_key``.
        """
        if not api_key:
            raise DataError("TiingoDataSource.live requires a non-empty api_key.")
        return cls(api_key=api_key, timeout=timeout)

    # -- DataSource protocol -----------------------------------------------

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a :class:`PriceData` panel of adjusted closes for ``assets``.

        Fetches each asset (from fixtures or live), restricts to ``[start, end]``
        inclusive, aligns all series on a common date index, and forward-fills
        small gaps before dropping any date still missing a price (mirroring the
        CSV loader's policy so panels are dense and PSD-friendly downstream).

        Raises
        ------
        DataError
            If ``assets`` is empty/duplicated, a symbol cannot be sourced, the
            window leaves no overlapping dates, or the panel cannot be assembled.
        """
        if not assets:
            raise DataError("get_prices requires at least one asset.")
        symbols = [str(a) for a in assets]
        if len(set(symbols)) != len(symbols):
            raise DataError(f"Duplicate assets requested: {symbols}")
        if start > end:
            raise DataError(f"start ({start}) must not be after end ({end}).")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        columns: dict[str, pd.Series] = {}
        for symbol in symbols:
            raw = self._fetch_raw(symbol, start, end)
            series = _parse_records(symbol, raw)
            window = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]
            if window.empty:
                raise DataError(
                    f"No Tiingo prices for {symbol!r} in [{start}, {end}] "
                    f"(available {series.index.min().date()}..{series.index.max().date()})."
                )
            columns[symbol] = window

        frame = pd.DataFrame(columns)
        # Align across assets: forward-fill within each column, then drop any
        # date still missing data (e.g. a symbol that simply did not trade yet).
        frame = frame.sort_index().ffill().dropna(how="any")
        if frame.empty:
            raise DataError(f"No overlapping Tiingo dates across {symbols} in [{start}, {end}].")
        frame.columns = symbols  # preserve requested order
        return PriceData(frame.loc[:, symbols])

    # -- transports ---------------------------------------------------------

    def _fetch_raw(self, symbol: str, start: date, end: date) -> list[Mapping[str, Any]]:
        """Return the raw Tiingo JSON record list for ``symbol`` over the window."""
        if self._fixture_root is not None:
            return self._read_fixture(symbol)
        return self._fetch_live(symbol, start, end)

    def _read_fixture(self, symbol: str) -> list[Mapping[str, Any]]:
        """Load a recorded ``{symbol}.json`` fixture from disk."""
        assert self._fixture_root is not None  # narrowed by caller
        path = self._fixture_root / f"{symbol}.json"
        if not path.is_file():
            raise DataError(
                f"No recorded Tiingo fixture for {symbol!r} at {path}. "
                f"Available: {sorted(p.stem for p in self._fixture_root.glob('*.json'))}."
            )
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"Could not read Tiingo fixture {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise DataError(f"Tiingo fixture {path} must be a JSON array, got {type(payload)}.")
        return payload

    def _fetch_live(self, symbol: str, start: date, end: date) -> list[Mapping[str, Any]]:
        """Issue the live HTTPS GET to Tiingo and return the parsed JSON list.

        ``requests`` is imported lazily so the package — and its offline tests —
        never depend on it being installed unless the live path is actually used.
        """
        try:
            import requests  # lazy import: keep it off the offline path
        except ImportError as exc:  # pragma: no cover - exercised only live
            raise DataError(
                "The live Tiingo transport needs the 'requests' package; install it "
                "or use TiingoDataSource.from_fixtures() for the offline cache."
            ) from exc

        session = self._session or requests.Session()
        url = _BASE_URL.format(symbol=symbol)
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
            "columns": "date,close,adjClose",
        }
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }
        try:  # pragma: no cover - network path unreachable in this environment
            response = session.get(url, params=params, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover
            raise DataError(f"Tiingo request for {symbol!r} failed: {exc}") from exc
        if not isinstance(payload, list):  # pragma: no cover
            raise DataError(f"Tiingo returned a non-list payload for {symbol!r}: {payload!r}.")
        return payload

    def available_symbols(self) -> Sequence[str]:
        """List symbols available in the fixture cache (offline transport only)."""
        if self._fixture_root is None:
            raise DataError("available_symbols is only defined for the fixture transport.")
        return sorted(p.stem for p in self._fixture_root.glob("*.json"))


def tiingo_data_source(
    api_key: str | None = None,
    *,
    fixture_root: Path | str | None = None,
) -> TiingoDataSource:
    """Registry-friendly factory (BUILD_PLAN §5.2) for :class:`TiingoDataSource`.

    With neither ``api_key`` nor ``fixture_root`` it falls back to the committed
    offline fixture cache, so ``tiingo_data_source()`` always returns a working
    (offline) ``DataSource`` in this environment.
    """
    if api_key is None and fixture_root is None:
        return TiingoDataSource.from_fixtures()
    if api_key is not None:
        return TiingoDataSource.live(api_key)
    return TiingoDataSource.from_fixtures(fixture_root)


__all__ = [
    "TiingoDataSource",
    "tiingo_data_source",
]
