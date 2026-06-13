"""Long-span **daily** multi-asset deep-history adapter (BUILD_PLAN §8).

Where :mod:`riskbudget.data.providers.shiller` / :mod:`riskbudget.data.providers.gold`
serve *monthly* deep history, this adapter serves the longest **daily** multi-asset
panel reachable from the GitHub-only allowlist of this build environment. It mirrors
the same two-transport pattern as the Tiingo / Shiller / gold providers:

- :meth:`DailyPanelDataSource.from_github` — **live and canonical**, downloads
  five daily source CSVs from GitHub raw mirrors (``requests`` imported lazily,
  falling back to ``urllib``) and re-assembles the panel via
  :func:`assemble_panel`. This is the **current path**: it fetches fresh on every
  call so the panel stays up to date with the upstream mirrors. Prefer this for
  real usage.
- :meth:`DailyPanelDataSource.from_fixtures` — **offline**, reads a committed,
  pre-assembled daily panel CSV under ``examples/data/daily_panel_long.csv``
  (``Date`` index + ``STOCKS`` / ``STOCKS_TR`` / ``BONDS`` / ``GOLD`` columns).
  The fixture is an **offline CI cache** of the live result, periodically
  refreshed/regenerated from the live sources; it is what the test-suite and the
  ``--offline`` example path use, and it needs no network.

Both transports converge on the same in-memory frame; the offline tests therefore
exercise the real alignment/derivation logic, only the *transport* differs.

Sources (all verified fetchable via ``raw.githubusercontent.com`` in this sandbox)
---------------------------------------------------------------------------------
- ``STOCKS`` — S&P 500 / SPX **daily close**, **spliced** from two mirrors so the
  series is both *deep* and *current*:

  * the **deep** leg is the Stooq ``^spx`` reconstruction mirrored at
    ``ai357060/flower`` (``Data/spx_d.csv``), true daily from **1885-01-01** but
    **static**, ending **2024-02-26**. Pre-1885 rows are monthly-spaced backfill
    and are dropped.
  * the **current** leg is the ``SPX Index`` column of ``Indices.csv`` in the same
    ``juanfp02/commodities_and_sovereigns`` repo that serves the bond yield. It
    spans **2000-11-20 -> present** (last dump 2025-12-18) and the maintainer
    refreshes the whole repo periodically alongside ``DGS10.csv``.

  The two legs are spliced (:func:`_splice_stocks`) at the deep leg's last day:
  the deep prices are kept verbatim **in full**, and the current leg is used only
  where it *extends* the deep leg (dates after its end), multiplied by
  ``deep[seam] / current[seam]`` (seam = last common date) so the level is
  continuous; if the current leg ever regresses to end *before* the deep leg, the
  deep tail is preserved rather than truncated. The splice sanity-checks the seam
  (scale within ±5%, last-20-common-days agreement within 2% mean abs) and raises
  ``DataError`` on upstream format/content changes. Over their 5,851-day overlap
  (2000-11-20 -> 2024-02-26) the two SPX series agree to a mean relative
  difference of ~0.0002% (100% of days within 0.5%; both are the same SPX, only
  timestamp/rounding differs), so the splice scale factor is ~1.0016 and the seam
  is invisible.
- ``STOCKS_TR`` — a **dividend-adjusted total-return** index for the S&P price
  leg. The annualized monthly dividend yield ``y = Dividend / SP500`` (from the
  Shiller datahub mirror) is lagged one month (so days in month ``M`` carry
  month ``M-1``'s knowable yield — no intra-month look-ahead; see
  :func:`_stocks_tr_index`), forward-filled onto the daily index and spread
  across the trading year (``daily_div = y / 252``); the daily total return is
  the price change plus that carry, ``r_t = STOCKS.pct_change() + daily_div``,
  cumulated to a level series rebased to 100 on the first aligned date.
- ``BONDS`` — a 10-year Treasury **total-return PROXY** built from the daily
  ``DGS10`` constant-maturity yield (FRED series, daily from **1962-01-02**,
  mirrored at ``juanfp02/commodities_and_sovereigns``: ``data/DGS10.csv``). The
  daily return is the carry minus a duration-scaled price move::

      r_t = y_{prev}/252 - ModDur * (y_t - y_{prev})      (ModDur = 8.0)

  cumulated to a growth index based at 1.0. This is the **same** constant-duration
  (~8y) proxy methodology the Shiller provider documents — it ignores convexity,
  roll-down and the exact bond cash-flow schedule. Use it for illustrative
  multi-asset risk-budgeting demos, not production bond analytics.
- ``GOLD`` — daily LBMA Gold Price PM fix (USD/oz), daily from **1968-04-01**,
  **current and auto-updating**: the ``unbalancedparentheses/forex-centuries``
  repo refreshes ``data/sources/lbma/lbma_gold_daily.csv`` via a weekly GitHub
  Actions cron (Mondays 06:00 UTC), so the live transport stays fresh through the
  present. The file is ``date,gold_pm_usd,gold_pm_gbp,gold_pm_eur``; only the USD
  column is used (located by the ``gold_pm_usd`` header name via
  :func:`_parse_two_col`, falling back to column 2 when the header is absent, so
  an upstream column re-order cannot swap currencies). This
  replaces the previously-used static ``UtaHagen/PortfolioProject`` mirror, which
  ended 2023-12-28; over their 14k-day overlap the two series agree to a mean
  relative difference of ~0.4% (AM/PM-fix and spot/fix timing account for the few
  isolated larger gaps), so the swap preserves the historical level.

The aligned (inner-join) daily panel spans **1968-04-01 -> ~2025-12** (~14.5k
rows); the start date is bounded by the gold series (1968-04-01) and the end date
by whichever of the now-current legs (STOCKS, BONDS, GOLD, Shiller dividends)
stops first — after this splice the binding end leg is the bond yield (DGS10,
2025-12-16), not the previously-stale STOCKS mirror.

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

# Live GitHub raw mirrors of the daily source series.
# STOCKS is spliced from a deep-but-static leg and a shallow-but-current leg
# (see the module docstring + :func:`_splice_stocks`).
_STOCKS_DEEP_URL = "https://raw.githubusercontent.com/ai357060/flower/master/Data/spx_d.csv"
_STOCKS_CURRENT_URL = (
    "https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/Indices.csv"
)
# Back-compat alias: the deep Stooq leg remains the canonical "stocks" URL name.
_STOCKS_URL = _STOCKS_DEEP_URL
_DGS10_URL = (
    "https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/DGS10.csv"
)
_GOLD_URL = (
    "https://raw.githubusercontent.com/unbalancedparentheses/forex-centuries/main/"
    "data/sources/lbma/lbma_gold_daily.csv"
)
# Shiller datahub mirror (monthly ``Date,SP500,Dividend,...``) used to derive the
# dividend yield for STOCKS_TR. The ``Dividend`` column is the annualized dividend
# per index point; the monthly yield is ``Dividend / SP500``.
_SHILLER_DIV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

# Modified duration used by the BONDS total-return proxy (years). Documented above.
_BOND_MOD_DUR = 8.0

# True-daily granularity in the SPX reconstruction begins here; earlier rows in the
# source file are monthly-spaced backfill and are dropped before assembly.
_STOCKS_DAILY_START = pd.Timestamp("1885-01-01")

# The assets this source serves (panel column order).
_ASSETS = ("STOCKS", "STOCKS_TR", "BONDS", "GOLD")

# Live-transport URL tuple:
# ``(stocks_deep, dgs10, gold, shiller, stocks_current)``.
_UrlTuple = tuple[str, str, str, str, str]


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


def _parse_two_col(text: str, value_name: str, *, prefer_columns: Sequence[str] = ()) -> pd.Series:
    """Parse a generic two-column ``date,value`` daily CSV into a Series.

    The first column is always treated as the date; the *second* column is the
    value (so this tolerates the different header names the three mirrors use,
    e.g. ``Close``/``DGS10``/``Gold Price``). Rows whose value is blank or the
    FRED ``.`` placeholder are skipped.

    ``prefer_columns`` lets a caller pin the value column by **header name**
    instead of trusting column position: the first listed name found in the
    header row wins (case-insensitive, whitespace-stripped). When none match,
    the generic second-column behavior applies, so existing callers are
    unaffected. The gold payload passes ``("gold_pm_usd",)`` so an upstream
    column re-order cannot silently swap USD for GBP/EUR.
    """
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    if not rows:
        raise DataError(f"{value_name} CSV is empty (no header row).")
    header = [cell.strip().lower() for cell in rows[0]]
    value_col = 1
    for name in prefer_columns:
        if name.strip().lower() in header:
            value_col = header.index(name.strip().lower())
            break
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows[1:]:
        if len(row) <= value_col or not row[0]:
            continue
        raw = row[value_col].strip()
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


def _parse_flexible_decimal(raw: str) -> float:
    """Parse a numeric string whose decimal convention is detected per value.

    - Contains **both** ``.`` and ``,`` → European convention (``.`` thousands
      separator, ``,`` decimal comma), e.g. ``"6.800,26"`` → ``6800.26``.
    - Contains **only** ``,`` → decimal comma, e.g. ``"6800,26"`` → ``6800.26``.
    - Contains **only** ``.`` (or neither) → plain US decimal, e.g. ``"6800.26"``.

    This avoids blindly stripping ``.`` (which would corrupt a US-format dump if
    the upstream file's convention changes). Raises ``ValueError`` on
    unparseable input, like :func:`float`.
    """
    if "," in raw and "." in raw:
        return float(raw.replace(".", "").replace(",", "."))
    if "," in raw:
        return float(raw.replace(",", "."))
    return float(raw)


def _indices_spx_close(text: str) -> pd.Series:
    """Extract the daily SPX close from ``juanfp02/.../data/Indices.csv``.

    The file is **semicolon-delimited** with a ``Dates;EMB US Equity;SPX Index``
    header. Dates are ``DD.MM.YYYY``; numbers currently use the European
    convention (``,`` decimal comma, ``.`` thousands separator) but the decimal
    convention is detected **per value** via :func:`_parse_flexible_decimal` so
    an upstream switch to US formatting is parsed correctly rather than
    silently corrupted. Bloomberg ``#N/A N/A`` / blank cells are skipped. The
    SPX column is located by header name so a column re-order upstream is
    tolerated.
    """
    reader = csv.reader(StringIO(text), delimiter=";")
    rows = list(reader)
    if not rows:
        raise DataError("SPX Index CSV is empty (no header row).")
    header = rows[0]
    try:
        spx_col = header.index("SPX Index")
    except ValueError as exc:
        raise DataError(f"SPX Index column not found in Indices CSV header: {header!r}.") from exc
    dates: list[pd.Timestamp] = []
    closes: list[float] = []
    for row in rows[1:]:
        if len(row) <= spx_col or not row[0]:
            continue
        raw = row[spx_col].strip()
        if raw in ("", ".", "#N/A N/A", "#N/A", "NaN"):
            continue
        try:
            ts = pd.Timestamp(pd.to_datetime(row[0], format="%d.%m.%Y")).normalize()
            close = _parse_flexible_decimal(raw)
        except (TypeError, ValueError):
            continue
        if close <= 0.0:
            continue
        dates.append(ts)
        closes.append(close)
    if not dates:
        raise DataError("SPX Index CSV contained no usable rows.")
    series = pd.Series(closes, index=pd.DatetimeIndex(dates), name="STOCKS")
    return series[~series.index.duplicated(keep="last")].sort_index()


# Sanity limits for the STOCKS splice (both legs are the same SPX index, so the
# seam scale should be ~1.0 and the overlap window should agree tightly; live
# observed values are scale ~1.0016 with ~0.0002% mean relative difference).
_SPLICE_SCALE_BOUNDS = (0.95, 1.05)
_SPLICE_OVERLAP_WINDOW = 20  # last N common days checked for agreement
_SPLICE_OVERLAP_TOL = 0.02  # max mean absolute relative difference


def _splice_stocks(deep: pd.Series, current: pd.Series) -> pd.Series:
    """Splice a deep-but-static SPX leg onto a shallow-but-current one.

    The deep prices are kept **verbatim in full** (including any dates past the
    last common date); the current leg is used only where it *extends* the deep
    leg — dates strictly after the deep leg's end — multiplied by
    ``deep[seam] / current[seam]`` (seam = the last common date) so the joined
    level is continuous. In particular, if the current leg ends *before* the
    deep leg (an upstream regression), the deep leg's tail is preserved rather
    than truncated. If the two legs do not overlap at all, the current leg is
    rebased onto the deep leg's final level instead (best effort, no seam
    sanity check possible).

    Sanity checks (live-path protection against upstream format/content
    changes): when the legs overlap, a :class:`DataError` is raised unless the
    seam scale lies within :data:`_SPLICE_SCALE_BOUNDS` **and** the last
    :data:`_SPLICE_OVERLAP_WINDOW` common days agree (after scaling) to within
    :data:`_SPLICE_OVERLAP_TOL` mean absolute relative difference — both legs
    are the same SPX, so any larger disagreement means a corrupted leg.

    The result is a strictly-positive daily price level, ascending and de-duped,
    spanning ``min(deep.start, current.start)`` to ``max(deep.end, current.end)``.
    """
    if deep.empty:
        out = current.copy()
        out.name = "STOCKS"
        return out
    if current.empty:
        out = deep.copy()
        out.name = "STOCKS"
        return out
    overlap = deep.index.intersection(current.index)
    if len(overlap) > 0:
        seam = overlap.max()
        scale = float(deep.loc[seam]) / float(current.loc[seam])
        lo, hi = _SPLICE_SCALE_BOUNDS
        if not (lo <= scale <= hi):
            raise DataError(
                f"STOCKS splice seam scale {scale:.4f} outside [{lo}, {hi}]: the deep "
                f"({float(deep.loc[seam]):.2f}) and current ({float(current.loc[seam]):.2f}) "
                f"SPX legs disagree at the seam ({seam.date()}) — likely an upstream "
                "format or content change."
            )
        window = overlap.sort_values()[-_SPLICE_OVERLAP_WINDOW:]
        rel = (deep.loc[window] / (current.loc[window] * scale) - 1.0).abs()
        mean_abs = float(rel.mean())
        if mean_abs > _SPLICE_OVERLAP_TOL:
            raise DataError(
                f"STOCKS splice overlap check failed: the last {len(window)} common days "
                f"of the deep and current SPX legs differ by {mean_abs:.2%} mean absolute "
                f"relative difference (tolerance {_SPLICE_OVERLAP_TOL:.0%}) — likely an "
                "upstream format or content change."
            )
    else:
        scale = float(deep.iloc[-1]) / float(current.iloc[0])
    # Keep the deep leg in full; use the current leg only where it EXTENDS it.
    deep_end = deep.index.max()
    tail = current[current.index > deep_end] * scale
    spliced: pd.Series = pd.concat([deep, tail])
    spliced = spliced[~spliced.index.duplicated(keep="last")].sort_index()
    spliced.name = "STOCKS"
    return spliced


def _shiller_div_yield(text: str) -> pd.Series:
    """Parse the monthly dividend yield ``y = Dividend / SP500`` from Shiller data.

    The datahub Shiller CSV carries a monthly ``Dividend`` column (annualized
    dividend per index point) alongside the ``SP500`` price level. The yield is
    their ratio; rows with a missing/zero price are skipped (a zero ``Dividend``
    is a legitimate yield of 0 and is kept).
    """
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise DataError("Shiller dividend CSV is empty (no header row).")
    required = {"Date", "SP500", "Dividend"}
    missing = required.difference(reader.fieldnames)
    if missing:
        raise DataError(
            f"Shiller dividend CSV is missing required columns: {sorted(missing)}. "
            f"Found: {reader.fieldnames!r}."
        )
    dates: list[pd.Timestamp] = []
    yields: list[float] = []
    for row in reader:
        raw_date = row.get("Date")
        if not raw_date:
            continue
        raw_price = (row.get("SP500") or "").strip()
        raw_div = (row.get("Dividend") or "").strip()
        if raw_price in ("", ".") or raw_div in ("", "."):
            continue
        try:
            ts = pd.Timestamp(raw_date).normalize()
            price = float(raw_price)
            div = float(raw_div)
        except (TypeError, ValueError):
            continue
        if price <= 0.0:
            continue
        dates.append(ts)
        yields.append(div / price)
    if not dates:
        raise DataError("Shiller dividend CSV contained no usable rows.")
    series = pd.Series(yields, index=pd.DatetimeIndex(dates), name="DIV_YIELD")
    return series[~series.index.duplicated(keep="last")].sort_index()


def _stocks_tr_index(stocks: pd.Series, div_yield: pd.Series) -> pd.Series:
    """Build the daily dividend-adjusted total-return index for the S&P price leg.

    The annualized monthly yield is forward-filled onto the (already date-aligned)
    daily ``stocks`` index and spread across the trading year, then added to the
    daily price change and cumulated. See the module docstring for the formula.
    The result is rebased to 100 on the first aligned date.

    Look-ahead caveat
    -----------------
    Applying the Shiller month-``M`` yield to days *within* month ``M`` would
    embed a mild **intra-month look-ahead** (the month's dividend/price are not
    knowable until the month ends). To stay strictly point-in-time, the yield
    series is lagged by one observation (one month, since it is
    monthly-stamped) before the forward-fill, so days in month ``M`` carry
    month ``M-1``'s yield. For a smooth ``y/252`` carry spread this changes the
    cumulative total return only at the basis-point level (month-over-month
    yield changes are tiny and largely telescope out), so STOCKS_TR is
    economically unchanged.
    """
    # shift(1) on the monthly-stamped series shifts by one OBSERVATION = one
    # month: month-M's stamp now carries month M-1's (knowable) yield.
    daily_div = div_yield.shift(1).reindex(stocks.index, method="ffill") / 252.0
    # Fill each component separately: a missing dividend yield (dates before the
    # first Shiller observation) must not zero out the PRICE return for that day.
    r_t = stocks.pct_change().fillna(0.0) + daily_div.fillna(0.0)
    tr = (1.0 + r_t).cumprod()
    tr = tr / float(tr.iloc[0]) * 100.0
    return tr.rename("STOCKS_TR")


def _bond_tr_from_yield(yield_pct: pd.Series) -> pd.Series:
    """Build the constant-duration 10Y total-return proxy from a daily yield (%).

    See the module docstring for the formula; ``ModDur`` is :data:`_BOND_MOD_DUR`.
    """
    y = yield_pct / 100.0
    daily_ret = (y.shift(1) / 252.0 - _BOND_MOD_DUR * (y - y.shift(1))).fillna(0.0)
    return (1.0 + daily_ret).cumprod().rename("BONDS")


def assemble_panel(
    stocks_text: str,
    dgs10_text: str,
    gold_text: str,
    shiller_text: str,
    stocks_current_text: str | None = None,
) -> pd.DataFrame:
    """Assemble the aligned daily panel from the raw source payloads.

    This is the shared derivation used by the **live** transport; the offline
    transport reads the already-assembled result. ``STOCKS_TR`` is the
    dividend-adjusted total-return index for the S&P price leg (derived from the
    Shiller dividend yield in ``shiller_text``), rebased to 100 on the first row
    of the aligned (inner-join) panel; the ``BONDS`` proxy is rebased to 1.0 on
    that same first row.

    ``stocks_text`` is the **deep** static Stooq SPX leg. When
    ``stocks_current_text`` (the ``Indices.csv`` ``SPX Index`` column) is given it
    is **spliced** onto the deep leg via :func:`_splice_stocks` so the STOCKS leg
    runs to the present; when ``None`` only the deep leg is used (back-compat).

    Raises
    ------
    DataError
        If any source parses empty or the inner join leaves no overlapping dates.
    """
    stocks = _stooq_spx_close(stocks_text)
    if stocks_current_text is not None:
        stocks = _splice_stocks(stocks, _indices_spx_close(stocks_current_text))
    bonds = _bond_tr_from_yield(_parse_two_col(dgs10_text, "DGS10"))
    # Pin the LBMA USD column by header name (falls back to column 1 when the
    # header is absent) so an upstream column re-order cannot swap currencies.
    gold = _parse_two_col(gold_text, "GOLD", prefer_columns=("gold_pm_usd",))

    panel = pd.concat([stocks, bonds, gold], axis=1, join="inner").dropna(how="any")
    if panel.empty:
        raise DataError("Daily panel is empty after aligning STOCKS/BONDS/GOLD.")
    panel.columns = ["STOCKS", "BONDS", "GOLD"]

    # Derive STOCKS_TR on the aligned dates so it is rebased to the panel start.
    div_yield = _shiller_div_yield(shiller_text)
    panel["STOCKS_TR"] = _stocks_tr_index(panel["STOCKS"], div_yield)

    panel = panel.loc[:, list(_ASSETS)].copy()
    panel["BONDS"] = panel["BONDS"] / panel["BONDS"].iloc[0]
    panel.index.name = "Date"
    return panel


class DailyPanelDataSource:
    """A :class:`~riskbudget.core.interfaces.DataSource` over the long daily panel.

    Two transports, one logical panel:

    - **fixture** (offline): reads the committed pre-assembled CSV.
    - **github** (network): downloads the daily sources and re-assembles.

    Parameters
    ----------
    fixture_path:
        Path to a committed panel CSV. When set, the adapter reads from disk and
        never touches the network.
    urls:
        ``(stocks_deep, dgs10, gold, shiller, stocks_current)`` live CSV URLs
        (github transport). Ignored when ``fixture_path`` is set.
    timeout:
        Per-request timeout in seconds for the live transport.
    """

    def __init__(
        self,
        *,
        fixture_path: Path | str | None = None,
        urls: _UrlTuple | None = None,
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
        urls: _UrlTuple | None = None,
        timeout: float = 30.0,
    ) -> DailyPanelDataSource:
        """Build a **live** adapter that downloads + re-assembles from GitHub.

        This is the canonical/current path: it fetches fresh on every
        :meth:`get_prices` call so the panel tracks the upstream mirrors.
        """
        return cls(
            urls=urls
            or (
                _STOCKS_DEEP_URL,
                _DGS10_URL,
                _GOLD_URL,
                _SHILLER_DIV_URL,
                _STOCKS_CURRENT_URL,
            ),
            timeout=timeout,
        )

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
        """Download the sources and re-assemble. Network path only."""
        assert self._urls is not None  # narrowed by caller
        (
            stocks_text,
            dgs10_text,
            gold_text,
            shiller_text,
            stocks_current_text,
        ) = (self._download(u) for u in self._urls)
        return assemble_panel(
            stocks_text,
            dgs10_text,
            gold_text,
            shiller_text,
            stocks_current_text,
        )

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
    urls: _UrlTuple | None = None,
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
