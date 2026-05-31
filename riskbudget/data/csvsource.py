"""CSV / parquet price-panel loader (BUILD_PLAN §4, Agent 2).

:class:`CsvDataSource` loads a price panel from a CSV or parquet file laid out as
``index = dates`` (first column) and ``columns = asset ids``, validates it, and
serves it through the :class:`~riskbudget.core.interfaces.DataSource` protocol
(``get_prices(assets, start, end) -> PriceData``).

Missing-data policy (documented and configurable)
-------------------------------------------------
Real price panels have gaps: a non-trading day for one asset, a late IPO, a
delisting. The loader's default policy is:

1. **Sort** rows by date and coerce the index to datetimes.
2. **Forward-fill** (``ffill``) each asset's prices: a gap inherits the last
   observed price, which models "no new information / hold last close" — the
   standard convention for daily price series. This never looks ahead.
3. **Back-fill the leading gap** is *not* done by default (it would invent a
   price before the asset existed and leak the future). Instead, leading NaNs
   (an asset with no observation yet) remain NaN and are handled by the
   ``on_missing`` policy below.
4. After the fill, any column that is still entirely/partly NaN within the
   requested window is handled per ``on_missing``: ``"error"`` (default) raises
   :class:`~riskbudget.core.errors.DataError`; ``"drop"`` drops offending assets;
   the strict :class:`PriceData` constructor enforces positivity and finiteness
   on whatever survives.

The forward-fill is applied on the *full* loaded panel before date slicing, so a
gap straddling the window boundary is filled from the correct prior observation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData

__all__ = [
    "CsvDataSource",
    "csv_data_source",
]

OnMissing = Literal["error", "drop"]

_PARQUET_SUFFIXES = {".parquet", ".pq", ".parq"}
_CSV_SUFFIXES = {".csv", ".txt"}


class CsvDataSource:
    """A :class:`DataSource` that reads a price panel from CSV or parquet.

    Parameters
    ----------
    path:
        File path. ``.csv`` / ``.txt`` are read as CSV; ``.parquet`` / ``.pq`` /
        ``.parq`` as parquet. The format can be forced via ``fmt``.
    fmt:
        ``"csv"`` or ``"parquet"`` to override extension-based detection;
        ``None`` (default) infers from the suffix.
    date_column:
        Name of the date column for CSV input. ``None`` (default) uses the first
        column as the index. Ignored for parquet (its index is used directly, or
        the first column if the index is a default RangeIndex).
    forward_fill:
        Whether to forward-fill gaps (default ``True``). See module docstring.
    on_missing:
        Policy for assets still carrying NaNs after the fill within a requested
        window: ``"error"`` (default) or ``"drop"``.

    Raises
    ------
    DataError
        If the file is missing, unreadable, has an unparseable date index, or
        contains no usable price columns.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fmt: Literal["csv", "parquet"] | None = None,
        date_column: str | None = None,
        forward_fill: bool = True,
        on_missing: OnMissing = "error",
    ) -> None:
        self._path = Path(path)
        if on_missing not in ("error", "drop"):
            raise DataError(f"on_missing must be 'error' or 'drop', got {on_missing!r}.")
        self._forward_fill = forward_fill
        self._on_missing = on_missing
        self._fmt = self._resolve_fmt(self._path, fmt)
        self._date_column = date_column
        self._frame = self._load()

    @staticmethod
    def _resolve_fmt(
        path: Path, fmt: Literal["csv", "parquet"] | None
    ) -> Literal["csv", "parquet"]:
        if fmt is not None:
            if fmt not in ("csv", "parquet"):
                raise DataError(f"fmt must be 'csv' or 'parquet', got {fmt!r}.")
            return fmt
        suffix = path.suffix.lower()
        if suffix in _PARQUET_SUFFIXES:
            return "parquet"
        if suffix in _CSV_SUFFIXES:
            return "csv"
        raise DataError(
            f"Cannot infer format from suffix {suffix!r}; pass fmt='csv' or fmt='parquet'."
        )

    def _load(self) -> pd.DataFrame:
        """Read, normalize, validate, and (optionally) forward-fill the panel."""
        if not self._path.exists():
            raise DataError(f"Price file not found: {self._path}.")

        try:
            if self._fmt == "parquet":
                raw = pd.read_parquet(self._path)
                raw = self._index_from_frame(raw, self._date_column)
            else:
                index_col = self._date_column if self._date_column is not None else 0
                raw = pd.read_csv(self._path, index_col=index_col)
        except DataError:
            raise
        except Exception as exc:
            raise DataError(f"Failed to read {self._path}: {exc}.") from exc

        if raw.shape[1] == 0:
            raise DataError(f"No asset columns found in {self._path}.")

        # Coerce the index to datetimes.
        try:
            raw.index = pd.to_datetime(raw.index)
        except (TypeError, ValueError) as exc:
            raise DataError(f"Could not parse the date index of {self._path}: {exc}.") from exc

        raw.columns = [str(c) for c in raw.columns]
        if pd.Index(raw.columns).has_duplicates:
            raise DataError(f"Duplicate asset columns in {self._path}.")
        if raw.index.has_duplicates:
            raise DataError(f"Duplicate dates in the index of {self._path}.")

        # Coerce values to numeric (strings -> NaN -> caught by policy / validator).
        frame: pd.DataFrame = raw.apply(pd.to_numeric, errors="coerce")
        frame = frame.sort_index()

        if self._forward_fill:
            frame = frame.ffill()
        return frame

    @staticmethod
    def _index_from_frame(frame: pd.DataFrame, date_column: str | None) -> pd.DataFrame:
        """Pick the date index for a parquet frame.

        Uses ``date_column`` if given; else an existing non-default index; else
        the first column.
        """
        if date_column is not None:
            if date_column not in frame.columns:
                raise DataError(f"date_column {date_column!r} not present in parquet file.")
            return frame.set_index(date_column)
        if not isinstance(frame.index, pd.RangeIndex):
            return frame
        if frame.shape[1] < 2:
            raise DataError(
                "Parquet file has a default index and a single column; cannot infer dates."
            )
        return frame.set_index(frame.columns[0])

    @property
    def assets(self) -> list[str]:
        """All asset ids available in the loaded file."""
        return [str(c) for c in self._frame.columns]

    @property
    def dates(self) -> pd.DatetimeIndex:
        """The (ascending) date index of the loaded file."""
        idx = self._frame.index
        return idx if isinstance(idx, pd.DatetimeIndex) else pd.DatetimeIndex(idx)

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a validated :class:`PriceData` panel for ``assets`` in ``[start, end]``.

        Parameters
        ----------
        assets:
            Asset ids to select. Every id must exist in the file.
        start, end:
            Inclusive date bounds.

        Raises
        ------
        DataError
            If ``assets`` is empty/unknown, ``start > end``, the window is empty,
            or assets still carry missing/non-positive prices under the
            ``on_missing="error"`` policy.
        """
        if not assets:
            raise DataError("get_prices requires a non-empty list of assets.")
        requested = [str(a) for a in assets]
        available = set(self._frame.columns)
        unknown = [a for a in requested if a not in available]
        if unknown:
            raise DataError(
                f"Unknown assets in {self._path}: {unknown}. Available: {sorted(available)}."
            )
        if len(set(requested)) != len(requested):
            raise DataError("Duplicate assets requested.")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts > end_ts:
            raise DataError(f"start ({start_ts.date()}) must not be after end ({end_ts.date()}).")

        window = self._frame.loc[(self._frame.index >= start_ts) & (self._frame.index <= end_ts)]
        sub = window.loc[:, requested]
        if sub.shape[0] == 0:
            raise DataError(
                f"No rows in the window [{start_ts.date()}, {end_ts.date()}] for {self._path}."
            )

        # Apply the missing-data policy on the selected window.
        missing_cols = [c for c in sub.columns if sub[c].isna().any()]
        if missing_cols:
            if self._on_missing == "drop":
                sub = sub.drop(columns=missing_cols)
                if sub.shape[1] == 0:
                    raise DataError(
                        "All requested assets had missing prices in the window and were dropped."
                    )
            else:
                raise DataError(
                    f"Assets with missing prices in [{start_ts.date()}, {end_ts.date()}]: "
                    f"{missing_cols}. Use on_missing='drop' to skip them or widen the window "
                    "so a prior observation exists to forward-fill from."
                )

        try:
            return PriceData(sub)
        except Exception as exc:
            raise DataError(f"Loaded panel failed validation: {exc}.") from exc


def csv_data_source(
    path: str | Path,
    *,
    fmt: Literal["csv", "parquet"] | None = None,
    date_column: str | None = None,
    forward_fill: bool = True,
    on_missing: OnMissing = "error",
) -> CsvDataSource:
    """Registry-friendly factory for :class:`CsvDataSource` (BUILD_PLAN §5.2)."""
    return CsvDataSource(
        path,
        fmt=fmt,
        date_column=date_column,
        forward_fill=forward_fill,
        on_missing=on_missing,
    )
