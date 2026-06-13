"""Loadable default universe helper over the curated instrument catalog.

``docs/instrument_universe.csv`` is a hand-curated catalog of US-traded
instruments spliced for maximum price history (equities, ETFs, futures, rates,
commodities, volatility). It is *metadata* — symbols + provenance, not prices —
so this helper only exposes the symbol list (and the raw catalog rows) as a
convenience default universe; actual prices still come from a
:class:`~riskbudget.core.interfaces.DataSource`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from riskbudget.core.errors import DataError

# docs/ sits next to the riskbudget/ package at the repo root.
_UNIVERSE_CSV = Path(__file__).resolve().parent.parent / "docs" / "instrument_universe.csv"


def load_universe(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load the curated instrument catalog as a list of row dicts.

    Parameters
    ----------
    path:
        Optional override of the catalog CSV; defaults to
        ``docs/instrument_universe.csv``.

    Raises
    ------
    DataError
        If the catalog file is missing or cannot be parsed.
    """
    csv_path = Path(path) if path is not None else _UNIVERSE_CSV
    if not csv_path.exists():
        raise DataError(f"Instrument-universe catalog not found at {csv_path}.")
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DataError(f"Could not read instrument-universe catalog: {exc}") from exc
    if not rows:
        raise DataError("Instrument-universe catalog is empty.")
    return rows


def default_universe(
    *,
    asset_class: str | None = None,
    limit: int | None = None,
    path: str | Path | None = None,
) -> list[str]:
    """Return the curated default universe as a list of symbols.

    Parameters
    ----------
    asset_class:
        Optional filter on the catalog's ``asset_class`` column (e.g.
        ``"equity_universe"``, ``"etf"``, ``"rates"``).
    limit:
        Optional cap on the number of symbols returned (in catalog order).
    path:
        Optional override of the catalog CSV.

    Raises
    ------
    DataError
        If the catalog is missing, has no ``symbol`` column, or the filter yields
        no symbols.
    """
    rows = load_universe(path)
    if "symbol" not in rows[0]:
        raise DataError("Instrument-universe catalog has no 'symbol' column.")
    symbols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if asset_class is not None and row.get("asset_class") != asset_class:
            continue
        sym = (row.get("symbol") or "").strip()
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    if not symbols:
        raise DataError(
            "No symbols in the instrument-universe catalog"
            + (f" for asset_class={asset_class!r}." if asset_class else ".")
        )
    return symbols[:limit] if limit is not None else symbols


__all__ = ["default_universe", "load_universe"]
