"""Real market-data provider adapters (BUILD_PLAN §8, Agent 1).

The always-offline backbone of the system is :mod:`riskbudget.data.synthetic`
(plus the :mod:`riskbudget.data.csvsource` loader, both owned by Agent 2). This
sub-package adds *optional, network-gated* adapters to real data providers for a
future real-money path.

Per the data-source evaluation in :doc:`docs/data-sources.md`, the recommended
first real adapter is **Tiingo** (:class:`TiingoDataSource`): a clean REST/JSON
API with split- and dividend-adjusted closes, a generous free tier, and an
explicit API key (no scraping of an undocumented endpoint, unlike yfinance).

For the **deep-history backbone** (:doc:`docs/instrument-universe.md` §6) two
free, GitHub-mirrored monthly providers are added:

- :class:`ShillerDataSource` — the Robert Shiller S&P 500 dataset (1871-present)
  exposing ``SP500_NOMINAL`` / ``SP500_REAL`` / ``SP500_TR`` (dividend total
  return) and ``US10Y_TR`` (a documented duration-8 10Y Treasury TR *proxy*).
- :class:`GoldDataSource` — the datahub monthly gold spot series (1833-present),
  asset ``GOLD``.

Both mirror Tiingo's two-transport design (``from_fixtures()`` offline /
``from_github()`` live), ship committed trimmed CSV fixtures so CI is fully
offline, and import their network transport lazily.

Network reality in *this* build environment
-------------------------------------------
Outbound network here is an allowlist that resolves only PyPI, PythonHosted and
GitHub; every market-data host (Tiingo, Yahoo/yfinance, Stooq, Alpha Vantage,
FRED, Polygon) returns ``Host not in allowlist``. This was confirmed empirically
(see ``docs/data-sources.md`` §"Network probe"). The adapter is therefore built
and tested against a small *recorded* fixture committed under
``providers/fixtures/tiingo/`` and exercises the exact JSON shape Tiingo returns
live. :meth:`TiingoDataSource.from_fixtures` wires it to that offline cache;
:meth:`TiingoDataSource.live` (or the default constructor with an API key) wires
it to the real HTTP endpoint when a network path exists.

Both classes satisfy the :class:`~riskbudget.core.interfaces.DataSource`
protocol (``get_prices(assets, start, end) -> PriceData``) and are exposed as a
registry-friendly factory callable (``tiingo_data_source``) per BUILD_PLAN §5.2.
"""

from __future__ import annotations

from riskbudget.data.providers.gold import (
    GoldDataSource,
    gold_data_source,
)
from riskbudget.data.providers.shiller import (
    ShillerDataSource,
    shiller_data_source,
)
from riskbudget.data.providers.tiingo import (
    TiingoDataSource,
    tiingo_data_source,
)

__all__ = [
    "GoldDataSource",
    "ShillerDataSource",
    "TiingoDataSource",
    "gold_data_source",
    "shiller_data_source",
    "tiingo_data_source",
]
