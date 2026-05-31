"""Real market-data provider adapters (BUILD_PLAN §8, Agent 1).

The always-offline backbone of the system is :mod:`riskbudget.data.synthetic`
(plus the :mod:`riskbudget.data.csvsource` loader, both owned by Agent 2). This
sub-package adds *optional, network-gated* adapters to real data providers for a
future real-money path.

Per the data-source evaluation in :doc:`docs/data-sources.md`, the recommended
first real adapter is **Tiingo** (:class:`TiingoDataSource`): a clean REST/JSON
API with split- and dividend-adjusted closes, a generous free tier, and an
explicit API key (no scraping of an undocumented endpoint, unlike yfinance).

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

from riskbudget.data.providers.tiingo import (
    TiingoDataSource,
    tiingo_data_source,
)

__all__ = [
    "TiingoDataSource",
    "tiingo_data_source",
]
