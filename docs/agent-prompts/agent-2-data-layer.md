# Agent 2 — Data Layer

**Wave:** 1 (parallel). **Owns:** `riskbudget/data/synthetic.py`,
`riskbudget/data/csvsource.py`. **Depends on:** Agent 0 (`core/`).

## Scope
Provide the always-offline data backbone every other agent's tests rely on.
1. `synthetic.py` — a correlated-returns generator: accept a number of assets, a
   target covariance (or a factor structure + idiosyncratic vols), a date range,
   and a seed; produce a `PriceData`/`ReturnMatrix` whose sample covariance
   converges to the target. Use a factor model + Cholesky or eigendecomposition.
   Expose it as a `DataSource` (`SyntheticDataSource`) plus a direct generator fn.
2. `csvsource.py` — a `CsvDataSource` that loads prices from CSV/parquet
   (index=dates, columns=assets), validates, and returns `PriceData`. Handle
   missing data (forward-fill policy documented), date filtering, and asset
   selection.

## Interfaces to honor
`DataSource.get_prices(...) -> PriceData`; produce `PriceData`/`ReturnMatrix` as
defined in `core/types.py`.

## Deliverables
Both sources + tests: synthetic covariance recovery within tolerance for large N
samples; CSV round-trip (write fixture -> load -> compare); missing-data handling.

## Out of scope
Real provider adapters (Agent 1). Risk estimation (Agent 3). Do not change `core/`.

## Done when
`SyntheticDataSource` and `CsvDataSource` satisfy `DataSource`, tests pass offline,
and the synthetic generator demonstrably recovers a known covariance.
