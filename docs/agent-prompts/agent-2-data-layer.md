# Agent 2 — Data Layer & Simulation

**Wave:** 1 (parallel). **Owns:** `riskbudget/data/synthetic.py`,
`riskbudget/data/csvsource.py`, `riskbudget/simulate/gbm.py`.
**Depends on:** Agent 0 (`core/`).

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
3. `simulate/gbm.py` — geometric Brownian motion price-path simulation (per the
   EDHEC course `gbm()`): given `n_years`, `steps_per_year`, drift `mu`, vol
   `sigma`, `n_scenarios`, and a seed, return simulated price/return paths as a
   DataFrame (and convertible to `PriceData`). Add `terminal_values()` /
   `terminal_stats()` helpers summarizing wealth outcomes across scenarios
   (mean, median, percentiles, and probability of breaching a floor). This feeds
   the dynamic-allocation agent (8) and stress testing.

## Interfaces to honor
`DataSource.get_prices(...) -> PriceData`; produce `PriceData`/`ReturnMatrix` as
defined in `core/types.py`.

## Deliverables
Both sources + tests: synthetic covariance recovery within tolerance for large N
samples; CSV round-trip (write fixture -> load -> compare); missing-data handling.

## Out of scope
Real provider adapters (Agent 1). Risk estimation (Agent 3). Do not change `core/`.

Add tests: GBM simulated log-returns recover the target `mu`/`sigma` within
sampling error; `terminal_stats` summarizes scenarios correctly.

## Done when
`SyntheticDataSource` and `CsvDataSource` satisfy `DataSource`, GBM simulation
matches its parameterization, tests pass offline, and the synthetic generator
demonstrably recovers a known covariance.
