# Agent 1 — Data Source Research

**Wave:** 1 (parallel). **Owns:** `docs/data-sources.md` + one prototype adapter
under `riskbudget/data/providers/`. **Depends on:** Agent 0 (`core/` interfaces).

## Scope
Settle the market-data question with evidence, then prototype the winner.
1. Evaluate the candidate set from BUILD_PLAN §8: synthetic (baseline),
   CSV/parquet, Tiingo (free tier), yfinance, Stooq, Alpha Vantage, FRED, and one
   paid option (Polygon/EOD) noted for the future.
2. Score each on: **this environment's network policy compatibility**,
   licensing/ToS, asset coverage, history depth, corporate-action / adjusted-close
   handling, survivorship bias, rate limits, API ergonomics, key management.
3. First, empirically determine what outbound network access this environment
   actually permits (try a couple of candidate endpoints) and record the result —
   it gates everything.
4. Write `docs/data-sources.md`: the scored matrix, the tradeoffs, and a clear
   recommendation.
5. Build **one** working prototype adapter for the recommended provider that
   implements the `DataSource` Protocol, under `riskbudget/data/providers/`. If
   network is blocked, deliver the adapter against a recorded/cached fixture and
   document how it would work live.

## Interfaces to honor
`DataSource.get_prices(assets, start, end) -> PriceData` from `core/interfaces.py`.

## Deliverables
`docs/data-sources.md` (evaluation + recommendation) and one prototype provider
adapter with tests (using cached/fixture data so tests stay offline).

## Out of scope
Do not build the synthetic generator or CSV loader (Agent 2 owns those). Do not
change `core/`.

## Done when
The doc names a recommended provider with justification, the prototype adapter
satisfies `DataSource`, and its tests pass offline.
