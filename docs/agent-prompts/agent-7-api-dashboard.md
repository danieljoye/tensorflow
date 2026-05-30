# Agent 7 — API, Dashboard & CLI

**Wave:** 3 (last). **Owns:** `riskbudget/api/*`, `riskbudget/dashboard/*`,
`riskbudget/cli/*`. **Depends on:** Agent 0 + integrates all prior agents.

## Scope
Expose the whole pipeline to humans and services.
1. `api/schemas.py` — pydantic v2 request/response models for portfolio
   construction and backtesting.
2. `api/app.py` — FastAPI app with at least:
   - `POST /construct` — given assets, a budget spec, a risk-model choice, and
     constraints, return weights + risk contributions.
   - `POST /backtest` — given the above plus a date range and schedule, run a
     backtest and return metrics + (optionally) the equity curve.
   - `GET /health`.
   Wire data → risk model → optimizer → backtester → analytics via the `core`
   interfaces; select implementations by name (e.g. `risk_model="ledoit_wolf"`).
3. `dashboard/app.py` — Streamlit UI: pick a universe, budget, risk model, and
   date range; show weights, a risk-contribution chart, the equity curve, and the
   summary metrics table. May call the API or the library directly.
4. `cli/main.py` — `construct` and `backtest` subcommands mirroring the API,
   reading CSV input and writing JSON/report output.

## Interfaces to honor
Only the public `core` types and each package's public entry points — do not reach
into other packages' internals.

## Deliverables
API + dashboard + CLI, with tests: FastAPI `TestClient` checks `/construct` and
`/backtest` return schema-valid responses on the synthetic source; CLI smoke test;
dashboard import/build test (no live run required in CI).

## Out of scope
Auth/multi-tenant, live trading, changing any other package's internals or `core/`.

## Done when
Endpoints return valid responses end-to-end on synthetic data, the CLI runs, the
dashboard imports and builds, and tests pass.
