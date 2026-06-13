# Data-source evaluation & recommendation (Agent 1)

> Scope: settle the market-data question for the Portfolio Risk Budgeting system
> with evidence (BUILD_PLAN §8), then prototype the winner. This document is the
> evaluation; the prototype is `riskbudget/data/providers/tiingo.py` with offline
> tests in `tests/test_data_provider.py`.

## TL;DR recommendation

1. **Ship `synthetic` + `csvsource` as the always-works core.** They need no
   network, no key, and no third-party host — so the entire test-suite, CI, the
   examples, and the dashboard run deterministically offline. This is the
   backbone (Agent 2 owns those two modules).
2. **Add Tiingo as the first *real* provider adapter** (`data/providers/tiingo.py`),
   built here. Clean REST/JSON, split- and dividend-**adjusted closes**, deep
   history, generous free tier, explicit API key (no scraping). It is the
   recommended real-money path.
3. **Keep yfinance as a documented convenience fallback** (not built in v1): zero
   key, broad coverage, but an undocumented/unofficial endpoint with ToS and
   stability caveats — fine for ad-hoc exploration, not for a production path.
4. **FRED** is the macro/factor overlay (risk-free curve, CPI) rather than an
   equity price source; relevant when the risk-free-rate convention (§3.1) needs
   a real series. **Polygon/EOD** are the noted paid options for a future
   real-money deployment.

**This environment cannot reach any of the real providers** (see the probe
below), so the recommendation is shaped accordingly: the real adapter is
**optional and network-gated**, and it is developed/tested against a committed
recorded fixture so it is green offline. The synthetic+CSV core is what actually
runs here and in CI.

## Network probe (empirical, this environment)

Outbound network in this build environment is an **allowlist**. Probed with
`curl` on 2026-05-31:

| Host | Endpoint probed | HTTP | Result |
|------|-----------------|------|--------|
| api.tiingo.com | `/tiingo/...`, `/api/test` | 403 | `Host not in allowlist` |
| query1.finance.yahoo.com (yfinance/Yahoo) | `/v8/finance/chart/AAPL` | 403 | `Host not in allowlist` |
| stooq.com | `/q/d/l/?s=aapl.us&i=d` | 403 | `Host not in allowlist` |
| www.alphavantage.co | `/query?...` | 403 | `Host not in allowlist` |
| api.stlouisfed.org (FRED) | `/fred/series/observations` | 403 | `Host not in allowlist` |
| api.polygon.io | `/v2/aggs/ticker/AAPL/...` | 403 | `Host not in allowlist` |
| **pypi.org** | `/simple/tiingo/` | **200** | reachable |
| **files.pythonhosted.org** | `/` | **200** | reachable |
| **github.com** | `/` | **200** | reachable |

**Conclusion:** only PyPI, PythonHosted, and GitHub resolve. **Every** market-data
host — including the recommended Tiingo — returns `403 Host not in allowlist`.
This is the single most important constraint and it gates the whole design: any
real adapter must be *optional* and must be testable *without* its host. Hence
the synthetic+CSV core, and the fixture-backed prototype.

## Scoring matrix

Scale: ✅ strong / 🟠 ok / ⚠️ weak / ❌ blocking — for *this project's* needs
(daily EOD prices for risk budgeting, up to ~500 assets × ~5000 periods, §3.1).

| Dimension | synthetic | CSV/parquet | **Tiingo (free)** | yfinance | Stooq | Alpha Vantage | FRED | Polygon (paid) |
|---|---|---|---|---|---|---|---|---|
| **This env's network policy** | ✅ none needed | ✅ none needed | ❌ 403 blocked | ❌ 403 blocked | ❌ 403 blocked | ❌ 403 blocked | ❌ 403 blocked | ❌ 403 blocked |
| **Licensing / ToS** | ✅ ours | ✅ user's own files | ✅ clear ToS, free non-commercial tier, redistribution limits | ⚠️ unofficial Yahoo endpoint; ToS gray area, no redistribution | 🟠 free, terms thin/ambiguous | ✅ clear free ToS | ✅ public-domain US gov data | ✅ commercial license (paid) |
| **Asset coverage** | ✅ arbitrary synthetic | ✅ whatever you load | ✅ US+intl equities, ETFs, mutual funds, FX, crypto | ✅ very broad global | 🟠 US/EU equities, indices, FX | 🟠 equities/FX/crypto, US-centric | ⚠️ macro/rates only (not equities) | ✅ broad US (+ options/FX paid) |
| **History depth** | ✅ unlimited (generated) | ✅ unlimited (your files) | ✅ decades for US equities | ✅ long, varies | 🟠 decent but spotty | 🟠 ~20y daily | ✅ long macro series | ✅ long (paid tiers) |
| **Corporate actions / adjusted close** | n/a | depends on file | ✅ `adjClose` + `divCash`/`splitFactor` per row | 🟠 `Adj Close` (occasionally inconsistent) | ⚠️ adjusted but undocumented method | 🟠 adjusted endpoint exists | n/a | ✅ adjusted + actions |
| **Survivorship bias** | ✅ none (synthetic) | depends on file | 🟠 includes many delisted (better than free peers) | ⚠️ delisted tickers vanish | ⚠️ survivorship-biased | ⚠️ survivorship-biased | n/a | ✅ point-in-time (paid) |
| **Rate limits** | ✅ none | ✅ none | 🟠 free: ~50 symbols/hr, 1000 req/day, 500 unique symbols/mo | ⚠️ unofficial throttling/blocks | 🟠 informal limits | ⚠️ 5 req/min, 25/day (free) | ✅ generous w/ key | ✅ high (paid) |
| **API ergonomics** | ✅ trivial | ✅ trivial | ✅ clean REST/JSON, stable, documented | 🟠 Python lib, brittle scraping | 🟠 CSV URL, minimal | 🟠 REST/JSON, quirky | ✅ clean REST/JSON | ✅ clean REST/JSON |
| **Key management** | ✅ none | ✅ none | ✅ single API token, env var | ✅ none | ✅ none | ✅ single key | ✅ single key | 🔑 paid key |

### Reading the matrix

- **synthetic / CSV** are the only two that are ✅ on *this env's network policy*.
  They are the production backbone for everything that must run in CI.
- Among the *real* providers, **Tiingo** dominates the free tier on the
  dimensions that matter for a risk model: documented **adjusted closes** with
  per-row dividend/split factors (corporate-action correctness, §3.1), deep
  history, an explicit token (reproducible, no scraping), and a stable JSON API.
  Its only real weaknesses are the rate limits (fine for nightly batch / research
  cadence, not tick-by-tick — which is out of scope per §9) and the free-tier
  redistribution restriction (we cache, we do not redistribute).
- **yfinance** is the pragmatic *fallback*: no key and the broadest coverage, but
  it rides an undocumented Yahoo endpoint with ToS ambiguity, occasional silent
  adjustment glitches, and survivorship gaps (delisted tickers disappear). Good
  for a quick look, not a dependable production source.
- **Stooq / Alpha Vantage** add little over Tiingo for our use case: Stooq's
  adjustment method is undocumented and its terms are thin; Alpha Vantage's free
  rate limit (5 req/min, 25/day) is too tight to pull a 500-asset universe.
- **FRED** is not an equity price source at all — it is the right place to source
  a real **risk-free rate / macro state** series for the §3.1 risk-free
  convention and the conditional-budget state variable, so it is a complementary
  *overlay*, not a competitor.
- **Polygon / EOD Historical Data** are the paid options to graduate to for a
  real-money deployment: point-in-time (survivorship-free) history, high rate
  limits, full corporate actions — at a subscription cost not warranted for v1.

## Why Tiingo for the prototype

It is the best *real* adapter to build first because it scores highest on the
dimensions a risk model is sensitive to:

- **Adjusted close is first-class.** Each record carries `adjClose` alongside raw
  `close`, plus `divCash` and `splitFactor`, so the return series is
  corporate-action consistent without us re-deriving adjustments. The adapter
  reads `adjClose` as the canonical price.
- **Explicit, reproducible auth** (a single `Authorization: Token …` header) — no
  scraping of an undocumented endpoint, so the contract is stable and the key
  management is a one-liner (env var).
- **Clean JSON** that maps directly onto `PriceData`, and history deep enough for
  the §3.1 scale targets.

## The prototype adapter (offline, fixture-backed)

`riskbudget/data/providers/tiingo.py` implements the `DataSource` protocol
(`get_prices(assets, start, end) -> PriceData`) and is exposed as the
registry-friendly factory `tiingo_data_source` (BUILD_PLAN §5.2).

Because the host is blocked here, it ships **two transports that share one
parser**:

- `TiingoDataSource.from_fixtures()` — **offline.** Reads recorded
  `providers/fixtures/tiingo/{symbol}.json` files (committed: `AAPL`, `MSFT`, 40
  daily records each, the exact JSON shape Tiingo returns). This is what the
  tests use; no network, no key.
- `TiingoDataSource.live(api_key)` — **network.** Issues the documented HTTPS GET
  to `api.tiingo.com/tiingo/daily/{symbol}/prices` with `startDate`/`endDate`
  and an `Authorization: Token` header, then runs the **identical** parsing and
  alignment path. It works wherever Tiingo is reachable.

Because both transports funnel through the same `_parse_records` /
`get_prices` code, the offline tests cover the live logic; only the *transport*
differs. `requests` is imported lazily inside the live path, so neither the
package nor its offline tests depend on it.

### How it would work live (reproducible request shape)

```
GET https://api.tiingo.com/tiingo/daily/AAPL/prices
    ?startDate=2023-01-03&endDate=2023-02-27&format=json
    &columns=date,close,adjClose
Authorization: Token $TIINGO_API_KEY
```

returns a JSON array of `{date, close, adjClose, ...}` records. The adapter takes
`adjClose` per asset, restricts to `[start, end]` inclusive, aligns all assets on
a common date index (forward-fill small gaps, drop dates still missing any
asset), and returns a dense `PriceData`. To go live: set `TIINGO_API_KEY`,
construct `TiingoDataSource.live(os.environ["TIINGO_API_KEY"])`, and call
`get_prices(...)` unchanged.

### Tests

`tests/test_data_provider.py` (28 tests, all offline) covers: `DataSource`
protocol conformance, the happy path (correct shape, **adjusted** close used,
returns conversion), requested-order preservation, inclusive date-window
restriction, multi-asset alignment/density, the full error taxonomy
(`DataError` for empty/duplicate assets, unknown symbol, bad window, missing
fixture dir, missing/blank key), the live-transport guard rails (constructed
without network), the shared parser (adjClose selection, normalization,
dedupe, schema errors), and fixture-schema integrity.

## Recommendation, restated

- **Default everywhere (incl. CI):** synthetic + CSV (offline backbone).
- **First real adapter:** **Tiingo** free tier — built here, fixture-tested,
  network-gated; flip to `.live(api_key)` when a network path exists.
- **Convenience fallback (documented, not built v1):** yfinance.
- **Macro/factor overlay:** FRED (risk-free / state variable), not equity prices.
- **Future real-money path:** Polygon or EOD Historical Data (paid,
  survivorship-free).

## Sources actually reached from this environment

Per the allowlist, **no market-data provider page or API was reachable** — every
one returned `403 Host not in allowlist` (table above). The provider
characteristics in the matrix are from each provider's public documentation as of
the author's knowledge, **not** verified by live fetch from this environment. The
only hosts that resolved were `pypi.org`, `files.pythonhosted.org`, and
`github.com` (all HTTP 200), used solely to confirm the allowlist contrast. The
Tiingo JSON *shape* in the fixtures reflects the documented
`/tiingo/daily/{symbol}/prices` response; it was reconstructed for offline use,
not pulled live from here.
