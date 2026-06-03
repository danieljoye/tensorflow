# Daily deep-history data sources (longest-span free daily panel)

> Scope: source the **longest-span free daily** price history reachable from this
> sandbox for a multi-asset risk-budgeting universe (US equity, long
> Treasuries/bonds, gold), assemble it into an aligned offline fixture, and wire a
> loader. Companion to `docs/data-sources.md` (which settled the *monthly*
> deep-history backbone — Shiller + datahub gold). The prototype here is
> `riskbudget/data/providers/daily.py`, fixture `examples/data/daily_panel_long.csv`,
> tests `tests/test_data_daily_panel.py`.

## TL;DR

The longest aligned **daily** multi-asset panel verified in this environment:

| Panel | Assets | Span (daily) | Rows |
|-------|--------|--------------|------|
| `examples/data/daily_panel_long.csv` | `STOCKS`, `STOCKS_TR`, `BONDS` (10Y TR proxy), `GOLD` | **1968-04-01 → 2024-02-26** | **13,593** |

The start date (**1968**) is bounded by the **gold** series; the **end** date is
now bounded by the **STOCKS** (Stooq SPX) mirror, which ends 2024-02-26. Equity
reaches all the way back to **1885** daily and the bond yield to **1962** daily —
but with no free *daily* gold before 1968 in the allowlist, the three-asset
intersection begins 1968-04-01 (the first row of the current LBMA gold mirror).
(Adding daily VIX, available 1990+, would further shrink the
intersection to 1990, so VIX is left out of the committed panel and noted below
as a bonus risk series.)

Everything below was **verified by actually fetching/loading the bytes** (not by
trusting READMEs). All sources are reachable via `raw.githubusercontent.com`,
the only market-data-bearing host in this environment's allowlist.

## Network reality (this environment)

Outbound network is an **allowlist**: only `raw.githubusercontent.com` /
`github.com` and PyPI (`pypi.org`, `files.pythonhosted.org`) resolve. Every
dedicated market-data host (Stooq, Yahoo/yfinance, FRED, datahub's own host,
LBMA, Quandl, …) returns *Host not in allowlist*. So daily data must come from
(a) a CSV hosted in a GitHub repo, or (b) a PyPI package that *bundles* data in
its wheel. PyPI fama-french-style packages were checked and **download at runtime**
(network-blocked) — e.g. `getFamaFrenchFactors` ships a 4.6 kB wheel with no
bundled CSV — so option (a) is what actually works here.

## Per-asset deepest daily sources (ranked, verified)

### 1. US equity — daily S&P 500 / SPX back to **1885** ✅ deepest

| Field | Value |
|-------|-------|
| Series | `STOCKS` (SPX daily close) |
| URL | `https://raw.githubusercontent.com/ai357060/flower/master/Data/spx_d.csv` |
| Daily span | **1885-01-01 → 2024-02-26** (true daily; ~300 obs/yr pre-1952 incl. Saturday sessions, ~252/yr after) |
| Rows | 39,141 total; 1789–1884 rows are **monthly-spaced backfill** and are dropped (`_STOCKS_DAILY_START = 1885-01-01`) |
| Format | Stooq export, Polish OHLCV header `Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen`; close is col 5 |
| Quality | No NaN / non-positive / duplicate closes in the daily region; price-only index (not total return) |
| Confidence | High (loaded & spacing-profiled) |

Runner-up daily equity mirrors found but shallower: Yahoo `^GSPC` dumps in many
ML repos (typically 2001/2005/2010+), `QuantSoftware/.../$SPX.csv` (2012+).

Load snippet:
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/ai357060/flower/master/Data/spx_d.csv")
spx = pd.to_numeric(df.iloc[:, 4]).set_axis(pd.to_datetime(df.iloc[:, 0]))  # daily SPX close
```

### 2. Long Treasuries / rates — daily 10Y yield back to **1962** ✅

| Field | Value |
|-------|-------|
| Series | `BONDS` = constant-duration 10Y **total-return PROXY** derived from daily `DGS10` yield |
| URL | `https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/DGS10.csv` |
| Daily span | **1962-01-02 → 2025-12-16** (FRED `DGS10` constant-maturity 10Y) |
| Rows | 16,686 (712 blank/`.` non-business placeholders skipped) |
| Format | `observation_date,DGS10` (percent yield) |
| Proxy | `r_t = y_{prev}/252 − ModDur·Δy`, `ModDur = 8.0`, cumulated to a growth index rebased to 1.0 — the **same** methodology the Shiller provider documents (ignores convexity/roll-down/cash-flow schedule; illustrative, not production bond analytics) |
| Confidence | High (loaded; identical 1962-01-02 start cross-checked against several independent repo mirrors — Apress, FeanorKingofNoldor, lukasz-f, etc.) |

Direct daily bond-*ETF* total returns (`TLT`/`IEF`) only start ~2002, so the
yield-proxy is both deeper and consistent with the existing monthly provider.

### 3. Gold — daily LBMA PM fix back to **1968**, **current & auto-updating** ✅ (binds the panel start)

| Field | Value |
|-------|-------|
| Series | `GOLD` (USD/oz daily LBMA Gold Price PM fix) |
| URL | `https://raw.githubusercontent.com/unbalancedparentheses/forex-centuries/main/data/sources/lbma/lbma_gold_daily.csv` |
| Daily span | **1968-04-01 → 2026-02-25** (current; ~14,544 rows) |
| Auto-update | **Yes** — `forex-centuries` runs a weekly GitHub Actions cron (`--all`, Mondays 06:00 UTC), so the live `from_github()` transport stays fresh through the present |
| Format | `date,gold_pm_usd,gold_pm_gbp,gold_pm_eur`; only the USD column (2nd col, consumed by `_parse_two_col`) is used — 0 blank/non-positive USD rows |
| Confidence | High (loaded & spacing-profiled; overlap-validated, see below) |

**Replaces** the previously-used static `UtaHagen/PortfolioProject`
(`economic_indicators_daily.csv`) mirror, which ended **2023-12-28** and was the
binding stale leg. Over the **14,001-day overlap** (1968-04-01 → 2023-12-28) the
two series agree to a **mean relative difference of ~0.4%** (max abs ~$90 on a
handful of isolated days attributable to AM/PM-fix and spot/fix timing), so the
swap preserves the historical level while extending currency to 2026 and making
the live transport self-refreshing. No splice was needed: the new mirror is both
deep (1968) **and** current in a single file.

Why not deeper than 1968: free *daily* gold before 1968 does not exist in the
allowlist — the LBMA daily fix itself only begins 1968, and the datahub
`gold-prices` mirror (used by the monthly provider) is **monthly-only** (no
`data/daily.csv`; verified 404). The current `FeziweMelvin/XAUUSD-Gold-Price`
mirror (`XAU_1d_data.csv`, auto-updated weekdays, last row 2025-06) was also
verified current but only reaches **2004**, so `forex-centuries` (1968 + current)
is preferred.

### Bonus: daily VIX back to **1990**

| Field | Value |
|-------|-------|
| Series | VIX close |
| URL | `https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv` |
| Daily span | **1990-01-02 → 2026-06-01** (9,197 rows) |
| Format | `DATE,OPEN,HIGH,LOW,CLOSE` |

Excluded from the committed panel because an inner join with VIX would cut the
multi-asset start from 1968 to 1990. Easy to add as a separate risk overlay.

## The assembled panel

`examples/data/daily_panel_long.csv` — `Date` index + `STOCKS`, `STOCKS_TR`,
`BONDS`, `GOLD` columns, **inner-joined** daily, dropna, **1968-04-01 →
2024-02-26, 13,593 rows**. `BONDS` is rebased to 1.0 and `STOCKS_TR` to 100 on the
first aligned row. The start is bound by gold (1968-04-01), the end by the STOCKS
mirror (2024-02-26).

```python
from riskbudget.data.providers import DailyPanelDataSource
from datetime import date
panel = DailyPanelDataSource.from_fixtures().get_prices(
    ["STOCKS", "STOCKS_TR", "BONDS", "GOLD"], date(1968, 1, 1), date(2024, 12, 31)
)
```

The loader mirrors the Tiingo/Shiller/gold two-transport pattern:
`from_fixtures()` (offline, reads the committed CSV) and `from_github()` (live,
downloads the three sources above and re-assembles via `assemble_panel`). The
live re-assembly was verified to reproduce the committed fixture (the fixture is
regenerated from the four live sources via `assemble_panel`; 13,593-row index),
so the offline tests exercise the same alignment/derivation that runs live.
Registered as data source `daily_panel`.

## Caveats

- `STOCKS` is a **price** index (no dividends) — for total-return equity deep
  history use the monthly Shiller `SP500_TR` provider; this daily series is for
  daily-frequency risk-budgeting demos.
- `BONDS` is a **constant-duration (~8y) yield proxy**, not an actual bond-index
  total return (no convexity, roll-down, or on-the-run adjustment).
- Sources are third-party GitHub mirrors of Stooq/FRED/LBMA dumps; they are
  pinned by URL but not cryptographically versioned. The committed fixture is the
  reproducible offline truth; `from_github()` is the convenience refresh path.
- Honest bound: the deepest *aligned multi-asset daily* panel starts **1968**,
  gated by daily gold (1968-04-01), and currently ends **2024-02-26**, gated by
  the STOCKS mirror. Equity alone is daily to 1885 and the bond yield to 1962; the
  gold and bond mirrors auto-update past 2024, so the binding end leg is now
  equity rather than gold.
