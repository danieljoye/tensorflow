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
| `examples/data/daily_panel_long.csv` | `STOCKS`, `BONDS` (10Y TR proxy), `GOLD` | **1968-01-02 → 2023-12-28** | **13,692** |

The start date (**1968**) is bounded by the **gold** series. Equity reaches all
the way back to **1885** daily and the bond yield to **1962** daily — but with no
free *daily* gold before 1968 in the allowlist, the three-asset intersection
begins 1968-01-02. (Adding daily VIX, available 1990+, would further shrink the
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

### 3. Gold — daily spot back to **1968** ✅ (binds the panel start)

| Field | Value |
|-------|-------|
| Series | `GOLD` (USD/oz daily spot, London fix lineage) |
| URL | `https://raw.githubusercontent.com/UtaHagen/PortfolioProject/main/Data/economic_indicators_daily.csv` |
| Daily span | **1968-01-02 → 2023-12-28** |
| Rows | 14,150 (1 NaN, 0 non-positive, 0 duplicate) |
| Format | `date,Gold Price` |
| Confidence | High (loaded & spacing-profiled: ~244–254 obs/yr → genuinely daily) |

Why not deeper: free *daily* gold before 1968 does not exist in the allowlist —
the LBMA daily fix itself only begins 1968, and the datahub `gold-prices` mirror
(used by the monthly provider) is **monthly-only** (no `data/daily.csv`; verified
404). `GLD` ETF daily mirrors are abundant but only start **2004-11-18**. A second
independent 1968 daily mirror (`foresights/public/data_d.csv`, Gold/Silver/Oil)
corroborates the 1968-01-02 start and the 35.18 opening print.

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

`examples/data/daily_panel_long.csv` — `Date` index + `STOCKS`, `BONDS`, `GOLD`
columns, **inner-joined** daily, dropna, **1968-01-02 → 2023-12-28, 13,692 rows**.
`BONDS` is rebased to 1.0 on the first aligned row.

```python
from riskbudget.data.providers import DailyPanelDataSource
from datetime import date
panel = DailyPanelDataSource.from_fixtures().get_prices(
    ["STOCKS", "BONDS", "GOLD"], date(1968, 1, 1), date(2024, 1, 1)
)
```

The loader mirrors the Tiingo/Shiller/gold two-transport pattern:
`from_fixtures()` (offline, reads the committed CSV) and `from_github()` (live,
downloads the three sources above and re-assembles via `assemble_panel`). The
live re-assembly was verified to reproduce the committed fixture **exactly**
(max abs diff 5e-7 on `BONDS`, 0 on `STOCKS`/`GOLD`; identical 13,692-row index),
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
- Honest bound: the deepest *aligned three-asset daily* panel is **1968**, gated
  by daily gold. Equity alone is daily to 1885 and the bond yield to 1962.
