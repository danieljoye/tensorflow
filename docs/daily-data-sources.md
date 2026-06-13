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
| `examples/data/daily_panel_long.csv` | `STOCKS`, `STOCKS_TR`, `BONDS` (10Y TR proxy), `GOLD` | **1968-04-01 → 2025-12-16** | **14,035** |

The start date (**1968**) is bounded by the **gold** series; the **end** date is
now bounded by the **BONDS** (FRED `DGS10`) mirror, which currently ends
2025-12-16. The previously-binding stale STOCKS leg (Stooq SPX, ended 2024-02-26)
has been **spliced** with a current SPX mirror (below) so it now runs to
~2025-12. Equity reaches all the way back to **1885** daily and the bond yield to
**1962** daily — but with no free *daily* gold before 1968 in the allowlist, the
three-asset intersection begins 1968-04-01 (the first row of the current LBMA
gold mirror).
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

### 1. US equity — daily S&P 500 / SPX back to **1885**, now **current** via splice ✅ deepest

The STOCKS leg is a **splice** of two mirrors — a deep-but-static leg and a
shallow-but-current leg — so the series is both ~140-years deep and runs to the
present:

**Deep leg (static, 1885 → 2024-02-26):**

| Field | Value |
|-------|-------|
| URL | `https://raw.githubusercontent.com/ai357060/flower/master/Data/spx_d.csv` |
| Daily span | **1885-01-01 → 2024-02-26** (true daily; ~300 obs/yr pre-1952 incl. Saturday sessions, ~252/yr after) |
| Rows | 39,141 total; 1789–1884 rows are **monthly-spaced backfill** and are dropped (`_STOCKS_DAILY_START = 1885-01-01`) |
| Format | Stooq export, Polish OHLCV header `Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen`; close is col 5 (`_stooq_spx_close`) |
| Quality | No NaN / non-positive / duplicate closes in the daily region; price-only index (not total return); **STATIC** — last commit ends 2024-02-26 |

**Current leg (extends to present):**

| Field | Value |
|-------|-------|
| URL | `https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/Indices.csv` |
| Daily span | **2000-11-20 → 2025-12-18** (current; last "Dump" 2025-12) |
| Format | **Semicolon-delimited** `Dates;EMB US Equity;SPX Index`; `Dates` = `DD.MM.YYYY`, European decimals (comma), SPX in col 3 located by header name (`_indices_spx_close`); Bloomberg `#N/A N/A` cells skipped |
| Auto-update | **No GitHub Actions** — the maintainer manually re-dumps the whole repo (same repo as the `DGS10` bond yield, which is dumped alongside it; both currently 2025-12). Provenance is the same maintainer we already trust for BONDS, so the live `from_github()` refreshes whenever they push. |
| Quality | Loaded; SPX column clean (last two rows duplicate the same close, harmless). The 2nd column `EMB US Equity` is `#N/A N/A` for the SPX-history range and is unused. |

**Splice (`_splice_stocks`):** the deep prices are kept verbatim up to and
including the **seam** (the deep leg's last day that also exists in the current
leg, 2024-02-26); the current leg *after* the seam is multiplied by
`deep[seam] / current[seam]` so the joined level is continuous. **Overlap
agreement:** over the **5,851-day** overlap (2000-11-20 → 2024-02-26) the two SPX
series agree to a **mean relative difference of ~0.0002%** (median 0.0%, max
0.16% on the seam day; **100% of days within 0.5%**) — they are the same SPX, so
the seam scale is ~1.0016 and the splice is invisible.

Runner-up daily equity mirrors found but shallower/static: Yahoo `^GSPC` dumps in
many ML repos (typically 2001/2005/2010+, all static), `vijinho/sp500` (1950+ but
**archived/read-only** Jan-2025), `datasets/s-and-p-500` (current to 2026 but
**monthly**, not daily — it remains our dividend-yield source for `STOCKS_TR`).
The gold repo `unbalancedparentheses/forex-centuries` (deep + auto-updating) was
checked for an equity series but carries **only** forex/gold/macro, no S&P/SPX.

Load snippet (current leg):
```python
import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/juanfp02/commodities_and_sovereigns/main/data/Indices.csv",
    sep=";", decimal=",",
)
spx = pd.to_numeric(df["SPX Index"], errors="coerce").set_axis(
    pd.to_datetime(df["Dates"], format="%d.%m.%Y")
).dropna()  # daily SPX close, 2000-11 → present
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
2025-12-16, 14,035 rows**. `BONDS` is rebased to 1.0 and `STOCKS_TR` to 100 on the
first aligned row. The start is bound by gold (1968-04-01); the end is now bound
by the `DGS10` bond-yield mirror (2025-12-16) — gold runs to 2026-02-25 and the
spliced STOCKS leg to 2025-12-18, so DGS10 stops first.

```python
from riskbudget.data.providers import DailyPanelDataSource
from datetime import date
panel = DailyPanelDataSource.from_fixtures().get_prices(
    ["STOCKS", "STOCKS_TR", "BONDS", "GOLD"], date(1968, 1, 1), date(2026, 12, 31)
)
```

The loader mirrors the Tiingo/Shiller/gold two-transport pattern:
`from_fixtures()` (offline, reads the committed CSV) and `from_github()` (live,
downloads the **five** sources above — deep + current STOCKS, DGS10, gold, Shiller
dividends — splices STOCKS and re-assembles via `assemble_panel`). The committed
fixture was **regenerated from the live sources** via `assemble_panel`
(14,035-row index, 1968-04-01 → 2025-12-16), so the offline tests exercise the
same alignment/derivation/splice that runs live. Registered as data source
`daily_panel`.

## Caveats

- `STOCKS` is a **price** index (no dividends) — for total-return equity deep
  history use the monthly Shiller `SP500_TR` provider; this daily series is for
  daily-frequency risk-budgeting demos.
- `STOCKS` is **spliced** from two mirrors (deep static `ai357060/flower` 1885 +
  current `juanfp02/.../Indices.csv` 2000→present). The two agree to ~0.0002%
  mean over their 5,851-day overlap, so the seam is invisible — but the current
  leg is **manually re-dumped** (no GitHub Actions), so live freshness depends on
  the maintainer's pushes (same provenance as the `DGS10` bond leg).
- `BONDS` is a **constant-duration (~8y) yield proxy**, not an actual bond-index
  total return (no convexity, roll-down, or on-the-run adjustment).
- Sources are third-party GitHub mirrors of Stooq/FRED/LBMA dumps; they are
  pinned by URL but not cryptographically versioned. The committed fixture is the
  reproducible offline truth; `from_github()` is the convenience refresh path.
- Honest bound: the deepest *aligned multi-asset daily* panel starts **1968**,
  gated by daily gold (1968-04-01), and currently ends **2025-12-16**, now gated
  by the `DGS10` bond-yield mirror. Equity alone is daily to 1885 (deep leg) and
  current to ~2025-12 (spliced current leg), gold to 2026-02, and the bond yield
  to 1962→2025-12; after the STOCKS splice the binding end leg is the bond yield,
  not equity.
