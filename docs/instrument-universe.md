# Instrument Universe & Price-History Catalog

> The investable universe for the Portfolio Risk Budgeting system: the liquid
> US-traded instruments across asset classes, each with the **longest available
> price history**, its **earliest start date**, and the **data source**. Scope
> (agreed): **broad US-traded universe**, **spliced to underlying indices for
> maximum history**. Now covers **173 instruments** across equities, equity-index
> futures, **volatility futures**, **FX/currency futures**, interest-rate/Treasury
> futures, commodity futures (energy/metals/grains/softs/livestock), ETFs, and the
> deep-history index/yield/total-return backbone. Compiled 2026-06-01 from two
> research rounds (10 asset-class sweeps). The machine-readable version is
> `docs/instrument_universe.csv`.

**Confidence flags:** `[H]` verified against exchange/issuer/primary or multiple
concordant sources · `[M]` single reputable secondary source · `[L]` approximate,
verify before relying on it.

**Splice convention:** use the tradable instrument from inception forward, back-fill
with the underlying index/series before that ("seam" = inception). Splice
adjusted/total-return to **TR** indices and raw price to **price** indices — never
mix (a yield-sized discontinuity results). Never splice a **yield** to a **total
return**. For futures, splice to the **cash index** (price), and **do not** naively
splice a spot series onto a futures series (basis/roll discontinuity) — back-adjust.

---

## 0. How the universe is built (survivorship-bias-aware)

- **Membership is point-in-time (PIT):** on each rebalance date use the names actually
  investable then, including later-delisted ones. Building from today's list is
  survivorship-biased and inflates returns.
- **Identifier continuity:** tickers get reused/reassigned (GM, DD, T, HON). Bias-free
  history must key on stable IDs (CRSP PERMNO/PERMCO), not tickers, and apply delisting
  returns so failed names' terminal value is captured, not silently dropped.
- **Free PIT membership:** S&P 500 constituents back to **1996** on GitHub
  (`fja05680/sp500`) `[H]`, reliable from ~2000. Russell has no good free PIT — licensed
  (FTSE Russell) or Norgate.
- **Delisted names** live only in paid/academic sets: **CRSP** (gold standard, incl.
  delisting returns), **Norgate** (~25k delisted since 1950), **Sharadar** (1998), EODHD.

---

## 1. Equities (broad US universe)

### 1a. Benchmark universes (tradable baskets)

| Universe | # names | Rebalance | Free PIT membership | Conf. |
|---|---|---|---|---|
| Dow 30 | 30 | Ad hoc (committee) | Wikipedia to 1896 | [H] |
| S&P 100 (OEX) | ~100 | Quarterly | current list only | [M] |
| **S&P 500** | 500 | Quarterly + ad hoc | GitHub change logs ~2000+ | [H] |
| S&P MidCap 400 | 400 | Quarterly | weak free PIT | [M] |
| S&P SmallCap 600 | 600 | Quarterly | weak free PIT | [M] |
| Nasdaq-100 | 100 | Annual Dec | Wikipedia changes | [H] |
| Russell 1000 / 2000 / 3000 | 1k / 2k / 3k | Annual Jun → **semi-annual 2026** | licensed (FTSE Russell) | [H] |

### 1b. Deep-history anchor stocks (longest single-name daily series)

Free feeds bottom at the **~1962 wall**; CRSP reaches **Dec 1925** daily for the oldest
NYSE names. Representative set (full ~33 in the CSV):

| Ticker | Company | Sector | Free-feed | CRSP earliest |
|---|---|---|---|---|
| GE, IBM, KO, PG, XOM, CVX, MO, PEP, GD, HON | (NYSE pre-1926) | mixed | 1962 | **1925-12** |
| CAT 1929 · GIS 1928 · BA 1934 · MMM 1946 · DIS 1957 · HPQ 1961 | | | 1962 | listing yr |
| JNJ 1944 · MRK 1946 · PFE 1944 · F 1956 · MCD 1965 · WMT 1970 | | | 1962–72 | listing yr |
| INTC 1972 · AXP 1972 · JPM 1969 · HD 1981 · GM 2010 (new) | | | varies | listing yr |

> Entity-discontinuity flags: GM (old delisted 2009, new IPO 2010), T (1984 breakup),
> DD/DOW (2017 merge / 2019 split), HON (AlliedSignal 1999) — follow PERMNO, not ticker.

### 1c. Sector representation (current liquid large caps, GICS)

IT: AAPL MSFT NVDA AVGO ORCL · Comm Svcs: GOOGL META NFLX DIS T · Cons Disc: AMZN TSLA
HD MCD NKE · Cons Staples: PG KO PEP COST WMT · Health: LLY JNJ UNH ABBV MRK · Financials:
BRK.B JPM V MA BAC · Industrials: CAT GE BA HON UNP · Energy: XOM CVX COP SLB EOG ·
Materials: LIN SHW FCX NEM ECL · Real Estate: AMT PLD EQIX WELL SPG · Utilities: NEE SO
DUK CEG AEP.

### 1d. Single-stock daily history by source

CRSP **1962-07-02** daily / **1926** Pre62 / **1925-12** monthly (bias-free) · Norgate
**1950** (bias-free) · Sharadar **1998** (bias-free) · Tiingo/yfinance/Stooq **~1962**
(survivor-only). Pre-1962 daily single-stock = CRSP-only.

---

## 2. Equity-Index Futures

| Root | Underlying | Native inception | Max-history (spliced cash index) | Liq. | Conf. |
|---|---|---|---|---|---|
| **ES** / SP | S&P 500 | 1997-09-09 / 1982-04-21 | S&P 500 cash **1957** (composite 1927) | top | [H] |
| MES | S&P 500 micro | 2019-05-06 | onto ES | top | [H] |
| **NQ** / ND | Nasdaq-100 | 1999-06-21 / 1996-04-10 | Nasdaq-100 cash **1985-01-31** | top | [H] |
| MNQ | Nasdaq-100 micro | 2019-05-06 | onto NQ | top | [H] |
| **YM** / MYM | DJIA | 2002-04-05 / 2019-05-06 | DJIA cash **1896** | top | [H] |
| **RTY** / M2K | Russell 2000 | 2017-07-10 / 2019-05-06 | RUT cash **1987**; CME 1993→ICE 2008→CME 2017 | top | [H] |
| EMD | S&P MidCap 400 | ~2002 | S&P 400 cash **1991** (MD full-size 1992) | sec | [M] |
| SMC | S&P SmallCap 600 | 2000s (illiquid) | S&P 600 cash **1994** (use cash) | sec | [M] |
| NKD | Nikkei 225 (USD) | 1990-09-25 | Nikkei 225 cash **1950** | sec | [M] |

> SP full-size **delisted 2021-09-17** (converted to ES). Splice to the **price** cash
> index. Russell needs cross-venue bridging (CME↔ICE↔CME).

### 2b. Volatility futures

| Root | Name | Exchange | Inception | History note | Conf. |
|---|---|---|---|---|---|
| **VX** | Cboe VIX future | CFE | **2004-03-26** | clean from ~2007 (early years thin); VIX index back-calc **1990** (VXO **1986** for '87 crash) | [H] |
| VXM | Mini VIX future | CFE | 2020-08-10 | onto VX | [H] |

> Do **not** naively splice spot VIX onto VX futures (term-structure/contango) — spot is
> context only; the tradable continuous starts 2004 (clean ~2007).

---

## 3. FX / Currency Futures (CME IMM unless noted)

The CME **IMM** launched the world's first financial/currency futures on **1972-05-16**
(7 contracts: GBP, CAD, DEM, FRF, JPY, MXN, CHF) — the deepest splice anchor in the catalog.

| Root | Pair | Native inception | Max-history (spliced) | Liq. | Conf. |
|---|---|---|---|---|---|
| **6E** | EUR/USD | 1999-01-04 | **DEM (6D) IMM 1972** @ 1.95583 DM/€ → 6E 1999 | top | [H] |
| **6J** | JPY | **1972-05-16** | native (original 7) | top | [H] |
| **6B** | GBP | **1972-05-16** | native (original 7) | top | [H] |
| 6C | CAD | **1972-05-16** | native | sec | [H] |
| 6S | CHF | **1972-05-16** | native | sec | [H] |
| 6A | AUD | 1987-01 | native | sec | [M] |
| 6N | NZD | 2004 | native (thin pre-2004) | sec | [M] |
| 6M | MXN | 1995-05 | modern 1995 (1972 line lapsed — **no splice**) | sec | [H] |
| DX | US Dollar Index | 1985-11-20 (ICE) | USDX index **1973** → DX futures 1985 | sec | [H] |
| M6E / M6A / M6B | micro EUR/AUD/GBP | **2009-03-22** | onto parent (FX micros are 2009, not 2019) | sec | [H/M] |

> **Euro deep-history:** back-adjust 6E onto **Deutsche Mark** futures (liquid from
> mid-1970s) using the 1998-12-31 fixing — *not* the thin pre-euro ECU contract.

---

## 4. Interest-Rate / Treasury Futures

| Root | Underlying | Inception | History note | Liq. | Conf. |
|---|---|---|---|---|---|
| **ZB** / UB | T-Bond / Ultra | 1977-08 / 2010-01 | splice ZB→UB (back-adjust; basket drifted) | top/sec | [H] |
| **ZN** / TN | 10Y / Ultra 10Y | 1982-05 / 2016-01 | TN kept separate from ZN | top/sec | [H] |
| **ZF** | 5Y Note | 1988 | native | top | [H] |
| **ZT** | 2Y Note | 1990 | native | top | [H] |
| **ZQ** | 30-Day Fed Funds | 1988-10 | native (FedWatch) | top | [H] |
| **SR3** / SR1 | 3M / 1M SOFR | 2018-05-07 | **chain GE (Eurodollar) 1981→2023 +26.161bp fallback** into SR3 | top/sec | [H] |
| GE | Eurodollar 3M (legacy) | 1981-12 | first cash-settled future; retired→SOFR 2023-04-14 | hist | [H] |

> **SOFR deep-history:** chain Eurodollar (GE, 1981) into SR3 with the 26.161 bp ISDA
> fallback spread used in CME's actual 2023 conversion → a ~40-year STIR series.

**Yield backbone (FRED, deepest; yields not returns):** Shiller long rate **1871** → GS10
**1953** → DGS10 daily **1962**; short rate TB3MS **1934**; DGS30 1977 (gap 2002–06).
**Bond total-return (for backtests):** Damodaran 10Y **1928** (annual), Ibbotson/SBBI
**1926** (pre-1977 modeled), Bloomberg US Agg **1976**, US Corp IG 1973, US HY 1983.

---

## 5. Commodity Futures

**Energy** (top-tier CL, NG; rest secondary)

| Root | Commodity | Inception | Deep splice | Conf. |
|---|---|---|---|---|
| **CL** / MCL / QM | WTI crude / micro / e-mini | 1983-03-30 / 2021-07 / 2002 | WTI annual **1946**, EIA daily 1986 | [H] |
| BZ | Brent (financial) | 1988-06 | Brent spot 1987 | [H] |
| HO / RB | heating oil/ULSD / RBOB gasoline | 1978 / 2005-10 | RB splice HU 1984 | [H] |
| **NG** / QG / MNG | Henry Hub gas / e-mini / micro | 1990-04 / 2002 / 2023-11 | EIA spot 1997 | [H] |

**Metals** (top-tier GC, SI, HG)

| Root | Commodity | Inception | Deep splice | Conf. |
|---|---|---|---|---|
| **GC** / MGC | gold / micro | 1974-12-31 / 2010-10 | gold annual **1833**, LBMA fix **1968** | [H] |
| **SI** / SIL | silver / micro | 1963 / ~2010 | LBMA silver; continuous ~1975 | [H/L] |
| **HG** / MHG | copper (high-grade) / micro | 1988-07-29 / 2022-05 | copper **1959** | [H] |
| PL / PA | platinum / palladium | 1956 / 1968 | continuous ~1968 / ~1977 | [M] |
| ALI | aluminum (NA physical) | 2014-05-05 | LME for deep history | [H] |

**Grains & oilseeds** (top-tier ZC, ZS, ZW) — pit origins 1877–1947; clean continuous ~1959–73

| Root | Commodity | Pit inception | Continuous | Conf. |
|---|---|---|---|---|
| ZC / ZW / ZO | corn / SRW wheat / oats | 1877 | ~1959 | [H/M] |
| ZS / ZL / ZM | soybeans / oil / meal | 1936 / 1946 / 1947 | ~1959–69 | [H] |
| KE / MWE | HRW / HRS wheat | 1876 / 1883 | ~1970s | [M] |
| ZR | rough rice | 1994 (CBOT) | ~1986 (MidAm) | [M] |

**Softs & livestock** (top-tier SB, KC, LE, HE)

| Root | Commodity | Pit inception | Continuous | Conf. |
|---|---|---|---|---|
| SB / KC / CC / CT | sugar#11 / coffee-C / cocoa / cotton#2 | 1961 / 1960s / 1925 / 1870 | ~1959–73 | [H/M] |
| OJ / LBR | orange juice / lumber | 1966 / 2023 (LBS 1969) | ~1967 / 2023 | [M/H] |
| LE / GF / HE | live cattle / feeder / lean hogs | 1964 / 1971 / 1966 | native | [H] |

> **HE splice break:** live-hog (physical) pre-1997 → lean-hog (cash-settled) 1997; basis
> definition changed. Pit-vs-vendor gap: grains/softs trace to the 1800s but clean
> machine-readable continuous data starts ~1959–1973 (electronic-only from ~2008).

**Broad commodity indices (deep splice anchors):** CRB **1957** (live, oldest), S&P GSCI
back-tested **1970**, Bloomberg BCOM back-calc **1960** (GSCI/BCOM pre-1991 are hypothetical).

---

## 6. Deep-history index backbone (the splice anchors)

| Series | Earliest | Free source |
|---|---|---|
| Shiller S&P Composite (monthly real TR) | **1871** | econ.yale.edu/~shiller |
| Ken French market TR (daily) | **1926-07** | Dartmouth |
| Dow Jones Industrial Average | **1896** | S&P DJI / `^DJI` |
| S&P 500 (official) | **1957** (TR daily 1988) | FRED / `^GSPC` |
| Damodaran asset-class annual TR | **1928** | NYU Stern |
| Gold annual (NMA) | **1833** | nma.org |
| VIX index (VXO methodology) | **1986** (VIX 1990) | Cboe |
| FX (IMM currency futures) | **1972-05-16** | CME |

---

## 7. ETFs (47 liquid names) — see `instrument_universe.csv`

Unchanged from the prior round: broad equity (SPY 1993 → S&P 500 TR 1988; QQQ, IWM, DIA,
VTI, MDY, Russell 1000 family), 11 sector SPDRs (1998-12-16; XLRE 2015, XLC 2018), bond
ETFs (SHY/IEF/TLT/LQD 2002, AGG 2003, HYG 2007 → Bloomberg TR indices 1973–83), commodity
ETFs (GLD 2004→LBMA 1968 spot; futures-backed USO/UNG/DBC/GSG → **rolled-futures indices,
not spot**), and international (EFA, EEM). Full table in the CSV.

---

## 8. Data vendors by access tier

- **Free, offline-capable here:** synthetic + CSV; plus the free deep-history *files*
  (Shiller, Ken French, Damodaran, NMA gold, FRED) — static CSV, committable as fixtures.
- **Freemium + key (network-gated):** Tiingo (best free real EOD; equities/ETFs to 1962),
  Alpha Vantage, Polygon; Stooq/yfinance/FRED-API (free, network-gated).
- **Paid, systematic-grade:** **Norgate** (stocks 1950 + futures, survivorship-free, best
  value), **CSI** (deepest futures, ~1949–59 oldest grains), **Pinnacle** (deep futures,
  cheap), **Stevens/SCF**, **Sharadar** (bias-free equities 1998), EODHD.
- **Paid, institutional:** CRSP/WRDS (1926), Bloomberg, Refinitiv/LSEG.

For **futures**, the canonical continuous-contract vendors are **Norgate / CSI / Pinnacle
/ Stevens-SCF**; the old free Quandl CHRIS is discontinued (2018).

---

## 9. Reachability from this build environment

The network allowlist blocks **every** market-data host (only PyPI/PythonHosted/GitHub
resolve). The system runs here on **synthetic + committed CSV fixtures**; this catalog is
the *sourcing plan* for when it runs where a network path or paid vendor exists. The free
deep-history files (Shiller/French/Damodaran/NMA/FRED) are static downloads committable as
offline fixtures to seed the universe.

---

## 10. Machine-readable catalog

`docs/instrument_universe.csv` — one row per instrument. Schema:

```
asset_class, symbol, description, instrument_type, native_start,
max_history_start, history_source, splice_chain, live_source, confidence, notes
```

173 rows across `equity_universe`, `equity`, `equity_index_future`, `vol_future`,
`fx_future`, `treasury_future`, `commodity_future`, `rate_series`, `bond_total_return`,
`commodity_index`, `commodity_spot`, `equity_index`, `etf`. Intended as the seed for a
`StrategySpec.universe`; extend toward full S&P 500 / Russell PIT membership when a
constituents source is wired in.

---

## 11. Verification caveats

Publisher/exchange/FRED/SSRN pages largely block automated fetch (403); dates verified via
concurring secondary indexes and provider/issuer pages, with per-row `[H]/[M]/[L]`
confidence. Confirm before relying on: platinum/palladium inception `[L/M]`; Micro Silver
SIL launch `[L]`; several FRED DGS starts + DGS30 2002–06 gap `[M]`; EMD/SMC futures `[M]`;
exact per-symbol continuous-series starts (grains/softs ~1959–73, far later than pit); 6A
(1987) and modern 6M (1995) / 6N (2004); CME day-level first-trade dates (CME's "Historical
First Trade Dates" page blocks automated fetch). Full source URLs are in the ten
research-agent reports underlying this catalog.
