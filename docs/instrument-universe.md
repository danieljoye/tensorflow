# Instrument Universe & Price-History Catalog

> The investable universe for the Portfolio Risk Budgeting system. Scope (agreed):
> **broad US-traded universe**, **spliced for maximum history**, organized **by
> exposure** so each line shows the three vehicles side by side — the **futures
> contract**, the **ETF equivalent**, and the **deep cash/spot price history**.
> Commodities are **energy + base metals + precious metals only** (no grains/softs/
> livestock, no micros). **No FX/currency futures.** Equity-index and commodity
> **micros are excluded.** Machine-readable version: `docs/instrument_universe.csv`.

**Confidence flags:** `[H]` verified against exchange/issuer/primary or multiple
sources · `[M]` single reputable source / known-but-verify · `[L]` approximate.

**Splice rules:** instrument from inception forward, back-filled with the underlying
series before that (seam = inception). Splice adjusted/total-return to **TR** series,
raw price to **price** series — never mix. For futures, splice to the **cash index**
(price); **do not** naively splice spot onto a futures roll (back-adjust). ETF
adjusted-close ≈ total return; splice it to the index **TR**.

---

## 1. Equities

### 1a. Benchmark universes (tradable baskets)

| Universe | # names | Rebalance | Free point-in-time membership | Conf. |
|---|---|---|---|---|
| Dow 30 | 30 | Ad hoc (committee) | Wikipedia to 1896 | [H] |
| S&P 100 (OEX) | ~100 | Quarterly | current list only | [M] |
| **S&P 500** | 500 | Quarterly + ad hoc | GitHub change logs, reliable ~2000+ | [H] |
| S&P MidCap 400 | 400 | Quarterly | weak free PIT | [M] |
| S&P SmallCap 600 | 600 | Quarterly | weak free PIT | [M] |
| Nasdaq-100 | 100 | Annual (Dec) | Wikipedia changes | [H] |
| Russell 1000 / 2000 / 3000 | 1k / 2k / 3k | Annual Jun → **semi-annual 2026** | licensed (FTSE Russell / Norgate) | [H] |

PIT membership matters: build the universe as it *was* on each date (incl. later-delisted
names) or backtests are survivorship-biased. Free S&P 500 membership ~2000+; Russell is licensed.

### 1b. Deep-history anchor stocks (explicit — longest single-name daily series)

Free retail feeds bottom at the **~1962 wall**; only **CRSP** (paid/academic) goes deeper —
daily to **1926** (Pre62), monthly to **Dec 1925**. Per-name earliest:

| Ticker | Company | GICS sector | Free-feed daily | CRSP earliest |
|---|---|---|---|---|
| GE | General Electric | Industrials | 1962 | **1925-12** (NYSE 1892) |
| IBM | IBM | Info Tech | 1962 | **1925-12** (1915) |
| KO | Coca-Cola | Staples | 1962 | **1925-12** (1919) |
| PG | Procter & Gamble | Staples | 1962 | **1925-12** (1891) |
| XOM | Exxon Mobil | Energy | 1962 | **1925-12** |
| CVX | Chevron | Energy | 1962 | **1925-12** |
| MO | Altria | Staples | 1962 | **1925-12** |
| PEP | PepsiCo | Staples | 1962 | **1925-12** |
| GD | General Dynamics | Industrials | 1962 | **1925-12** |
| HON | Honeywell | Industrials | 1962 | **1925-12** |
| T | AT&T | Comm Svcs | 1962 | 1925-12 (old AT&T; entity break 1984) |
| GIS | General Mills | Staples | 1962 | 1928 |
| CAT | Caterpillar | Industrials | 1962 | 1929 |
| BA | Boeing | Industrials | 1962 | 1934 |
| MRK | Merck | Health Care | 1962 | ~1946 |
| MMM | 3M | Industrials | 1962 | 1946 |
| JNJ | Johnson & Johnson | Health Care | 1962 | 1944 |
| PFE | Pfizer | Health Care | 1962–72 | 1944 |
| F | Ford Motor | Cons Disc | 1962–72 | 1956 |
| DIS | Walt Disney | Comm Svcs | 1962 | 1957 |
| HPQ | HP Inc | Info Tech | 1962 | 1961 |
| MCD | McDonald's | Cons Disc | 1966 | 1965 |
| JPM | JPMorgan Chase | Financials | ~1980 | 1969 |
| WMT | Walmart | Staples | 1972 | 1970 |
| INTC | Intel | Info Tech | ~1980 | 1972 (Nasdaq) |
| AXP | American Express | Financials | 1972 | 1972 |
| HD | Home Depot | Cons Disc | 1981 | 1981 |
| GM | General Motors | Cons Disc | 2010 | 2010 (new entity) |

> **Entity ≠ ticker:** GM (old delisted 2009, new IPO 2010), T (1984 breakup), DD/DOW (2017
> merge / 2019 split), HON (AlliedSignal 1999). Bias-free history must follow CRSP PERMNO chains
> and apply delisting returns — not the ticker.

### 1c. Sector representation (current liquid large caps, by GICS)

IT: AAPL MSFT NVDA AVGO ORCL · Comm Svcs: GOOGL META NFLX DIS T · Cons Disc: AMZN TSLA HD MCD
NKE · Staples: PG KO PEP COST WMT · Health: LLY JNJ UNH ABBV MRK · Financials: BRK.B JPM V MA
BAC · Industrials: CAT GE BA HON UNP · Energy: XOM CVX COP SLB EOG · Materials: LIN SHW FCX NEM
ECL · Real Estate: AMT PLD EQIX WELL SPG · Utilities: NEE SO DUK CEG AEP.

### 1d. Single-stock daily history by source

CRSP **1962-07-02** daily / **1926** Pre62 / **1925-12** monthly (bias-free, incl. delisting
returns) · Norgate **1950** (bias-free) · Sharadar **1998** (bias-free) · Tiingo / yfinance /
Stooq **~1962** (survivor-only). Pre-1962 daily single-stock is CRSP-only.

---

## 2. Equity-Index Exposures (futures + ETF + cash history, stitched)

No micros. Each exposure has a futures contract, an ETF equivalent, and deep cash-index history.

| Underlying | Futures (inception) | ETF (inception) | Cash-index history (earliest) |
|---|---|---|---|
| **S&P 500** | ES 1997-09 (SP 1982-04) | **SPY 1993-01** (IVV/VOO) | cash **1957**; Shiller composite **1871**; TR daily 1988 |
| **Nasdaq-100** | NQ 1999-06 (ND 1996-04) | **QQQ 1999-03** | NDX cash **1985-01** |
| **Dow 30** | YM 2002-04 | **DIA 1998-01** | DJIA **1896** |
| **Russell 2000** | RTY 2017-07 (CME 1993→ICE 2008→CME 2017) | **IWM 2000-05** | RUT cash **1987** |
| **S&P MidCap 400** | EMD ~2002 (thin) | **MDY 1995-05** | S&P 400 **1991** |
| **S&P SmallCap 600** | SMC (illiquid — use ETF) | **IJR 2000-05** | S&P 600 **1994** |
| US total market | — | **VTI 2001-05** (VTSMX 1992) | — |
| Russell 1000 / Value / Growth | — | IWB / IWD / IWF 2000-05 | Russell **1978-12** |
| Nikkei 225 (intl, optional) | NKD 1990-09 | (EWJ proxy) | Nikkei cash **1950** |

> SP full-size **delisted 2021-09-17** (→ ES). Splice futures to the **price** cash index.
> Deepest *daily* equity-market total return (free): Ken French market TR from **1926-07**.

**Sector ETFs (Select Sector SPDRs):** XLK XLF XLE XLV XLI XLY XLP XLU XLB all **1998-12-16**;
**XLRE 2015-10** (carve-out from XLF) and **XLC 2018-06** (GICS realignment) have no pre-launch
ETF price — index back-history only.

---

## 3. Volatility

| Exposure | Futures (inception) | ETF (inception) | History note |
|---|---|---|---|
| VIX | **VX 2004-03-26** (clean ~2007) | VXX 2009-01 / VIXY 2011-01 | VIX index back-calc **1990** (VXO **1986** for '87 crash) |

> Do **not** splice spot VIX onto VX futures (term structure/contango) — spot is context only.
> VXX/VIXY suffer roll decay; not buy-and-hold instruments.

---

## 4. Treasuries & Fixed Income (futures + ETFs + history)

### 4a. Treasury exposures by tenor (futures + ETF, stitched)

| Tenor | Futures (inception) | Treasury ETF(s) (inception) | Deep history |
|---|---|---|---|
| Bills (0–3mo) | ZQ 1988-10 · SR3/SR1 2018 (Eurodollar **1981**→SOFR) | BIL 2007 · SHV 2007 | TB3MS **1934** |
| 1–3y (~2Y) | **ZT 1990** | **SHY 2002** · VGSH 2009 | DGS2 1976 |
| 3–7y (~5Y) | **ZF 1988** | IEI 2007 · VGIT 2009 | DGS5 1962 |
| 7–10y (~10Y) | **ZN 1982** · TN 2016 | **IEF 2002** | DGS10 **1962**, GS10 **1953**, Shiller **1871** |
| 10–20y | — | TLH 2007 | — |
| 20y+ (long) | **ZB 1977** · UB 2010 (splice ZB→UB) | **TLT 2002** · VGLT 2009 | Bloomberg Long Tsy TR **1973** |
| STRIPS 25–30y (zero-coupon) | — | EDV 2007 · ZROZ 2009 | — |
| Broad Treasury | — | **GOVT 2012** | Bloomberg US Tsy TR **1973** |

> ZB's deliverable basket drifted shorter — splice **ZB→UB** (back-adjust) for a constant
> long-end. Treasury ETFs are all 2002-and-later (first iShares Treasury ETFs **2002-07-22**),
> so for pre-2002 history splice to the Bloomberg (ex-Barclays/Lehman) Treasury TR sub-indices
> (Long Treasury TR to **1973**) or reconstruct from FRED constant-maturity yields (**1962**).

### 4b. Broader fixed income (credit / aggregate ETFs)

| Exposure | ETF (inception) | Splice to (earliest) |
|---|---|---|
| US Aggregate | **AGG 2003** · BND 2007 | Bloomberg US Agg TR **1976** |
| IG corporate | **LQD 2002** | Bloomberg US Corp IG TR **1973** |
| High yield | **HYG 2007** · JNK 2007 | Bloomberg US HY TR **1983** |
| TIPS (inflation) | TIP 2003 | Bloomberg US TIPS ~**1997** |
| Agency MBS | MBB 2007 | Bloomberg US MBS **1976** |

### 4c. Yield & total-return history (the deep backbone)

**Yields (FRED; not returns):** Shiller long rate **1871** → GS10 monthly **1953** → DGS10 daily
**1962**; short rate TB3MS **1934**; DGS30 1977 (gap 2002–06). **Total return (for backtests):**
Damodaran 10Y **1928** (annual), Ibbotson/SBBI **1926** (pre-1977 modeled), Bloomberg US Agg **1976**.

---

## 5. Commodities — Energy, Base Metals, Precious Metals (futures + ETF + spot)

No micros; no grains/softs/livestock. Futures-backed ETFs track a **rolled-futures index, not
spot** (they diverge under contango); physical-metal ETFs track spot.

| Exposure | Futures (inception) | ETF (inception) | Deep spot/annual history |
|---|---|---|---|
| **WTI crude** | **CL 1983-03** | USO 2006 (roll) | WTI annual **1946**, EIA daily 1986 |
| **Brent crude** | BZ 1988-06 | BNO 2010 (roll) | Brent spot **1987** |
| **Natural gas** | **NG 1990-04** | UNG 2007 (roll; contango drag) | EIA Henry Hub spot 1997 |
| Heating oil/ULSD | HO 1978 | (in DBE) | — |
| Gasoline (RBOB) | RB 2005-10 (HU 1984) | UGA 2008 (roll) | — |
| Energy basket | — | DBC 2006 · DBE 2007 · GSG 2006 | DBIQ ~1989 / GSCI **1970** |
| **Gold** | **GC 1974-12** | **GLD 2004** · IAU · SGOL (spot) | gold annual **1833**, LBMA fix **1968** |
| **Silver** | **SI 1963** | **SLV 2006** · SIVR (spot) | LBMA silver; continuous ~1975 |
| Platinum | PL 1956 | PPLT 2010 (spot) | platinum spot |
| Palladium | PA 1968 | PALL 2010 (spot) | palladium spot |
| **Copper** | **HG 1988-07** | CPER 2011 (roll) | copper **1959** |
| Aluminum | ALI 2014-05 | (none liquid US) | LME for deep history |
| Precious basket | — | DBP 2007 | — |

**Broad commodity indices (deep splice anchors):** CRB **1957** (oldest live), S&P GSCI back-tested
**1970**, Bloomberg BCOM back-calc **1960** (GSCI/BCOM pre-1991 are hypothetical — flag in backtests).

---

## 6. Deep-history index/price backbone (splice anchors)

| Series | Earliest | Free source |
|---|---|---|
| Shiller S&P Composite (monthly real TR) | **1871** | econ.yale.edu/~shiller |
| Ken French US market TR (daily) | **1926-07** | Dartmouth |
| Dow Jones Industrial Average | **1896** | S&P DJI / `^DJI` |
| S&P 500 (official) | **1957** (TR daily 1988) | FRED / `^GSPC` |
| Damodaran asset-class annual TR | **1928** | NYU Stern |
| Gold annual (NMA) | **1833** | nma.org |
| WTI annual (Macrotrends) | **1946** | macrotrends.net |
| VIX index (VXO methodology) | **1986** (VIX 1990) | Cboe |

---

## 7. Data vendors by access tier

- **Free, offline-capable here:** synthetic + CSV; plus the free deep-history *files* (Shiller,
  Ken French, Damodaran, NMA gold, FRED) — static CSV, committable as offline fixtures.
- **Freemium + key (network-gated):** Tiingo (best free real EOD; equities/ETFs to 1962), Polygon,
  Alpha Vantage; Stooq / yfinance / FRED-API (free, network-gated).
- **Paid, systematic-grade:** **Norgate** (stocks 1950 + futures, survivorship-free, best value),
  **CSI** (deepest futures), **Pinnacle** (deep futures), **Stevens/SCF**, **Sharadar** (bias-free
  equities 1998), EODHD.
- **Paid, institutional:** CRSP/WRDS (1926), Bloomberg, Refinitiv/LSEG.

Futures continuous-contract vendors: **Norgate / CSI / Pinnacle / Stevens-SCF** (the old free
Quandl CHRIS is discontinued, 2018).

---

## 8. Reachability from this build environment

The network allowlist blocks **every** market-data host (only PyPI/PythonHosted/GitHub resolve).
The system runs here on **synthetic + committed CSV fixtures**; this catalog is the *sourcing plan*
for when it runs where a network path or paid vendor exists. The free deep-history files
(Shiller/French/Damodaran/NMA/FRED) are static downloads committable as offline fixtures.

---

## 9. Machine-readable catalog

`docs/instrument_universe.csv` — one row per instrument. Schema:

```
asset_class, symbol, description, instrument_type, native_start,
max_history_start, history_source, splice_chain, live_source, confidence, notes
```

Asset classes: `equity_universe`, `equity`, `equity_index`, `equity_index_future`, `etf`,
`vol_future`, `vol_etf`, `treasury_future`, `treasury_etf`, `credit_etf`, `rate_series`,
`bond_total_return`, `commodity_future`, `commodity_etf`, `commodity_index`, `commodity_spot`.
Intended as the seed for `StrategySpec.universe`.

---

## 10. Verification caveats

Publisher/exchange/FRED pages largely block automated fetch (403); dates verified via concurring
secondary indexes and issuer pages, per-row `[H]/[M]/[L]`. Added ETF inception dates flagged `[M]`
("verify day") come from known issuer facts, not a live fetch — confirm exact first-trade days in
the prospectus/SEC 497 if day-level precision matters (especially the Vanguard/abrdn/Invesco lines:
VGSH/VGIT/VGLT, EDV, ZROZ, SGOL/SIVR/PPLT/PALL, CPER, BNO, UGA, DBE, DBP, IEI, TLH, SHV). Futures
continuous-series start dates are vendor-dependent; commodity futures here all have modern contracts
whose vendor data ≈ inception (energy/metals), unlike the (now-excluded) grains/softs.
