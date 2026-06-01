# Instrument Universe & Price-History Catalog

> The investable universe for the Portfolio Risk Budgeting system: the liquid
> US-traded instruments across asset classes, each with the **longest available
> price history**, its **earliest start date**, and the **data source**. Scope
> (agreed): **broad US-traded universe**, **spliced to underlying indices for
> maximum history**. Compiled 2026-06-01 from five asset-class research sweeps;
> see `docs/data-sources.md` for the *software* adapters and BUILD_PLAN §11 for
> research provenance.

**Confidence flags:** `[H]` verified against exchange/issuer/primary or multiple
concordant sources · `[M]` single reputable secondary source · `[L]` approximate,
verify before relying on it.

**Splice convention:** use the tradable instrument from its inception forward, and
back-fill with the underlying index/series before that ("seam" = instrument
inception). Splice an instrument's **adjusted/total-return** series to the index's
**total-return** series, and its **raw price** to the **price** index — never mix
the two (a yield-sized discontinuity results). Never splice a **yield** series to a
**total-return** series.

---

## 0. How the universe is built (survivorship-bias-aware)

- **Membership is point-in-time (PIT).** On each rebalance date the universe must be
  the names that were actually investable *then* — including those later delisted,
  merged, or bankrupted. Building from *today's* index list and pulling full history
  is survivorship-biased: it drops every failure and inflates backtest returns.
- **Free PIT membership:** historical **S&P 500** constituents back to **1996** on
  GitHub (`fja05680/sp500`, `hanshof/sp500_constituents`) `[H]`. **Russell 1000**
  has no reliable free PIT history — it is licensed (FTSE Russell) or via Norgate `[M]`.
- **Delisted names** (to remove the bias) live only in paid/academic sets: **CRSP**
  (gold standard, incl. delisting returns), **Norgate** (~25k+ delisted since 1950),
  **Sharadar SEP** (since 1998), **EODHD** delisted endpoint (~2000). Free retail
  feeds (Yahoo/yfinance, Alpha Vantage, Stooq) **drop delisted tickers** → inherently
  biased.

---

## 1. Equities (broad US universe)

**Universe definition:** broad investable core = **Russell 1000** (~1000 names) or
**S&P 500** (deepest free history, cleanest membership). Reconstruct PIT membership
per rebalance; retain delisted series to their delisting date.

**Single-stock daily price history — earliest start by source:**

| Source | Earliest daily single-stock | Bias-free? | Access | Conf. |
|---|---|---|---|---|
| **CRSP (WRDS)** | **1962-07-02** daily; **1926** daily via Pre62; **1925-12** monthly | Yes (incl. delisting returns) | Paid/academic | [H] |
| **Norgate** | **1950** | Yes (incl. delisted) | Paid | [H] |
| **Sharadar SEP** (Nasdaq Data Link) | **1998** | Yes | Paid (key) | [H] |
| **Tiingo** | **1962** | Partial (active-focused) | Freemium + key | [H] |
| **EODHD** | **~2000** (US); delisted endpoint | Partial→Yes | Paid | [H] |
| **Polygon** | **2004** | Partial | Freemium + key | [H] |
| **yfinance / Yahoo** | **1962** (oldest names) | No | Free (unofficial) | [M] |
| **Stooq** | ~1990s–2000s (earliest unverified) | No | Free bulk CSV | [L] |
| **Alpha Vantage** | ~2000s (full = premium) | No | Free key (limited) | [H] |

**The "1962 wall":** every *free/retail* single-stock daily feed bottoms out ~**2 Jul
1962** (they trace to the CRSP daily file). Pre-1962 *daily* single-stock data is
effectively **CRSP-only**. Deepest free path before then is **index** series (§6).

**Deep-history anchors** (oldest continuously-listed blue chips — IBM, GE, KO, PG,
XOM): free daily ≈ 1962; Norgate daily to 1950; CRSP daily to 1926 / monthly to 1925.

---

## 2. Equity-Index Futures

| Root | Underlying | Native inception | Max-history (spliced to cash index) | Conf. |
|---|---|---|---|---|
| **ES** (E-mini S&P 500) | S&P 500 ($50) | **1997-09-09** | splice SP (1982) → S&P 500 cash **1957** (composite **1927**) | [H] |
| **SP** (full-size S&P 500) | S&P 500 ($250) | **1982-04-21** | S&P 500 cash 1957 | [H] |
| **NQ** (E-mini Nasdaq-100) | Nasdaq-100 ($20) | **1999-06-21** | splice ND (1996) → Nasdaq-100 cash **1985-01-31** | [H] |
| **ND** (full Nasdaq-100) | Nasdaq-100 ($100) | **1996-04** | Nasdaq-100 cash 1985 | [M] |
| **YM** (E-mini Dow) | DJIA ($5) | **2002-04** | splice → DJIA cash **1896-05-26** (mostly cash) | [M] |
| **RTY** (E-mini Russell 2000) | Russell 2000 ($50) | **2017-07-10** (CME relaunch) | predecessors: CME 1993 → ICE/TF 2008-09 → CME 2017; cash **1987** | [H] |
| **EMD** (E-mini S&P MidCap 400) | S&P 400 ($100) | **~2002** | full-size MD 1992 → S&P 400 cash **1991-06** | [L] |
| **MES / MNQ / M2K / MYM** (Micros) | resp. ($5/$2/$5/$0.50) | **2019-05-06** | splice onto ES/NQ/RTY/YM | [H] |

> Russell has **cross-exchange seams** (CME→ICE 2008, ICE→CME 2017) — bridge across
> exchanges, not just expiries. Splice to the **price** cash index (^GSPC/^NDX/^RUT/^DJI).

---

## 3. ETFs (47 liquid names; inception → splice index)

**Broad US equity**

| Ticker | Exposure | Inception | Splice index (earliest) | Conf. |
|---|---|---|---|---|
| SPY | S&P 500 | **1993-01-22** | S&P 500 TR (1988; price 1957) | [H] |
| IVV | S&P 500 | 2000-05-15 | S&P 500 TR | [H] |
| VOO | S&P 500 | 2010-09-07 | S&P 500 TR / VFINX (1976) | [H] |
| QQQ | Nasdaq-100 | **1999-03-10** | NDX / XNDX (1985) | [H] |
| IWM | Russell 2000 | 2000-05-22 | Russell 2000 TR (base 1978-12) | [H] |
| DIA | DJIA | 1998-01-14 | DJIA / DJITR (1896) | [H] |
| VTI | US total market | 2001-05-24 | CRSP US TM / VTSMX (1992) | [H] |
| MDY | S&P MidCap 400 | 1995-05-04 | S&P 400 (1991-06) | [H] |
| IWB / IWD / IWF | Russell 1000 / Value / Growth | 2000-05 | Russell 1000 (1978-12; style 1987) | [H] |

**Sector SPDRs** — XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLB all **1998-12-16** `[H]`;
**XLRE 2015-10-07** (carve-out from XLF) and **XLC 2018-06-18** (GICS realignment)
have **no pre-carve-out ETF price** — index back-history only.

**US Treasuries / rates**

| Ticker | Exposure | Inception | Splice index (earliest) | Conf. |
|---|---|---|---|---|
| SHY / IEF / TLT | 1-3y / 7-10y / 20y+ Treasury | **2002-07-22** | ICE/Bloomberg Treasury TR (Long Tsy TR 1973) | [H] |
| GOVT | broad Treasury | 2012-02-14 | Bloomberg US Treasury TR (1973) | [H] |
| BIL / SHV | 1-3mo / 0-1y T-bill | 2007 | Bloomberg T-Bill (1950s) | [H/M] |

**US credit / aggregate**

| Ticker | Exposure | Inception | Splice index (earliest) | Conf. |
|---|---|---|---|---|
| AGG | US Aggregate | 2003-09-22 | Bloomberg US Agg TR (**1976**) | [H] |
| BND | US Aggregate | 2007-04-03 | Bloomberg US Agg / VBMFX (1986) | [H] |
| LQD | IG corporate | 2002-07-22 | iBoxx IG (~2000) / Bloomberg US Corp TR (1973) | [H] |
| HYG / JNK | High yield | 2007 | Bloomberg US HY TR (**1983**) | [H] |
| TIP | TIPS | 2003-12-04 | Bloomberg US TIPS (~1997) | [H] |
| MBB | Agency MBS | ~2007-03 | Bloomberg US MBS (1976) | [M] |

**Commodities (ETF)**

| Ticker | Exposure | Inception | Splice (earliest) | Conf. |
|---|---|---|---|---|
| GLD / IAU | Gold (physical) | **2004-11-18** / 2005-01 | LBMA Gold PM fix (**1968**) — *spot* | [H] |
| SLV | Silver (physical) | 2006-04-21 | LBMA Silver (1968) — *spot* | [H] |
| USO | WTI crude (**futures**) | 2006-04-10 | WTI front-month **roll** (CL 1983) — NOT spot | [H] |
| UNG | Nat-gas (**futures**) | 2007-04-18 | Henry Hub roll (NG 1990) — NOT spot | [H] |
| DBC / PDBC | Broad (futures) | 2006-02 / 2014-11 | DBIQ Optimum Yield ER (~1989) | [H] |
| GSG | Broad (futures) | 2006-07-10 | S&P GSCI TR (back-cast **1970**) | [H] |
| DBA | Agriculture (futures) | 2007-01-05 | DBIQ Ag ER (~1989) | [H] |

> **Critical:** physical-metal ETFs (GLD/IAU/SLV) splice to **spot**; futures-backed
> ETFs (USO/UNG/DBC/GSG/DBA) splice to **rolled-futures indices, not spot** — they
> diverge massively under contango.

**International (US-listed, for completeness):** EFA (2001-08, MSCI EAFE **1969**),
EEM (2003-04, MSCI EM **1987**), VEA (2007-07), VWO (2005-03). VEA/VWO changed
MSCI→FTSE benchmarks in 2013 (methodology break).

---

## 4. US Treasuries & Fixed Income

**Layer 1 — Treasury & short-rate futures**

| Root | Underlying | Inception | Continuous source | Conf. |
|---|---|---|---|---|
| ZB | 30Y T-Bond | **1977-08-22** | Norgate/CSI/Pinnacle/Stevens | [H] |
| UB | Ultra T-Bond | 2010-01 | " | [H] |
| ZN | 10Y Note | **1982** | " | [H] |
| ZF | 5Y Note | 1988 | " | [M/H] |
| ZT | 2Y Note | 1990 | " | [H] |
| TN | Ultra 10Y | 2016-01 | " | [H] |
| GE | Eurodollar (3M) | **1981-12** → SOFR 2023 | " | [H] |
| SR3 | 3M SOFR | 2018-05-07 | " | [H] |
| ZQ | 30-Day Fed Funds | ~1988-10 | " | [L/M] |

> ZB's deliverable basket drifted shorter over time — "continuous ZB since 1977" is
> not a constant instrument. Splice ZB→UB, or use constant-maturity yields (Layer 2).

**Layer 2 — Yield / rate series (deep-history backbone; FRED). Yields, not returns.**

| Series | What | Earliest | Freq | Conf. |
|---|---|---|---|---|
| Shiller "Long Rate" | ≈10y govt yield | **1871-01** | Monthly | [H] |
| TB3MS | 3M T-bill | **1934-01** | Monthly | [H] |
| GS10 | 10Y CMT | **1953-04** | Monthly | [H] |
| DGS10 / DGS1 | 10Y / 1Y CMT | **1962-01-02** | Daily | [H] |
| DGS3 / DGS5 | 3Y / 5Y CMT | 1962 | Daily | [M] |
| DGS7 | 7Y CMT | 1969-07 | Daily | [M] |
| DGS2 | 2Y CMT | 1976-06 | Daily | [M] |
| DGS30 | 30Y CMT | 1977-02 (gap 2002–2006) | Daily | [M] |
| DGS3MO / DGS6MO | 3M / 6M CMT | ~1981–1982 | Daily | [L/M] |
| DGS20 | 20Y CMT | 1993-10 | Daily | [M] |
| DGS1MO | 1M CMT | 2001-07 | Daily | [M] |

**Layer 3 — Bond total-return history (the right object for return backtests)**

| Series | Source | Earliest | Conf. |
|---|---|---|---|
| Damodaran 10Y T-bond TR | NYU Stern `histretSP` | **1928** (annual) | [H] |
| Ibbotson/SBBI LT govt bond | Morningstar/CFA | **1926** (pre-1977 modeled) | [H] |
| Bloomberg US Aggregate TR | Bloomberg | **1976-01** | [H] |
| Bloomberg US Corp IG / HY TR | Bloomberg | 1973 / 1983 | [H] |

**Stitch for max history:** yields — Shiller 1871 → GS10 1953 → DGS10 1962 (daily);
short rate — TB3MS 1934 → DGS3MO 1981. Total return — Damodaran 1928 (annual) or
Ibbotson 1926; for tradable, bond ETFs (2002+) spliced onto Bloomberg TR indices.

---

## 5. Commodities

**Futures roots** (modern continuous series usually start later than pit inception):

| Root | Commodity | Modern inception | Deep splice (spot/annual) | Conf. |
|---|---|---|---|---|
| CL | WTI crude | **1983-03-30** | EIA WTI daily 1986; Macrotrends annual **1946** | [H] |
| BZ | Brent | 1988-06 (IPE) | — | [H] |
| NG | Henry Hub gas | **1990-04** | EIA spot 1997 | [H] |
| HO | Heating oil/ULSD | 1978 | — | [H/M] |
| RB | RBOB gasoline | 2005-10 | splice HU (1984) | [H] |
| GC | Gold | **1974-12-31** | LBMA PM fix **1968**; NMA annual **1833** | [H] |
| SI | Silver | ~1963-06 (COMEX) | LBMA silver; 1933 context | [M] |
| HG | Copper | ~1959 | Macrotrends 1959 | [M] |
| PL / PA | Platinum / Palladium | 1956 / 1968 | — | [L/M] |
| ZC / ZS / ZW | Corn / Soybeans / Wheat | pit 1877 / 1936 / 1877; continuous ~1959–1973 | — | [H pit / M cont.] |
| SB / KC / CC / CT | Sugar11 / Coffee / Cocoa / Cotton | pit 1914 / 1882 / 1925 / 1870; continuous ~1959–1973 | — | [H pit / M cont.] |
| LE / HE | Live cattle / Lean hogs | **1964** / 1966 (live)→1996 (lean) | — | [H] |

**Continuous-contract vendors:** **CSI** (deepest, ~1949–1959 for oldest grains),
**Pinnacle CLC** (~1960s–70s, deep+cheap), **Stevens/SCF** (Nasdaq Data Link),
**Norgate** (~1980+, retail-friendly), Barchart/FirstRate (shallower). Free Quandl
CHRIS is **discontinued** (legacy only).

**Broad commodity indices (deep splice anchors):**

| Index | Live | Back-calculated to | Conf. |
|---|---|---|---|
| CRB (Refinitiv/CoreCommodity) | 1957 | base 1947–49 | [H] |
| S&P GSCI | 1991 | back-tested **1970** | [H] |
| Bloomberg Commodity (BCOM) | 1991 | back-calc **1960** | [H] |

> GSCI/BCOM pre-~1991 values are explicitly **hypothetical/back-calculated** — flag
> in any backtest. CRB (1957) is the oldest *live* broad benchmark.

---

## 6. Deep-history index backbone (the splice anchors)

The longest series to back-fill any single name or build a deep benchmark:

| Series | Earliest | Free source | Use |
|---|---|---|---|
| Shiller S&P Composite (monthly, real TR) | **1871-01** | econ.yale.edu/~shiller (`ie_data.xls`) | Deepest equity index |
| Ken French market TR (daily) | **1926-07-01** | Dartmouth data library | Deepest free daily equity-market return |
| Dow Jones Industrial Avg | **1896-05-26** | S&P DJI; Yahoo `^DJI` | Price-only |
| S&P 500 (official 500-stock) | **1957-03-04** | S&P DJI; FRED `SP500`; `^GSPC` | Live index; TR daily from 1988 |
| Damodaran asset-class annual TR | **1928** | NYU Stern `histretSP` | Stocks/bonds/bills annual TR |
| Shiller long rate | **1871** | Shiller data | Yield backbone |
| Gold annual (NMA) | **1833** | nma.org | Longest gold |

---

## 7. Recommended data vendors by access tier

- **Free, offline-capable here:** synthetic + CSV (the system's backbone). Plus the
  free deep-history *files*: Shiller, Ken French, Damodaran, NMA gold, FRED series
  (downloadable CSV).
- **Free, network-gated (blocked in this env):** Stooq (bulk EOD), yfinance, FRED API.
- **Freemium + key:** Tiingo (best free real EOD; equities/ETFs to 1962), Alpha Vantage, Polygon.
- **Paid, systematic-grade:** **Norgate** (stocks 1950 + futures, survivorship-free,
  best value for backtesting), **CSI** (deepest futures), **Pinnacle** (deep futures,
  cheap), **Sharadar/Nasdaq Data Link** (bias-free equities 1998), **EODHD** (broad global).
- **Paid, institutional:** CRSP/WRDS (academic gold standard, 1926), Bloomberg, Refinitiv/LSEG.

---

## 8. Reachability from this build environment

The network allowlist (see `docs/data-sources.md`) blocks **every** market-data host —
only PyPI/PythonHosted/GitHub resolve. So in this environment the system runs on
**synthetic + committed CSV fixtures**; the catalog above is the *sourcing plan* for
when it runs where a network path (or paid vendor) exists. Several free deep-history
files (Shiller, Ken French, Damodaran, NMA) are static downloads that can be committed
as CSV fixtures to seed the universe offline.

---

## 9. Machine-readable catalog

The loadable version of this catalog (one row per instrument with start date, source,
splice chain, and confidence) is `docs/instrument_universe.csv`. Schema:

```
asset_class, symbol, description, instrument_type, native_start,
max_history_start, history_source, splice_chain, live_source, confidence, notes
```

It is intended as the seed for a `universe` definition the data layer can load
(`StrategySpec.universe`), and to be extended toward full S&P 500 / Russell 1000 PIT
membership when a constituents source is wired in.

---

## 10. Verification caveats

Publisher/exchange/SSRN/FRED pages largely **block automated fetch (403)**, so dates
were verified via concurring secondary indexes and provider/issuer pages where
reachable, with per-row `[H]/[M]/[L]` confidence. Items to confirm against primary
docs before relying on them: platinum/palladium inception (`[L/M]`), several FRED DGS
start dates and the DGS30 2002–2006 gap (`[M]`), EMD futures launch (`[L]`), Russell
1993 original CME date (`[M]`), and exact per-symbol continuous-series start dates
(CBOT grains / ICE softs start ~1959–1973 as continuous, far later than pit origins).
Full source URLs are in the five research-agent reports underlying this catalog.
