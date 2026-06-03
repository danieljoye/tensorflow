# Shiller source files (authoritative)

Robert J. Shiller's stock-market data, downloaded from his site
(https://shillerdata.com/ ; http://www.econ.yale.edu/~shiller/data.htm) and
uploaded into this repo because those hosts are outside this environment's
network allowlist.

- `ie_data.xls` — monthly S&P Composite, 1871-01 → 2026-05: price (P), dividend,
  earnings, CPI, 10y rate (GS10), real price, **real total-return price**, CAPE.
  This is the source for `riskbudget/data/providers/fixtures/shiller/sp500.csv`
  (regenerated into the datahub-compatible schema the Shiller provider parses).
- `chapt26.xlsx` — the annual long-run dataset (1871+) from *Market Volatility*
  (1989) / *Irrational Exuberance*, incl. consumption and present-value calcs.

## Validation (this data vs. what we had been using)
- GitHub datahub mirror vs. this `ie_data.xls`: median deviation **0.0000%**,
  mean 0.0026% across 1865 months — the mirror is a faithful copy.
- Our daily SPX (monthly average) vs. `ie_data.xls` P: median **0.019%**,
  return correlation **0.998** (1968-2023).
