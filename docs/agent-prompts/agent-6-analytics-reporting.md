# Agent 6 — Analytics & Reporting

**Wave:** 2. **Owns:** `riskbudget/analytics/*`, `riskbudget/reporting/*`.
**Depends on:** Agent 0 (`core/`), Agent 5's `BacktestResult` interface, and Agent
9's `diversification/metrics.py` (import ENB / Diversification Ratio — don't reimplement).

## Scope
Turn a `BacktestResult` (and point-in-time portfolios) into numbers and reports.
Follow the EDHEC course `edhec_risk_kit` formulations for the risk metrics.
1. `analytics/performance.py` — annualized return (`compound`-based), annualized
   volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate, and average
   turnover. Configurable periods/year.
2. `analytics/downside.py` — `drawdown()` (wealth index, running peak, drawdown
   series), `semideviation()` (vol of negative returns), and VaR/CVaR:
   `var_historic` (empirical percentile), `var_gaussian` with the **Cornish-Fisher**
   skew/kurtosis adjustment, and `cvar_historic` (expected shortfall). Confidence
   level configurable (default 5%).
3. `analytics/distribution.py` — `skewness()`, `kurtosis()`, and `is_normal()`
   (Jarque-Bera test).
4. `analytics/attribution.py` — risk-contribution attribution through time (each
   asset/group's TRC at each rebalance, verifying contributions sum to total risk;
   reuse Agent 4's `contributions.py`); realized vs. target budget drift; **ENB and
   Diversification Ratio through time** (import from Agent 9's `diversification/`);
   optional factor-exposure attribution if a factor model is supplied.
5. `analytics/summary.py` — `summary_stats()`: a canonical one-row-per-strategy
   table (annualized return, vol, Sharpe, max drawdown, skew, kurtosis,
   Cornish-Fisher VaR(5%), historic CVaR(5%)) for comparing strategies side by side.
6. `reporting/report.py` — assemble a report object/HTML with the summary-stats
   table plus Plotly charts: equity curve, drawdown, weights-through-time, and a
   risk-contribution bar/stacked-area chart. Pure rendering — no recomputation of
   strategy logic.

## Interfaces to honor
Consume `BacktestResult` and `Portfolio` from `core/types.py`; reuse Agent 4's
`contributions.py` for risk decomposition rather than reimplementing it.

## Deliverables
Metrics + downside risk + distribution + attribution + summary table + report
generator, with tests: metrics match hand-computed references on a toy return
series; Cornish-Fisher VaR ≥ Gaussian VaR on a left-skewed/fat-tailed sample;
attribution sums to total portfolio risk; `summary_stats` returns the expected
columns; report renders without error from a synthetic `BacktestResult`.

## Out of scope
Backtest mechanics (Agent 5), API/dashboard wiring (Agent 7). Do not change `core/`.

## Done when
Metrics are correct against references, downside-risk estimators agree with the
EDHEC formulations, attribution reconciles to total risk, `summary_stats` and the
report render, and tests pass.
