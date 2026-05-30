# Agent 6 — Analytics & Reporting

**Wave:** 2. **Owns:** `riskbudget/analytics/*`, `riskbudget/reporting/*`.
**Depends on:** Agent 0 (`core/`) + Agent 5's `BacktestResult` interface.

## Scope
Turn a `BacktestResult` (and point-in-time portfolios) into numbers and reports.
1. `analytics/performance.py` — annualized return, volatility, Sharpe, Sortino,
   max drawdown, Calmar, hit rate, and average turnover. Configurable periods/year.
2. `analytics/attribution.py` — risk-contribution attribution through time (each
   asset/group's TRC at each rebalance, verifying contributions sum to total risk);
   realized vs. target budget drift; optional factor-exposure attribution if a
   factor model is supplied.
3. `reporting/report.py` — assemble a report object/HTML with summary tables plus
   Plotly charts: equity curve, drawdown, weights-through-time, and a
   risk-contribution bar/stacked-area chart. Pure rendering — no recomputation of
   strategy logic.

## Interfaces to honor
Consume `BacktestResult` and `Portfolio` from `core/types.py`; reuse Agent 4's
`contributions.py` for risk decomposition rather than reimplementing it.

## Deliverables
Metrics + attribution + report generator, with tests: metrics match hand-computed
references on a toy return series; attribution sums to total portfolio risk; report
renders without error from a synthetic `BacktestResult`.

## Out of scope
Backtest mechanics (Agent 5), API/dashboard wiring (Agent 7). Do not change `core/`.

## Done when
Metrics are correct against references, attribution reconciles to total risk, the
report renders, and tests pass.
