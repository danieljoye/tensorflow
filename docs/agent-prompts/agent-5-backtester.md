# Agent 5 — Backtester

**Wave:** 2. **Owns:** `riskbudget/backtest/*`.
**Depends on:** Agent 0 (`core/`) + the interfaces of Agents 2, 3, 4.

## Scope
Walk-forward simulation of a risk-budgeted strategy.
1. `backtest/engine.py` — `WalkForwardBacktester` implementing `Backtester.run`:
   step through a `RebalanceSchedule`; at each rebalance estimate the covariance
   from a **trailing lookback window only** (no look-ahead), solve weights via the
   `Optimizer`, hold to the next rebalance, and accrue returns. Produce a
   `BacktestResult` (equity curve, weights-through-time, per-rebalance diagnostics).
2. `backtest/costs.py` — transaction cost model (proportional bps on turnover);
   applied at each rebalance.

Must be deterministic given a seed/fixture and must compose any `RiskModel` +
constructor passed in (don't hard-code estimator or solver). Accept any
`PortfolioConstructor` (ERC, GMV, MSR, equal-weight from Agent 4) — not just the
risk-budget `Optimizer` — so strategies can be compared head-to-head on the same
data/schedule. MSR/EF runs also take a `MeanModel` for expected returns.

## Interfaces to honor
`Backtester.run(data, model, optimizer, budget, schedule) -> BacktestResult`.

## Deliverables
Engine + cost model + tests: reproducible equity curve on synthetic data;
explicit no-look-ahead test (estimation window never includes the forward period);
turnover and costs reduce returns as expected; weights at each step satisfy the
budget within solver tolerance.

## Out of scope
Metrics/attribution (Agent 6), data/risk/optimizer internals. Do not change `core/`.

## Done when
`WalkForwardBacktester` satisfies `Backtester`, has no look-ahead, applies costs,
and tests pass on synthetic data.
