# Agent 0.5 — Core Extension

**Wave:** 0.5 (after Foundation, before Wave 1). **Owns:** additive changes to
`riskbudget/core/`. **Depends on:** Agent 0.

## Scope
Add the contracts the classical optimizers and dynamic-allocation layer need,
**without changing any existing Wave 0 signature** (purely additive). Read
BUILD_PLAN §5.1 first.
1. `core/types.py` — add `ExpectedReturns` (a μ vector over assets, annualized,
   with `.as_array(assets)` aligned like the other types). Add `AllocatorParams`
   (config for dynamic strategies: multiplier `m`, floor, drawdown limit, safe-rate).
2. `core/interfaces.py` — add `MeanModel` (`estimate(returns) -> ExpectedReturns`),
   `PortfolioConstructor` (`construct(cov, *, mu=None, budget=None, constraints) ->
   Portfolio`), and `Allocator` (`allocate(risky, safe, params) -> BacktestResult`)
   as `runtime_checkable` Protocols.
3. `core/__init__.py` — re-export the new names so `from riskbudget.core import …`
   works for them too.
4. Extend `tests/` to cover the new types' validation/alignment.

## Constraints
- Do NOT modify the behavior or signatures of existing `ReturnMatrix`, `PriceData`,
  `RiskBudget`, `Portfolio`, `BacktestResult`, `DataSource`, `RiskModel`,
  `Optimizer`, `Backtester`, `Constraints`, or `RebalanceSchedule`. Add only.
- Keep the ruff/mypy/pytest scoping established in Wave 0.

## Done when
New contracts importable from `riskbudget.core`, ruff/mypy/pytest all green, and no
existing test changed its meaning. Commit and push to the branch (no PR).
