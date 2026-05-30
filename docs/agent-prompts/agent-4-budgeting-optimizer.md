# Agent 4 — Budgeting & Optimizer

**Wave:** 1 (parallel). **Owns:** `riskbudget/budgeting/*`, `riskbudget/optimize/*`.
**Depends on:** Agent 0 (`core/`), Agent 0.5 (`PortfolioConstructor`,
`ExpectedReturns`), Agent 3's `RiskModel`/`MeanModel` interfaces (cov + μ input).

## Scope
The heart of the system: turn a covariance + risk budget into weights.
1. `budgeting/contributions.py` — risk decomposition: `MRCᵢ = (Σw)ᵢ/σ`,
   `TRCᵢ = wᵢ·MRCᵢ`, percentage contributions; verify `Σ TRCᵢ = σ`.
2. `budgeting/budget.py` — `RiskBudget` helpers: per-asset budgets, per-group
   budgets expanded to assets, and the ERC default (`bᵢ = 1/N`). (The dataclass
   itself lives in `core`; this adds construction/validation utilities.)
3. `optimize/convex.py` — the primary solver: cvxpy implementation of the
   log-barrier formulation `min ½wᵀΣw − Σ bᵢ ln(wᵢ)` s.t. `w ≥ 0`, then rescale
   to target leverage. Returns a `Portfolio`.
4. `optimize/scipy_solver.py` — SLSQP fallback minimizing the sum of squared
   risk-contribution deviations from budget; used as cross-check and when cvxpy
   is unavailable.
5. `optimize/constraints.py` — `Constraints`: long-only, leverage/budget=target,
   group/sector caps, turnover limit (vs. a previous portfolio). Wire into both
   solvers where expressible.
6. `optimize/classical.py` — benchmark constructors following the EDHEC course
   formulations, each implementing `PortfolioConstructor`:
   - **`equal_weight`** — naive 1/N baseline.
   - **`gmv`** — global minimum variance (`min wᵀΣw` s.t. constraints); cov only.
   - **`msr`** — max Sharpe / tangency (`max (wᵀμ − r_f)/σ`); needs `mu`
     (`ExpectedReturns` from Agent 3) and a risk-free rate.
   - **`efficient_frontier`** — trace min-variance weights across target returns
     (the `minimize_vol`/`optimal_weights` pattern) and return the frontier.
   These let the backtester compare ERC head-to-head against classical methods.

## Interfaces to honor
`Optimizer.solve(cov, budget, constraints) -> Portfolio` for the risk-budget path,
and `PortfolioConstructor.construct(cov, *, mu, budget, constraints) -> Portfolio`
(from Agent 0.5) for the unified path that the classical constructors and the
ERC optimizer both expose, so the backtester can drive any method through one
interface.

## Deliverables
Both solvers + contribution math + constraints + classical benchmarks, with tests:
on a known 2–3 asset case the ERC solution has equal TRCs within tol; convex and
scipy solutions agree within tol; arbitrary (non-equal) budgets are matched;
constraints respected (e.g. turnover cap reduces trade size); GMV beats random
portfolios on variance; MSR matches the analytic tangency on a toy 2-asset case;
the efficient frontier is monotone/convex. Use fixed synthetic cov + μ fixtures.

## Out of scope
Covariance estimation (Agent 3), backtesting (Agent 5). Do not change `core/`.

## Done when
Both `Optimizer` implementations solve ERC and arbitrary budgets correctly, agree
with each other, honor constraints, and tests pass.
