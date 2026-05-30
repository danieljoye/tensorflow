# Agent 4 — Budgeting & Optimizer

**Wave:** 1 (parallel). **Owns:** `riskbudget/budgeting/*`, `riskbudget/optimize/*`.
**Depends on:** Agent 0 (`core/`), Agent 3's `RiskModel` interface (covariance input).

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

## Interfaces to honor
`Optimizer.solve(cov, budget, constraints) -> Portfolio` from `core/interfaces.py`.

## Deliverables
Both solvers + contribution math + constraints, with tests: on a known 2–3 asset
case the ERC solution has equal TRCs within tol; convex and scipy solutions agree
within tol; arbitrary (non-equal) budgets are matched; constraints are respected
(e.g. turnover cap reduces trade size). Use a fixed synthetic covariance fixture.

## Out of scope
Covariance estimation (Agent 3), backtesting (Agent 5). Do not change `core/`.

## Done when
Both `Optimizer` implementations solve ERC and arbitrary budgets correctly, agree
with each other, honor constraints, and tests pass.
