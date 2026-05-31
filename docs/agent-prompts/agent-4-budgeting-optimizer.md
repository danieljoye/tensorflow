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
3. `optimize/ccd.py` — **the DEFAULT fast solver**: Spinu log-barrier cyclical
   coordinate descent (BUILD_PLAN §12.1), pure NumPy. Per-coordinate update is the
   positive root `xᵢ ← (aux + √(aux² + 4Σᵢᵢbᵢ))/(2Σᵢᵢ)`, `aux = xᵢΣᵢᵢ − (Σx)ᵢ`, with
   rank-1 maintenance of `Σx`; stop on `maxᵢ|RCᵢ/ΣRC − bᵢ| < 1e-8`. Handles long-only +
   bounds + leverage (final rescale). Cross-check against `pyrb`/`riskparity.py` values.
4. `optimize/convex.py` — cvxpy log-barrier program for **non-separable** constraints
   (group caps, turnover, arbitrary `Cw≤d`) that CCD can't express: `min ½wᵀΣw −
   bᵀlog(w)` s.t. the linear constraints. `optimize/scipy_solver.py` — the same
   log-barrier via `scipy.optimize` with analytic gradient `Σx − b/x` and Hessian
   `Σ + diag(b/x²)`: the cross-check oracle + bounded separable path.
4b. `optimize/router.py` — pick the cheapest path the `Constraints` allow: bounds/
   long-only/leverage → `ccd.py`; group caps / turnover / general linear → `convex.py`;
   scipy as oracle. This is what the `Optimizer.solve` implementation calls.
5. `optimize/constraints.py` — `Constraints`: long-only, leverage/budget=target,
   group/sector caps, turnover limit (vs. a previous portfolio). Wire into both
   solvers where expressible.
6. `optimize/classical.py` — benchmark constructors following the EDHEC course
   formulations, each implementing `PortfolioConstructor`:
   - **`equal_weight`** — naive 1/N baseline.
   - **`gmv`** — global minimum variance (`min wᵀΣw` s.t. constraints); cov only.
     Keep the closed-form `σ_gmv = √(1/Σ pinv(Σ))` as a sanity check.
   - **`msr`** — max Sharpe / tangency. Use the **Cornuéjols–Tütüncü variable
     substitution** (BUILD_PLAN §12.2), NOT a naive fractional-Sharpe maximization:
     `min wᵀΣw` s.t. `(μ−r_f)ᵀw = 1`, `Σw = k`, `k ≥ 0`; recover `w = w*/k`. Guard
     `max(μ) > r_f`. Needs `mu` (from Agent 3) and a risk-free rate.
   - **`efficient_msr`** — MSR using Agent 3's **risk-based / volatility-proxy** `mu`
     instead of sample means (Martellini 2008 / Efficient Indexation). Cite in docstring.
   - **`efficient_frontier`** — trace min-variance weights across target returns
     (the `minimize_vol`/`optimal_weights` pattern) and return the frontier.
   These let the backtester compare ERC head-to-head against classical methods.
7. `budgeting/budget.py` (conditional) + `optimize/conditional.py` — **conditional /
   state-dependent risk budgets** (Martellini–Milhau–Tarelli, "Toward Conditional
   Risk Parity"): allow target contributions `bᵢ` to be a function of an observable
   state variable (e.g. a passed rate/valuation signal), reducing to ERC when the
   signal is flat. Keep the mapping pluggable (a callable `state -> RiskBudget`).
8. `optimize/ensemble.py` — **ensemble / "diversifying the diversifiers"**
   constructor (Amenc–Goltz–Lodh–Martellini): average the normalized weights of a
   set of passed `PortfolioConstructor`s (GMV, Efficient-MSR, ERC, and — if available
   — Agent 9's MDP/max-ENB), with an optional **tracking-error-control overlay** that
   shrinks the blend toward a reference (equal- or cap-weight) when ex-ante TE exceeds
   a target. Must degrade gracefully if a constituent constructor is absent.

## Interfaces to honor
`Optimizer.solve(cov, budget, constraints) -> Portfolio` for the risk-budget path,
and `PortfolioConstructor.construct(cov, *, mu, budget, constraints) -> Portfolio`
(from Agent 0.5) for the unified path that the classical constructors and the
ERC optimizer both expose, so the backtester can drive any method through one
interface.

## Deliverables
Both solvers + contribution math + constraints + classical benchmarks + conditional
budgets + ensemble, with tests: on a known 2–3 asset case the ERC solution has equal
TRCs within tol; convex and scipy solutions agree within tol; arbitrary (non-equal)
budgets are matched; constraints respected (e.g. turnover cap reduces trade size);
GMV beats random portfolios on variance; MSR matches the analytic tangency on a toy
2-asset case; the efficient frontier is monotone/convex; the ensemble weights equal
the mean of constituents and the TE overlay lowers ex-ante tracking error; conditional
budgets reduce to ERC on a flat signal. Use fixed synthetic cov + μ fixtures. Cite the
source paper (BUILD_PLAN §11) in each non-trivial module docstring.

## Out of scope
Covariance estimation (Agent 3), backtesting (Agent 5). Do not change `core/`.

## Done when
Both `Optimizer` implementations solve ERC and arbitrary budgets correctly, agree
with each other, honor constraints, and tests pass.
