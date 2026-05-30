# Agent 9 — Diversification & Factor Risk Budgeting

**Wave:** 1 (parallel). **Owns:** `riskbudget/diversification/*`.
**Depends on:** Agent 0 (`core/`), Agent 0.5 (`PortfolioConstructor`).

## Scope
Implement diversification *measurement* and the *factor* sense of risk budgeting
(BUILD_PLAN §2, §11). This is the Deguest–Martellini–Meucci / Choueifaty line.
1. `diversification/metrics.py`
   - **Diversification Ratio** `DR(w) = (wᵀσ)/√(wᵀΣw)` (Choueifaty–Coignard).
   - **Effective Number of Bets** `ENB = exp(−Σ pᵢ ln pᵢ)`, where `pᵢ` are the
     normalized risk contributions of the portfolio's exposure to uncorrelated
     factors (PCA of `Σ` by default) — Meucci, "Managing Diversification."
2. `diversification/torsion.py` — the **minimum-torsion transform**: from `Σ`,
   produce uncorrelated factors as close as possible (min tracking error) to the
   original assets (closed-form iterative solution on the correlation matrix).
   Expose factor risk contributions and an ENB computed on the min-torsion basis
   (more stable than raw PCA) — Meucci–Santangelo–Deguest.
3. `diversification/constructors.py` — `PortfolioConstructor`s:
   - **Most Diversified Portfolio (MDP):** maximize `DR(w)` s.t. constraints.
   - **Max-ENB:** maximize the effective number of bets; also support an
     `ENB(w) ≥ k` constraint wrapper around a passed objective.
   - **Factor-risk-budget portfolio:** weights whose *factor* risk contributions
     (on the min-torsion basis) match a target budget — the factor analogue of the
     Agent 4 asset-level ERC.

## Interfaces to honor
`PortfolioConstructor.construct(cov, *, mu, budget, constraints) -> Portfolio` for
the constructors; pure functions for the metrics (take `weights` + `cov`). Cite the
source paper (BUILD_PLAN §11) in each module docstring.

## Deliverables
Metrics + min-torsion + constructors, with tests: ENB of equal-weight over `k`
independent factors ≈ `k`; ENB ≤ N always; min-torsion factors are uncorrelated
(off-diagonal correlation ≈ 0); MDP beats random portfolios on `DR`; max-ENB ≥ ENB
of GMV/equal-weight on a correlated fixture. Use fixed synthetic-cov fixtures.

## Out of scope
Asset-level ERC and classical optimizers (Agent 4 — but your constructors and theirs
share the `PortfolioConstructor` interface so the ensemble/backtester can mix them),
analytics wiring (Agent 6 imports your `metrics.py`). Do not change `core/`.

## Done when
Constructors satisfy `PortfolioConstructor`, metrics are correct on known fixtures,
min-torsion factors are verifiably uncorrelated, and tests pass offline.
