# Agent 9 — Diversification, Factor Risk Budgeting & Hierarchical

**Wave:** 1 (parallel). **Owns:** `riskbudget/diversification/*`, `riskbudget/clustered/hrp.py`.
**Depends on:** Agent 0 (`core/`), Agent 0.5 (`PortfolioConstructor`).

## Scope
Implement diversification *measurement* and the *factor* sense of risk budgeting
(BUILD_PLAN §2, §11). This is the Deguest–Martellini–Meucci / Choueifaty line.
1. `diversification/metrics.py` (formulas in BUILD_PLAN §12.5)
   - **Diversification Ratio** `DR(w) = (wᵀσ)/√(wᵀΣw)` (Choueifaty–Coignard).
   - **Effective Number of Bets** `ENB = exp(−Σ pᵢ ln pᵢ)` with diversification
     distribution `p = (tᵀ)⁻¹b ⊙ (t·Σ·b)/(bᵀΣb)` (`t` = torsion matrix; `Σpᵢ=1`).
     **Replicate the entropy guard verbatim:** use `pᵢ ln(1 + (pᵢ−1)·[pᵢ>1e-5])` to
     avoid `ln(0)`. Support both PCA-factor and min-torsion bases — Meucci.
2. `diversification/torsion.py` — the **minimum-torsion transform** (BUILD_PLAN §12.4):
   the polar fixed-point iteration on `c = sqrtm(C)` (the **symmetric/eigendecomp**
   root of the correlation matrix — NOT Cholesky). Init `d=1`; iterate `U=diag(d)·C·diag(d)`,
   `u=sqrtm(U)`, `q=u⁻¹·diag(d)·c`, `d=diag(q·c)`, `π=diag(d)·q`; stop on
   `|Δ‖c−π‖_F|/‖c−π‖_F/n ≤ 1e-8`; return `t = diag(s)·(π·c⁻¹)·diag(1/s)`. Also expose
   the PCA torsion and the one-shot approximate `t = diag(s)·C^{-1/2}·diag(1/s)`.
   Cross-check against `reckziegel/uncorbets` test vectors — Meucci–Santangelo–Deguest.
3. `diversification/constructors.py` — `PortfolioConstructor`s:
   - **Most Diversified Portfolio (MDP):** maximize `DR(w)` s.t. constraints.
   - **Max-ENB:** maximize the effective number of bets (minimize `−ENB` via
     `scipy.optimize.minimize(method="SLSQP")` under `Σw=1, 0≤w≤1`); also support an
     `ENB(w) ≥ k` constraint wrapper around a passed objective.
   - **Factor-risk-budget portfolio:** weights whose *factor* risk contributions
     (on the min-torsion basis) match a target budget — the factor analogue of the
     Agent 4 asset-level ERC.
4. `clustered/hrp.py` — **Hierarchical Risk Parity** (López de Prado 2016; BUILD_PLAN
   §12.6), a `PortfolioConstructor`: correlation distance `Dᵢⱼ=√(½(1−ρᵢⱼ))` →
   `scipy.cluster.hierarchy.linkage` (default `ward` or `single`) → `leaves_list`
   quasi-diagonalization → recursive bisection splitting capital inversely to cluster
   risk, inverse-variance weights within clusters. No matrix inversion (works on
   singular Σ). v1 = plain HRP, variance risk measure; NCO/HERC and alternative risk
   measures are roadmap (BUILD_PLAN §10).

## Interfaces to honor
`PortfolioConstructor.construct(cov, *, mu, budget, constraints) -> Portfolio` for
the constructors; pure functions for the metrics (take `weights` + `cov`). Cite the
source paper (BUILD_PLAN §11) in each module docstring.

## Deliverables
Metrics + min-torsion + constructors + HRP, with tests: ENB of equal-weight over `k`
independent factors ≈ `k`; ENB ≤ N always; min-torsion factors are uncorrelated
(off-diagonal correlation ≈ 0); MDP beats random portfolios on `DR`; max-ENB ≥ ENB
of GMV/equal-weight on a correlated fixture; HRP weights are positive, sum to 1, and
require no matrix inversion (works on a singular Σ). Use fixed synthetic-cov fixtures.

## Out of scope
Asset-level ERC and classical optimizers (Agent 4 — but your constructors and theirs
share the `PortfolioConstructor` interface so the ensemble/backtester can mix them),
analytics wiring (Agent 6 imports your `metrics.py`). Do not change `core/`.

## Done when
Constructors satisfy `PortfolioConstructor`, metrics are correct on known fixtures,
min-torsion factors are verifiably uncorrelated, and tests pass offline.
