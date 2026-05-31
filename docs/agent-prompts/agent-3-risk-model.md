# Agent 3 — Risk & Return Models

**Wave:** 1 (parallel). **Owns:** `riskbudget/riskmodel/*`.
**Depends on:** Agent 0 (`core/`) + Agent 0.5 (`MeanModel`, `ExpectedReturns`).

## Scope
Covariance estimation — the `Σ` that drives all risk budgeting — plus expected
returns for the classical optimizers (MSR / efficient frontier).
1. `sample.py` — sample covariance estimator (annualized `× frequency`).
2. `ewma.py` — exponentially-weighted covariance (`exp_cov`, configurable span≈180).
3. `semicov.py` — downside semicovariance: `drops = min(r−B, 0)`, `S = (dropsᵀdrops)/T · freq`.
4. `shrinkage.py` — Ledoit-Wolf with **both** `constant_variance` (sklearn) and
   `constant_correlation` (the target the literature recommends) targets, plus OAS
   (one-liner via `sklearn.covariance.oas`). Pure-NumPy port acceptable if avoiding sklearn.
5. `factor.py` — statistical (PCA) factor model: reconstruct covariance from the
   top-k factors plus idiosyncratic variance.
6. `psd.py` — **nearest-PSD fix, applied to the output of EVERY estimator** (BUILD_PLAN
   §12.3): Cholesky-based PSD test on `Σ + 1e-16·I`; spectral repair
   `Σ = V·diag(max(λ,0))·Vᵀ` from `eigh`. This is the single most important robustness
   step — anything that gets inverted or fed to a QP needs it. HIGH priority.
7. `returns_model.py` — expected-return estimators implementing `MeanModel`:
   historical mean (annualized; support geometric/CAGR `(1+r).prod()**(freq/N)−1` and
   arithmetic), EWMA mean, a CAPM-/market-implied option (β from a market-augmented
   covariance; equal-weight proxy if no benchmark), and a **risk-based / total-volatility
   proxy** (expected return ∝ total volatility or semi-deviation) per Martellini (2008)
   "Toward the Design of Better Equity Benchmarks" — avoids the unreliable sample mean
   and feeds Agent 4's **Efficient-MSR**. Return `ExpectedReturns` aligned to `returns.assets`.
8. `black_litterman.py` — **views-based posterior returns + covariance** (v1; formulas
   in BUILD_PLAN §12.7). Reverse-optimization prior `Π = δΣw_mkt`; market-implied
   `δ = (E[R_m]−r_f)/σ_m²`; views `(P, Q, Ω)` with He–Litterman default
   `Ω = diag(diag(τPΣPᵀ))` and an Idzorek confidence option; posterior
   `E(R) = Π + τΣPᵀ·solve(PτΣPᵀ+Ω, Q−PΠ)` **solved as a linear system, not by
   inverting** (lstsq fallback); posterior covariance per §12.7; implied
   `w = (δΣ)⁻¹E(R)`. Provide an `absolute_views` dict → `(P,Q)` helper. Define a small
   `Views`/`BLInputs` structure inside this module (market caps, P, Q, confidences,
   τ, δ, r_f) — do NOT modify `core/`. Output posterior `ExpectedReturns` (a `MeanModel`)
   + posterior covariance ndarray so it feeds Agent 4's MSR/EF/mean-variance paths.
9. *(Advanced / stretch — only if time permits)* `comoments.py` — structured /
   shrinkage estimators of co-skewness and co-kurtosis tensors (Martellini–Ziemann
   2010), shrinking the sample comoment toward a structured target. Cite the paper;
   keep it optional and clearly flagged. Full higher-moment optimization is roadmap.

The covariance estimators implement the `RiskModel` Protocol
(`estimate(returns) -> np.ndarray`, symmetric PSD — route every output through
`psd.py`); the return estimators implement `MeanModel`.

## Interfaces to honor
`RiskModel.estimate` and `MeanModel.estimate` from `core/interfaces.py`; covariance
ndarrays and `ExpectedReturns` both ordered to match `returns.assets`.

## Deliverables
Covariance estimators (sample, EWMA, semicov, shrinkage, factor) + the universal
PSD fix + the return estimators + tests: covariance outputs symmetric and PSD (the
PSD fix turns an indefinite input into a PSD output — test with a deliberately
non-PSD matrix); shrinkage lowers the condition number on ill-conditioned input;
PCA factor model reconstructs a low-rank-plus-noise covariance within tolerance;
expected-return estimators recover a known drift on synthetic data; Black-Litterman
with no views returns the prior `Π` (and a confident view shifts the posterior toward
`Q`), posterior is computed via a solve not an inverse; results cross-checked against
a numpy reference. Add an **optional skippable cross-check** (`pytest.importorskip("pypfopt")`)
of the shrinkage estimators and the Black-Litterman posterior against `pyportfolioopt`.
Use a local fixture / synthetic input.

## Out of scope
Optimization, risk-contribution math (Agent 4). Do not change `core/`.

## Done when
All estimators satisfy `RiskModel`, return symmetric PSD matrices, and tests pass.
