# Agent 3 — Risk & Return Models

**Wave:** 1 (parallel). **Owns:** `riskbudget/riskmodel/*`.
**Depends on:** Agent 0 (`core/`) + Agent 0.5 (`MeanModel`, `ExpectedReturns`).

## Scope
Covariance estimation — the `Σ` that drives all risk budgeting — plus expected
returns for the classical optimizers (MSR / efficient frontier).
1. `sample.py` — sample covariance estimator.
2. `ewma.py` — exponentially-weighted covariance (configurable halflife/lambda).
3. `shrinkage.py` — Ledoit-Wolf shrinkage toward a structured target (constant
   correlation or scaled identity); reuse scikit-learn or implement directly.
4. `factor.py` — statistical (PCA) factor model: reconstruct covariance from the
   top-k factors plus idiosyncratic variance.
5. `returns_model.py` — expected-return estimators implementing `MeanModel`:
   historical mean (annualized), EWMA mean, and a CAPM-/market-implied option.
   Return `ExpectedReturns` aligned to `returns.assets`. These feed Agent 4's MSR
   and efficient-frontier benchmarks.

The four covariance estimators implement the `RiskModel` Protocol
(`estimate(returns) -> np.ndarray`, symmetric PSD); the return estimators
implement `MeanModel`.

## Interfaces to honor
`RiskModel.estimate` and `MeanModel.estimate` from `core/interfaces.py`; covariance
ndarrays and `ExpectedReturns` both ordered to match `returns.assets`.

## Deliverables
Four covariance estimators + the return estimators + tests: covariance outputs
symmetric and PSD; shrinkage lowers the condition number on ill-conditioned input;
PCA factor model reconstructs a low-rank-plus-noise covariance within tolerance;
expected-return estimators recover a known drift on synthetic data; results
cross-checked against a numpy reference. Use a local fixture / synthetic input.

## Out of scope
Optimization, risk-contribution math (Agent 4). Do not change `core/`.

## Done when
All estimators satisfy `RiskModel`, return symmetric PSD matrices, and tests pass.
