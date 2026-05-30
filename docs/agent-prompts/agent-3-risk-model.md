# Agent 3 — Risk Model

**Wave:** 1 (parallel). **Owns:** `riskbudget/riskmodel/*`.
**Depends on:** Agent 0 (`core/`).

## Scope
Covariance estimation — the `Σ` that drives all risk budgeting.
1. `sample.py` — sample covariance estimator.
2. `ewma.py` — exponentially-weighted covariance (configurable halflife/lambda).
3. `shrinkage.py` — Ledoit-Wolf shrinkage toward a structured target (constant
   correlation or scaled identity); reuse scikit-learn or implement directly.
4. `factor.py` — statistical (PCA) factor model: reconstruct covariance from the
   top-k factors plus idiosyncratic variance.

All estimators implement the `RiskModel` Protocol:
`estimate(returns: ReturnMatrix) -> np.ndarray` returning a symmetric PSD matrix.

## Interfaces to honor
`RiskModel.estimate` from `core/interfaces.py`; input `ReturnMatrix`, output a
covariance ndarray ordered to match `returns.assets`.

## Deliverables
Four estimators + tests: outputs symmetric and PSD; shrinkage lowers the condition
number on ill-conditioned input; PCA factor model reconstructs a low-rank-plus-noise
covariance within tolerance; results cross-checked against a numpy reference. Use
Agent 2's synthetic source (or a local fixture) for inputs.

## Out of scope
Optimization, risk-contribution math (Agent 4). Do not change `core/`.

## Done when
All estimators satisfy `RiskModel`, return symmetric PSD matrices, and tests pass.
