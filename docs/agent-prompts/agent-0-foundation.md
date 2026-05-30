# Agent 0 — Foundation

**Wave:** 0 (runs first, alone). **Owns:** `riskbudget/core/`, `pyproject.toml`,
CI config, the test harness. **Depends on:** nothing.

## Scope
Lay the foundation every other agent builds on:
1. `pyproject.toml` — package metadata, deps (numpy, pandas, scipy, cvxpy,
   scikit-learn, fastapi, uvicorn, pydantic v2, streamlit, plotly) and dev deps
   (pytest, pytest-cov, ruff, mypy). Configure ruff + mypy + pytest sections.
2. `riskbudget/core/types.py` — implement the shared data types from
   BUILD_PLAN §5: `ReturnMatrix`, `PriceData`, `RiskBudget`, `Portfolio`,
   `BacktestResult`. These are real, working types (not stubs) with validation.
3. `riskbudget/core/interfaces.py` — `DataSource`, `RiskModel`, `Optimizer`,
   `Backtester` Protocols plus `Constraints` and `RebalanceSchedule` types.
4. `riskbudget/core/errors.py` — shared exception hierarchy.
5. Package `__init__.py` files; ensure `pip install -e .` works.
6. CI workflow that runs ruff + mypy + pytest. Test harness with shared fixtures
   (e.g. a small deterministic synthetic covariance + return matrix fixture).

## Interfaces to honor
You **define** the contracts in BUILD_PLAN §5 — make the signatures real and
typed. `RiskBudget.equal(assets)`, `Portfolio.risk_contributions(cov)`, and
`PriceData.to_returns()` must be fully implemented since downstream agents rely
on them immediately.

## Deliverables
Working `core/` package, installable project, green CI, and a `tests/conftest.py`
with reusable fixtures. Tests for the `core/` types' math (risk contributions on a
known covariance sum to portfolio vol).

## Out of scope
Do not implement data sources, risk models, optimizers, backtest, analytics, API,
or dashboard. Just the contracts and scaffolding.

## Done when
`pip install -e .` succeeds, `pytest` passes, ruff/mypy clean, and every interface
in §5 is importable from `riskbudget.core`.
