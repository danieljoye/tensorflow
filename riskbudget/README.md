# riskbudget

Construct portfolios by **risk budget** rather than capital weight.

Given a universe of assets and a target allocation of *risk* (how much each asset
or group should contribute to total portfolio risk), `riskbudget`:

1. Estimates a **risk model** (covariance of asset returns).
2. Solves for asset **weights** whose risk contributions match the target budgets
   (the convex log-barrier / ERC formulation).
3. **Backtests** the resulting strategy through time with rebalancing and costs.
4. Reports **performance and risk analytics**, including risk-contribution
   attribution.
5. Exposes the above through a **REST API** and an interactive **dashboard**.

## Layout

- `riskbudget.core` — shared, frozen contracts: data types (`PriceData`,
  `ReturnMatrix`, `RiskBudget`, `Portfolio`, `BacktestResult`), pipeline
  protocols (`DataSource`, `RiskModel`, `Optimizer`, `Backtester`), the
  `Constraints` / `RebalanceSchedule` config types, and the error hierarchy.

Downstream packages (`data`, `riskmodel`, `budgeting`, `optimize`, `backtest`,
`analytics`, `api`, `dashboard`) build against `riskbudget.core` only.

## Install (development)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Quality gate

```bash
ruff check .
ruff format --check .
mypy
pytest
```

See `docs/BUILD_PLAN.md` for the full specification.
