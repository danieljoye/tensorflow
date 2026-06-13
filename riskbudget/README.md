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

## Quick start

Everything is driven by one config object (`StrategySpec`) through one registry:

```python
import riskbudget as rb

spec = rb.StrategySpec(
    name="ERC",
    assets=["STOCKS_TR", "BONDS", "GOLD"],
    data_source={"name": "daily_panel"},      # live 1968->present daily panel
    method="erc",                              # equal risk contribution
    risk_model="ledoit_wolf",
    schedule={"frequency": "monthly", "lookback": 252},  # 252 daily returns
    periods_per_year=252,
    target_volatility=0.10,                    # vol-target the book to 10% ann.
)
result = rb.backtest(spec)                     # walk-forward, no look-ahead
table = rb.compare([spec, spec.model_copy(update={"name": "EW", "method": "equal_weight"})])

from riskbudget.reporting import build_comparison_report  # self-contained HTML
```

- **Methods:** `erc`, `risk_budget`, `gmv`, `msr`, `efficient_msr`, `equal_weight`,
  `mdp`, `max_enb`, `factor_risk_budget`, `hrp`, `ensemble` (+ dynamic allocators
  `cppi`, `fund_separation`, `glidepath`, `floor`, `drawdown` via `riskbudget.dynamic`).
- **Risk models:** `sample`, `ewma`, `semicov`, `ledoit_wolf`, `oas`, `pca`;
  **mean models** incl. `black_litterman`.
- **Data sources:** `synthetic`, `csv`, `tiingo`, `shiller` (1871+, monthly),
  `gold`, `daily_panel` (daily 1968→present; live-refreshing GitHub transport,
  committed fixture for offline CI). See `docs/daily-data-sources.md`.
- **Surfaces:** library, FastAPI (`uvicorn --factory riskbudget.api.app:create_app`),
  Streamlit (`streamlit run riskbudget/dashboard/app.py`), CLI
  (`python -m riskbudget.cli ...`). Runnable demos in `examples/`.

## Layout

- `riskbudget.core` — shared, frozen contracts: data types (`PriceData`,
  `ReturnMatrix`, `RiskBudget`, `Portfolio`, `BacktestResult`), pipeline
  protocols (`DataSource`, `RiskModel`, `Optimizer`, `Backtester`), the
  `Constraints` / `RebalanceSchedule` config types, and the error hierarchy.

Downstream packages (`data`, `riskmodel`, `budgeting`, `optimize`,
`diversification`, `clustered`, `backtest`, `analytics`, `reporting`, `dynamic`,
`api`, `dashboard`, `cli`) build against `riskbudget.core`; `spec`/`registry`/
`run`/`compare` are the glue.

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
