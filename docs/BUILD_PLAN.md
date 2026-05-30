# Portfolio Risk Budgeting System — Build Plan

> Status: **DRAFT for review.** No code written yet. This document is the single
> source of truth that every build agent reads before starting.

## 1. Purpose

Build a Python system that constructs portfolios by **risk budget** rather than by
capital weight. Given a universe of assets and a target allocation of *risk*
(how much each asset/group should contribute to total portfolio risk), the system:

1. Estimates a **risk model** (covariance of asset returns).
2. Solves for asset **weights** whose risk contributions match the target budgets.
3. **Backtests** the resulting strategy through time with rebalancing and costs.
4. Reports **performance and risk analytics**, including risk-contribution attribution.
5. Exposes all of the above through a **REST API** and an interactive **dashboard**.

## 2. Domain background (shared vocabulary)

For a weight vector `w` and covariance matrix `Σ`:

- **Portfolio volatility:** `σ(w) = sqrt(wᵀ Σ w)`
- **Marginal risk contribution:** `MRCᵢ = (Σ w)ᵢ / σ(w)`
- **Total risk contribution:** `TRCᵢ = wᵢ · MRCᵢ`, and `Σᵢ TRCᵢ = σ(w)`
- **Risk budget:** target fraction `bᵢ` with `Σ bᵢ = 1`; we want `TRCᵢ / σ(w) = bᵢ`.
- **Equal Risk Contribution (ERC / "risk parity"):** the special case `bᵢ = 1/N`.

**Solver of record.** Use the convex formulation (Maillard–Roncalli–Teïletche /
Spinu): minimize `½ wᵀ Σ w − Σᵢ bᵢ · ln(wᵢ)` subject to `w ≥ 0`, then rescale so
weights sum to the target leverage. This is convex, has a unique solution, and is
far more robust than Newton root-finding on the risk-contribution residuals.
`cvxpy` is the primary backend; a `scipy.optimize` SLSQP path is the dependency-light
fallback and a cross-check in tests.

## 3. Technology choices

- **Language:** Python 3.11+
- **Core numerics:** numpy, pandas, scipy
- **Optimization:** cvxpy (primary) + scipy (fallback/cross-check)
- **Covariance shrinkage:** scikit-learn (`LedoitWolf`) or a self-contained impl
- **API:** FastAPI + uvicorn + pydantic v2
- **Dashboard:** Streamlit + Plotly
- **Tooling:** pytest (+pytest-cov), ruff (lint+format), mypy (typing)
- **Packaging:** pyproject.toml, src-less layout rooted at `riskbudget/`

## 4. Repository layout

```
riskbudget/
  core/          # shared types & interfaces (Wave 0 owns this — the contract)
    types.py         # PriceData, ReturnMatrix, RiskBudget, Portfolio, BacktestResult
    interfaces.py    # DataSource, RiskModel, Optimizer, Backtester (ABCs/Protocols)
    errors.py
  data/          # DataSource implementations
    synthetic.py     # correlated-returns generator (always-offline backbone)
    csvsource.py     # CSV/parquet loader
    providers/       # real provider adapters (added after data-research agent)
  riskmodel/     # covariance estimators
    sample.py        # sample covariance
    ewma.py          # exponentially-weighted
    shrinkage.py     # Ledoit-Wolf
    factor.py        # statistical (PCA) factor model
  budgeting/     # risk-contribution math + budget specification
    contributions.py # MRC/TRC, risk decomposition
    budget.py        # RiskBudget spec (per-asset, per-group, ERC default)
  optimize/      # solvers + constraints
    convex.py        # cvxpy log-barrier formulation
    scipy_solver.py  # SLSQP fallback
    constraints.py   # long-only, leverage, group caps, turnover
  backtest/      # walk-forward engine
    engine.py        # rebalancing schedule, lookback windows, costs
    costs.py
  analytics/     # metrics & attribution
    performance.py   # return, vol, Sharpe, Sortino, max drawdown, turnover
    attribution.py   # risk-contribution time series, factor exposures
  reporting/     # tabular + chart report generation
    report.py
  api/           # FastAPI service
    app.py
    schemas.py
  dashboard/     # Streamlit UI
    app.py
  cli/           # command-line entry points
    main.py
tests/           # mirrors package layout; pytest
docs/
  BUILD_PLAN.md          # this file
  data-sources.md        # produced by the data-research agent
  agent-prompts/         # per-agent prompt pack
pyproject.toml
```

## 5. Core interface contracts (Wave 0 deliverable)

Every downstream agent codes against these signatures. Wave 0 ships them as
typed stubs with docstrings and `NotImplementedError` bodies so other agents can
import and test against them immediately.

```python
# core/types.py  (dataclasses / pydantic models)
ReturnMatrix    # wraps a pandas DataFrame: index=dates, columns=asset ids; .values, .assets, .dates
PriceData       # raw prices -> .to_returns(method="log"|"simple") -> ReturnMatrix
RiskBudget      # mapping asset_id -> budget (floats summing to 1); .equal(assets) classmethod
Portfolio       # weights: dict[asset_id, float]; .leverage, .risk_contributions(cov)
BacktestResult  # equity curve, weights-through-time, per-rebalance diagnostics, metrics

# core/interfaces.py  (Protocols / ABCs)
class DataSource(Protocol):
    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData: ...

class RiskModel(Protocol):
    def estimate(self, returns: ReturnMatrix) -> np.ndarray:  # covariance matrix
        ...

class Optimizer(Protocol):
    def solve(self, cov: np.ndarray, budget: RiskBudget,
              constraints: Constraints) -> Portfolio: ...

class Backtester(Protocol):
    def run(self, data: PriceData, model: RiskModel, optimizer: Optimizer,
            budget: RiskBudget, schedule: RebalanceSchedule) -> BacktestResult: ...
```

These signatures are the **frozen contract** for the first build round. Changing
them requires updating this document and notifying dependent agents.

## 6. Agent workstreams & execution waves

Parallelism works because Wave 0 publishes the contracts first; everyone else
builds against interfaces, not each other's internals.

| # | Agent | Owns (packages) | Depends on | Wave |
|---|-------|-----------------|-----------|------|
| 0 | Foundation | `core/`, pyproject, CI, test harness | — | 0 |
| 1 | Data research | `docs/data-sources.md` + prototype adapter | 0 | 1 |
| 2 | Data layer | `data/synthetic.py`, `data/csvsource.py` | 0 | 1 |
| 3 | Risk model | `riskmodel/*` | 0 | 1 |
| 4 | Budgeting + optimizer | `budgeting/*`, `optimize/*` | 0, 3 (iface) | 1 |
| 5 | Backtester | `backtest/*` | 0, 2/3/4 (iface) | 2 |
| 6 | Analytics + reporting | `analytics/*`, `reporting/*` | 0, 5 (iface) | 2 |
| 7 | API + dashboard | `api/*`, `dashboard/*`, `cli/*` | 0 + all | 3 |

**Run order:** Wave 0 alone → Wave 1 (agents 1–4 in parallel) → Wave 2
(agents 5–6) → Wave 3 (agent 7). The data-research agent (1) runs concurrently;
its provider pick is wired into `data/providers/` after it reports.

**Universal rules for every agent**
- Implement only your package(s); do not modify another agent's internals.
- Honor the `core/` contracts exactly; propose contract changes back, don't fork them.
- Ship pytest tests (target >90% coverage of your package) and type hints.
- Code must pass `ruff check`, `ruff format --check`, `mypy`, and `pytest`.
- The synthetic data source is the offline backbone — your tests must not need network.

## 7. Acceptance criteria (definition of done, per area)

- **Foundation:** `pip install -e .` works; `pytest` runs (even if empty); CI green; all interfaces importable.
- **Data:** synthetic generator produces a covariance-consistent return matrix; CSV round-trips; adapters satisfy `DataSource`.
- **Risk model:** estimators return symmetric PSD matrices; shrinkage reduces condition number on ill-conditioned input; cross-checked vs. numpy reference.
- **Budgeting/optimizer:** for ERC on a known 2–3 asset case, solved TRCs are equal within tol; convex and scipy solvers agree; constraints respected.
- **Backtester:** reproducible equity curve on synthetic data; turnover and costs applied; no look-ahead (estimation uses only past window).
- **Analytics:** metrics match hand-computed references on a toy series; risk attribution sums to total risk.
- **API/dashboard:** `/construct` and `/backtest` endpoints return valid schemas; dashboard renders weights, risk-contribution bars, and equity curve.

## 8. Data-source decision (owned by Agent 1)

Deliverable: `docs/data-sources.md` containing a scored evaluation matrix and a
recommendation, plus one working prototype adapter under `data/providers/`.

Scoring dimensions: environment network policy compatibility, licensing/ToS,
asset-class coverage, history depth, corporate-action / adjusted-close handling,
survivorship bias, rate limits, API ergonomics, key management.

Candidate set to evaluate: synthetic (baseline, always ships), CSV/parquet,
Tiingo (free tier), yfinance, Stooq, Alpha Vantage, FRED (factor/macro overlay),
and one paid option (Polygon/EOD) noted for a future real-money path.

Recommended default direction (validate, don't assume): ship synthetic + CSV as
the always-works core, add **Tiingo free tier** as the first real adapter with
**yfinance** as a convenience fallback.

## 9. Out of scope (v1)

Live trading/execution, intraday data, options/derivatives risk, multi-currency
FX hedging, and authentication/multi-tenant API concerns. These are noted for a
future roadmap, not built now.
