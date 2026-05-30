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

It also ships **classical construction methods as benchmarks** (equal-weight,
global minimum variance, max-Sharpe / tangency, efficient frontier) so a
risk-budgeted portfolio can always be compared against them, and a **dynamic
allocation** layer (CPPI and drawdown/floor strategies) for the *temporal* sense
of risk budgeting described in §2.

### Methodology note

The risk-analytics and classical-optimization methods follow the conventions in
EDHEC Business School's *Introduction to Portfolio Construction and Analysis with
Python* (Martellini / Vaidyanathan). We adopt that course's well-tested
formulations for drawdown, downside risk (historic / Gaussian–Cornish-Fisher VaR,
CVaR), the efficient frontier (GMV, MSR), CPPI, and Monte-Carlo (GBM) simulation,
and extend beyond it with the cross-sectional risk-budgeting / ERC optimizer that
is the core of this system.

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

### Two senses of "risk budgeting"

The system supports both, and they are complementary, not competing:

1. **Cross-sectional (the core):** distribute total portfolio risk *across assets*
   so each asset/group contributes its target share `bᵢ` at a point in time. This
   is the ERC / risk-contribution optimizer above.
2. **Temporal / dynamic (CPPI):** distribute a risk budget *over time* between a
   performance-seeking (risky) asset and a safe asset, subject to a floor or
   maximum-drawdown constraint. Allocation to the risky asset is `m × (cushion)`,
   where `cushion = (asset − floor)/asset` and `m` is the multiplier. This is the
   EDHEC course's notion of risk budgeting and lives in the `dynamic/` package.

### Downside-risk vocabulary (analytics layer)

- **Drawdown:** from a wealth index, the % decline from the running peak; max drawdown is the worst.
- **Semi-deviation:** volatility computed over negative returns only.
- **VaR (Value at Risk):** the loss not exceeded at confidence `1−α`. Three estimators: historic (empirical percentile), Gaussian (parametric), and **Cornish-Fisher modified VaR** (adjusts the Gaussian quantile for skewness and kurtosis).
- **CVaR / Expected Shortfall:** the mean loss conditional on breaching VaR.
- **Distribution diagnostics:** skewness, kurtosis, and the **Jarque-Bera** normality test.

### Classical optimizers (benchmarks)

- **GMV (global minimum variance):** `min wᵀΣw` s.t. weight constraints — needs only `Σ`.
- **MSR (max Sharpe / tangency):** maximizes `(wᵀμ − r_f)/σ(w)` — needs expected returns `μ`.
- **Efficient frontier:** the set of min-variance portfolios for each target return; plus equal-weight as a naive baseline.

These require an **expected-returns** input that the core risk-budgeting path does
not; see the additive contract in §5.

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
  riskmodel/     # covariance + expected-return estimators
    sample.py        # sample covariance
    ewma.py          # exponentially-weighted
    shrinkage.py     # Ledoit-Wolf
    factor.py        # statistical (PCA) factor model
    returns_model.py # expected returns: historical mean, EWMA, CAPM-implied (for MSR/EF)
  budgeting/     # risk-contribution math + budget specification
    contributions.py # MRC/TRC, risk decomposition
    budget.py        # RiskBudget spec (per-asset, per-group, ERC default)
  optimize/      # solvers + constraints
    convex.py        # cvxpy log-barrier risk-budget formulation
    scipy_solver.py  # SLSQP fallback
    classical.py     # GMV, MSR (tangency), efficient frontier, equal-weight benchmarks
    constraints.py   # long-only, leverage, group caps, turnover
  dynamic/       # temporal risk budgeting (EDHEC course module 4)
    cppi.py          # constant proportion portfolio insurance + drawdown/floor variants
    allocators.py    # fixed-mix, glidepath, floor, drawdown allocators; bt_mix driver
  simulate/      # scenario generation
    gbm.py           # geometric Brownian motion paths; terminal_stats over scenarios
  analytics/     # metrics & attribution
    performance.py   # annualized return/vol, Sharpe, Sortino, max drawdown, turnover
    downside.py      # drawdown, semideviation, VaR (historic/Gaussian/Cornish-Fisher), CVaR
    distribution.py  # skewness, kurtosis, Jarque-Bera normality test
    attribution.py   # risk-contribution time series, factor exposures
    summary.py       # summary_stats: canonical one-row-per-strategy metrics table
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

Every downstream agent codes against these signatures. **Status: shipped** by the
Wave 0 Foundation agent (commit `ca3c8d8`) as real, validated, tested
implementations — not stubs. Two refinements were made additively without changing
any signature: `RiskBudget` requires already-normalized budgets (use
`RiskBudget.from_weights()` for relative specs), and asset ordering is explicit
(covariance ndarrays carry no labels; pass `assets=` to align).

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

### 5.1 Additive contracts for classical optimizers & dynamic layer (Wave 0.5)

The classical optimizers (MSR, efficient frontier) need **expected returns**,
which the risk-budget `Optimizer.solve(cov, budget, constraints)` signature does
not carry. Rather than overload that frozen signature, Wave 0.5 adds these to
`core/` (purely additive — nothing existing changes):

```python
# core/types.py
ExpectedReturns   # vector μ over assets (annualized); .as_array(assets)

# core/interfaces.py
class MeanModel(Protocol):                       # estimates expected returns
    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns: ...

class PortfolioConstructor(Protocol):            # generalizes portfolio construction
    def construct(self, cov: np.ndarray, *, mu: ExpectedReturns | None = None,
                  budget: RiskBudget | None = None,
                  constraints: Constraints) -> Portfolio: ...

class Allocator(Protocol):                       # dynamic (temporal) allocation
    def allocate(self, risky: ReturnMatrix, safe: ReturnMatrix | float,
                 params: AllocatorParams) -> BacktestResult: ...
```

The existing `Optimizer` is retained for the risk-budget path; `PortfolioConstructor`
is the umbrella that ERC, GMV, MSR, and equal-weight all implement, so the
backtester can run any method through one interface. This Wave 0.5 touch-up is a
small, self-contained task (extend `core/`, keep everything importable and green)
to run before Wave 1 dispatches.

## 6. Agent workstreams & execution waves

Parallelism works because Wave 0 publishes the contracts first; everyone else
builds against interfaces, not each other's internals.

| # | Agent | Owns (packages) | Depends on | Wave |
|---|-------|-----------------|-----------|------|
| 0 | Foundation | `core/`, pyproject, CI, test harness | — | 0 ✅ done |
| 0.5 | Core extension | additive `ExpectedReturns`/`MeanModel`/`PortfolioConstructor`/`Allocator` | 0 | 0.5 |
| 1 | Data research | `docs/data-sources.md` + prototype adapter | 0 | 1 |
| 2 | Data layer + simulation | `data/synthetic.py`, `data/csvsource.py`, `simulate/gbm.py` | 0 | 1 |
| 3 | Risk + return models | `riskmodel/*` (incl. `returns_model.py`) | 0, 0.5 | 1 |
| 4 | Budgeting + optimizers | `budgeting/*`, `optimize/*` (ERC + GMV/MSR/EF benchmarks) | 0, 0.5, 3 (iface) | 1 |
| 6 | Analytics + reporting | `analytics/*`, `reporting/*` (full risk kit + summary_stats) | 0, 5 (iface) | 2 |
| 5 | Backtester | `backtest/*` | 0, 2/3/4 (iface) | 2 |
| 8 | Dynamic allocation | `dynamic/*` (CPPI, floor/drawdown/glidepath allocators) | 0, 0.5, 2 (iface) | 2 |
| 7 | API + dashboard | `api/*`, `dashboard/*`, `cli/*` | 0 + all | 3 |

**Run order:** Wave 0 ✅ → **Wave 0.5** (core extension, quick) → Wave 1 (agents
1, 2, 3, 4 in parallel) → Wave 2 (agents 5, 6, 8) → Wave 3 (agent 7). The
data-research agent (1) runs concurrently; its provider pick is wired into
`data/providers/` after it reports.

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
- **Optimizers (classical):** GMV minimizes variance vs. random portfolios; MSR maximizes Sharpe on a toy 2-asset case with a known analytic tangency; efficient frontier is monotone and convex.
- **Backtester:** reproducible equity curve on synthetic data; turnover and costs applied; no look-ahead (estimation uses only past window); runs any `PortfolioConstructor` (ERC, GMV, MSR, equal-weight) through one interface for head-to-head comparison.
- **Analytics:** metrics match hand-computed references on a toy series; risk attribution sums to total risk; VaR ordering sanity (Cornish-Fisher ≥ Gaussian when left-skewed/fat-tailed); `summary_stats` reproduces the per-strategy table.
- **Dynamic:** CPPI never breaches its floor on monotone-down synthetic paths; cushion/multiplier math matches a hand-worked step; floor and drawdown allocators respect their constraints.
- **Simulation:** GBM mean/vol of simulated log-returns match the parameterization within sampling error; `terminal_stats` summarizes scenarios correctly.
- **API/dashboard:** `/construct` and `/backtest` endpoints return valid schemas; method selectable (ERC/GMV/MSR/equal-weight); dashboard renders weights, risk-contribution bars, equity curve, drawdown, and a summary-stats table.

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

## 10. Roadmap (post-v1, explicitly deferred)

The EDHEC course's later modules and its sequel ("Advanced Portfolio
Construction") map to natural extensions we will *not* build in v1 but want to
keep the architecture open to:

- **Asset-Liability Management / LDI:** funding ratio, present-value of
  liabilities, duration matching, liability-hedging portfolios, and
  liability-relative ("surplus") risk budgeting. The CIR short-rate model and bond
  pricing sit here.
- **Factor-based risk budgeting:** budget risk across *factors* (via the PCA /
  fundamental factor model) rather than across assets — a direct extension of the
  Agent 3 factor model plus the Agent 4 optimizer.
- **Advanced estimation:** Black-Litterman expected returns, robust/DCC-GARCH
  covariance, and Sharpe-style factor attribution.

These are listed so agents make interfaces extension-friendly, not so they build
them now.
