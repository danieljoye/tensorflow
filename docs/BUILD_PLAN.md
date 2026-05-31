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

Beyond the course, the design draws on Martellini's published research and that of
his co-authors (Milhau, Deguest, Meucci, Amenc, Goltz, Ziemann) — factor risk
budgeting and the Effective Number of Bets, conditional/state-dependent risk
budgets, ensemble "diversifying the diversifiers" construction, risk-based expected
returns, and the PSP/LHP/safe-asset dynamic framework. Each such method is tagged
to its source in §11 and tiered into v1 vs. roadmap (§10). Methods *not* due to
Martellini (ERC: Maillard–Roncalli–Teïletche; MDP / Diversification Ratio:
Choueifaty–Coignard; ENB/min-torsion: Meucci et al.) are attributed accordingly.

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

**Three solver paths, cheapest-that-fits** (validated against the `riskparity.py`
and `pyrb` reference implementations — see §11, §12):
1. **`ccd.py` — Spinu cyclical coordinate descent (DEFAULT, fast path).** Exact,
   pure-NumPy, converges in a few sweeps; handles long-only + bounds + leverage
   (via final rescale). The per-coordinate update is the positive root of a scalar
   quadratic — formula in §12.1.
2. **`scipy_solver.py` — log-barrier via `scipy.optimize`** with analytic gradient
   `Σx − b/x` and Hessian `Σ + diag(b/x²)`; the cross-check oracle and the bounded
   separable path.
3. **`convex.py` — `cvxpy` log-barrier program** for genuinely non-separable
   constraints (group caps, turnover, arbitrary `Cw≤d`) that CCD cannot express.

The `Optimizer` routes to the cheapest path the `Constraints` allow (mirroring
pyrb's `_lambda_solve`). An optional expected-return tilt `c·σ(w) − πᵀw`
(Roncalli) is supported by the CCD path. CVaR / tail-risk budgeting is **roadmap**
(neither reference implements it; needs a Rockafellar–Uryasev LP — §10).

### Three senses of "risk budgeting"

The system supports all three; they are complementary, not competing:

1. **Cross-sectional, across assets (the core):** distribute total portfolio risk
   *across assets* so each asset/group contributes its target share `bᵢ` at a point
   in time. This is the ERC / risk-contribution optimizer above.
2. **Cross-sectional, across factors (Deguest–Martellini–Meucci):** correlated
   assets overstate diversification. Budget risk instead across *uncorrelated
   factors* (PCA or minimum-torsion), and target the **Effective Number of Bets**.
   See the diversification block below; lives in the `diversification/` package.
3. **Temporal / dynamic (CPPI, LDI):** distribute a risk budget *over time* between
   a performance-seeking and a safe asset, subject to a floor or maximum-drawdown
   constraint. Allocation to the risky asset is `m × cushion`, where
   `cushion = (asset − floor)/asset` and `m` is the multiplier. Generalizes to a
   three-fund PSP / liability-hedging / safe-asset split (Martellini–Milhau). Lives
   in the `dynamic/` package.

Budgets may also be **conditional / state-dependent** (Martellini–Milhau–Tarelli):
the target contributions `bᵢ` can be a function of an observable state variable
(e.g. interest-rate level, yield-curve slope, valuation), falling back to ERC when
no signal is present — addressing unconditional risk parity's structural bond
overweight in changing rate regimes.

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
- **Efficient-MSR (Martellini 2008):** an MSR variant whose `μ` is a *risk-based proxy*
  (total volatility / semi-deviation) rather than the unreliable sample mean.

These require an **expected-returns** input that the core risk-budgeting path does
not; see the additive contract in §5.

### Views-based expected returns — Black-Litterman (v1)

Black-Litterman blends a **market-equilibrium prior** with subjective **views** to
produce posterior expected returns that are far more stable than raw sample means —
making it the natural `μ` source for the MSR / efficient-frontier / mean-variance
paths. Pipeline (formulas in §12.7):
- **Prior** by reverse optimization: `Π = δ·Σ·w_mkt` from market-cap weights and a
  market-implied risk-aversion `δ`.
- **Views** `(P, Q, Ω)`: `P` picks the assets, `Q` the view returns, `Ω` the view
  uncertainty (He–Litterman default `Ω = diag(diag(τPΣPᵀ))`, or Idzorek confidences).
- **Posterior** returns `E(R)` and covariance via the He–Litterman master formula,
  solved as a linear system (not a matrix inverse) for numerical stability.

Implemented as a `MeanModel`-shaped component (posterior `ExpectedReturns`) plus a
posterior covariance; its weights can also be read directly via `w=(δΣ)⁻¹E(R)`.

### Diversification measurement & factor risk budgeting

- **Diversification Ratio (Choueifaty–Coignard):** `DR(w) = (wᵀσ)/√(wᵀΣw)`. The
  **Most Diversified Portfolio (MDP)** maximizes it.
- **Effective Number of Bets, ENB (Meucci):** PCA the covariance, take each factor's
  normalized risk contribution `pᵢ`, then `ENB = exp(−Σ pᵢ ln pᵢ)`. A diversification
  score that is honest about correlation (unlike a naive count of holdings).
- **Minimum-torsion bets (Meucci–Santangelo–Deguest):** raw PCA factors are unstable
  (sign/ordering spin as `Σ` shifts). The minimum-torsion transform yields uncorrelated
  factors as close as possible to the original assets — a stable basis for ENB and for
  **factor risk budgeting** (budget risk across these factors).
- **Max-ENB constructor (Deguest–Martellini–Meucci):** maximize ENB, or constrain
  `ENB(w) ≥ k` on top of another objective.

### Ensemble construction ("diversifying the diversifiers", Amenc–Goltz–Lodh–Martellini)

Any single weighting scheme can suffer severe short-term underperformance from
estimation/model risk. Blend several (GMV, Efficient-MSR, ERC, MDP, max-decorrelation)
by averaging their normalized weights, optionally with a **tracking-error-control
overlay** that shrinks the blend toward a reference (e.g. cap- or equal-weight) when
ex-ante TE exceeds a target. This is EDHEC's "Diversified Multi-Strategy" and the
primary model-risk-reduction lever.

### Higher-moment estimation (Martellini–Ziemann)

For non-normal assets, mean-variance is insufficient and naive sample co-skewness /
co-kurtosis tensors are too noisy to help. Structured / shrinkage estimators of the
higher-order comoments make higher-moment optimization pay off out-of-sample. Treated
as an **advanced/optional** estimator (stretch), and it dovetails with the
Cornish-Fisher VaR already in the analytics layer.

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
    ewma.py          # exponentially-weighted covariance (exp_cov, span≈180)
    semicov.py       # downside semicovariance
    shrinkage.py     # Ledoit-Wolf (constant-variance + constant-correlation), OAS
    factor.py        # statistical (PCA) factor model
    psd.py           # nearest-PSD fix (spectral eigenvalue clipping) — applied to EVERY cov output
    returns_model.py # expected returns: historical mean, EWMA, CAPM-implied, risk-based proxy
    black_litterman.py # views-based posterior returns + covariance (prior, P/Q/Ω, master formula)
  budgeting/     # risk-contribution math + budget specification
    contributions.py # MRC/TRC, risk decomposition
    budget.py        # RiskBudget spec (per-asset, per-group, ERC default)
  optimize/      # solvers + constraints
    ccd.py           # Spinu cyclical coordinate descent — DEFAULT fast risk-budget solver
    convex.py        # cvxpy log-barrier program (group caps / turnover / general linear)
    scipy_solver.py  # log-barrier via scipy — cross-check oracle + bounded path
    router.py        # picks the cheapest solver the Constraints allow
    classical.py     # GMV, MSR (Cornuéjols–Tütüncü), efficient frontier, equal-weight, Efficient-MSR
    conditional.py   # state-dependent / conditional risk budgets (Martellini–Milhau–Tarelli)
    ensemble.py      # "diversifying the diversifiers" blend + tracking-error-control overlay
    constraints.py   # long-only, leverage, group caps, turnover
  diversification/ # diversification measurement + factor risk budgeting
    metrics.py       # Diversification Ratio, Effective Number of Bets (ENB) + entropy guard
    torsion.py       # minimum-torsion transform (polar iteration on the correlation root)
    constructors.py  # Most Diversified Portfolio, max-ENB, factor-risk-budget portfolios
  clustered/     # hierarchical / clustered allocation (López de Prado)
    hrp.py           # Hierarchical Risk Parity (v1: corr-distance → linkage → quasi-diag → bisection)
  dynamic/       # temporal risk budgeting (EDHEC course module 4 + Martellini–Milhau LDI)
    cppi.py          # constant proportion portfolio insurance + drawdown/floor variants
    allocators.py    # fixed-mix, glidepath, floor, drawdown allocators; bt_mix driver
    fund_separation.py # three-fund PSP / liability-hedging / safe-asset allocator
  simulate/      # scenario generation
    gbm.py           # geometric Brownian motion paths; terminal_stats over scenarios
  analytics/     # metrics & attribution
    performance.py   # annualized return/vol, Sharpe, Sortino, max drawdown, turnover
    downside.py      # drawdown, semideviation, VaR (historic/Gaussian/Cornish-Fisher), CVaR
    distribution.py  # skewness, kurtosis, Jarque-Bera normality test
    attribution.py   # risk-contribution time series, factor exposures, ENB / div-ratio over time
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
| 3 | Risk + return models | `riskmodel/*` (cov + PSD-fix, return models, Black-Litterman) | 0, 0.5 | 1 |
| 4 | Budgeting + optimizers | `budgeting/*`, `optimize/*` (ERC, GMV/MSR/EF, conditional, ensemble) | 0, 0.5, 3 (iface) | 1 |
| 9 | Diversification + factor RB | `diversification/*` (ENB, min-torsion, MDP, max-ENB) + `clustered/hrp.py` | 0, 0.5 | 1 |
| 6 | Analytics + reporting | `analytics/*`, `reporting/*` (full risk kit + ENB/DR + summary_stats) | 0, 5, 9 (iface) | 2 |
| 5 | Backtester | `backtest/*` | 0, 2/3/4 (iface) | 2 |
| 8 | Dynamic allocation | `dynamic/*` (CPPI, allocators, PSP/LHP fund separation) | 0, 0.5, 2 (iface) | 2 |
| 7 | API + dashboard | `api/*`, `dashboard/*`, `cli/*` | 0 + all | 3 |

**Run order:** Wave 0 ✅ → **Wave 0.5** (core extension, quick) → Wave 1 (agents
1, 2, 3, 4, 9 in parallel) → Wave 2 (agents 5, 6, 8) → Wave 3 (agent 7). The
data-research agent (1) runs concurrently; its provider pick is wired into
`data/providers/` after it reports. Agent 4's `ensemble.py` consumes whatever
constructors exist (GMV/MSR/ERC and, if present, Agent 9's MDP/max-ENB) and
degrades gracefully if a method is absent, so 4 and 9 stay parallel.

**Universal rules for every agent**
- Implement only your package(s); do not modify another agent's internals.
- Honor the `core/` contracts exactly; propose contract changes back, don't fork them.
- Ship pytest tests (target >90% coverage of your package) and type hints.
- Code must pass `ruff check`, `ruff format --check`, `mypy`, and `pytest`.
- The synthetic data source is the offline backbone — your tests must not need network.

## 7. Acceptance criteria (definition of done, per area)

- **Foundation:** `pip install -e .` works; `pytest` runs (even if empty); CI green; all interfaces importable.
- **Data:** synthetic generator produces a covariance-consistent return matrix; CSV round-trips; adapters satisfy `DataSource`.
- **Risk model:** estimators return symmetric PSD matrices (the PSD fix repairs a deliberately indefinite input); shrinkage reduces condition number on ill-conditioned input; **Black-Litterman with empty views returns the prior `Π`, a confident view moves the posterior toward `Q`, and the posterior is computed by solving a linear system (no explicit inverse)**; cross-checked vs. numpy reference.
- **Budgeting/optimizer:** for ERC on a known 2–3 asset case, solved TRCs are equal within tol; convex and scipy solvers agree; constraints respected.
- **Optimizers (classical):** GMV minimizes variance vs. random portfolios; MSR maximizes Sharpe on a toy 2-asset case with a known analytic tangency; efficient frontier is monotone and convex.
- **Backtester:** reproducible equity curve on synthetic data; turnover and costs applied; no look-ahead (estimation uses only past window); runs any `PortfolioConstructor` (ERC, GMV, MSR, equal-weight) through one interface for head-to-head comparison.
- **Analytics:** metrics match hand-computed references on a toy series; risk attribution sums to total risk; VaR ordering sanity (Cornish-Fisher ≥ Gaussian when left-skewed/fat-tailed); `summary_stats` reproduces the per-strategy table.
- **Diversification:** ENB of an equal-weight portfolio of `k` independent factors ≈ `k`; ENB ≤ N always; minimum-torsion factors are uncorrelated (off-diagonal of their correlation ≈ 0); MDP maximizes the diversification ratio vs. random portfolios; max-ENB ≥ ENB of GMV/equal-weight on a correlated fixture.
- **Ensemble/conditional:** the blended weights equal the mean of constituent weights (and the TE overlay reduces ex-ante tracking error vs. the reference); conditional budgets reduce to ERC when the state signal is flat and shift as documented when it is not.
- **Dynamic:** CPPI never breaches its floor on monotone-down synthetic paths; cushion/multiplier math matches a hand-worked step; floor and drawdown allocators respect their constraints; the three-fund PSP/LHP/safe allocator keeps the funding ratio above its floor on synthetic paths.
- **Simulation:** GBM mean/vol of simulated log-returns match the parameterization within sampling error; `terminal_stats` summarizes scenarios correctly.
- **API/dashboard:** `/construct` and `/backtest` endpoints return valid schemas; method selectable across the full set (ERC/risk-budget, GMV, MSR, Efficient-MSR, Black-Litterman, MDP, max-ENB, HRP, ensemble, equal-weight); dashboard renders weights, risk-contribution bars, ENB/diversification-ratio, equity curve, drawdown, and a summary-stats table comparing the chosen method against a benchmark.

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

Drawn from Martellini & co-authors' research and the EDHEC course sequel. We do
*not* build these in v1 but keep interfaces extension-friendly for them.

- **Goal-Based Investing engine** (Deguest–Martellini–Milhau): goals with priority
  tiers (essential vs. aspirational) and required cash flows; allocate between a
  goal-hedging portfolio (secures the essential floor) and the PSP; optimize/report
  **probability of success** rather than mean-variance. Computable via state-grid
  dynamic programming (Das–Ostrov–Radhakrishnan–Srivastav 2020) — §11.
- **Asset-Liability Management / LDI:** funding ratio, present-value of liabilities,
  duration matching, and liability-relative ("surplus") risk mode in the optimizer
  (minimize surplus vol / maximize surplus Sharpe vs. a liability stream). CIR
  short-rate model and bond pricing sit here. (The v1 `dynamic/fund_separation.py`
  three-fund allocator is the entry point; full ALM is roadmap.)
- **Smart-beta / factor-tilted sleeves** (Amenc–Goltz–Martellini): a two-stage
  *selection → diversified-weighting* architecture so any factor tilt (value, size,
  low-vol, momentum, quality) pairs with any constructor, avoiding score-weighted
  concentration.
- **Regime-robustness diagnostics:** report each constructor's performance
  dispersion across market regimes; prefer the ensemble where dispersion is high.
- **Full higher-moment optimization:** promote the advanced co-skewness/co-kurtosis
  estimators (Martellini–Ziemann) into a polynomial-goal-programming / expected-
  utility objective beyond the v1 estimator.
- **Advanced estimation:** entropy-pooling views (a Black-Litterman generalization),
  robust/DCC-GARCH covariance, conditioning risk budgets on richer macro state.
  (Black-Litterman itself is **v1** — §2, §12.7 — not roadmap.)
- **Clustered allocation beyond HRP:** Nested Clustered Optimization (NCO) and
  HERC/HERC2 (López de Prado; Raffinot), optimal-`k` selection (gap statistic /
  silhouette), and DBHT linkage. v1 ships plain HRP only.
- **Risk-measure-pluggable budgeting:** extend risk budgeting beyond volatility to
  CVaR, CDaR, EVaR, MAD, etc. (Rockafellar–Uryasev LP for CVaR; the Riskfolio-Lib
  `rmeasures` set is the target list). v1 is volatility-based.
- **Critical Line Algorithm (CLA):** Markowitz/Bailey–López de Prado exact-frontier
  tracer — a solver-free NumPy cross-check of the cvxpy frontier; promote from §12
  note to a built optimizer if exact turning points are needed.

## 11. Research provenance & sources

Method → source map. All citations cross-verified across multiple independent
indexers; publisher/SSRN/EDHEC pages return 403 to automated fetch, so full-text
was not opened — titles/authors/years confirmed via concurring indexes. Items
marked *(roadmap)* are deferred per §10.

**Course basis**
- EDHEC, *Introduction to Portfolio Construction and Analysis with Python*
  (Martellini & Vaidyanathan) — https://www.coursera.org/learn/introduction-portfolio-construction-python
- Public `edhec_risk_kit` reference impl — https://github.com/WongYatChun/Introduction_to_Portfolio_Construction_and_Analysis_with_Python

**Cross-sectional risk budgeting (asset-level)**
- ERC: Maillard, Roncalli & Teïletche, *JPM* 2010, 36(4):60–70 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1271972
- Convex log-barrier solver: Spinu (2013) / Maillard–Roncalli–Teïletche (as above).

**Factor risk budgeting & diversification (Agent 9)**
- Deguest, Martellini & Meucci, "Risk Parity and Beyond," WP 2013 / *JPM* 2022, 48(4):108–135 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2355778
- Meucci, "Managing Diversification" (ENB), *Risk* 2009 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1358533 · http://symmys.com/node/199
- Meucci, Santangelo & Deguest, "Risk Budgeting and Diversification Based on Optimized Uncorrelated Factors" (min-torsion), WP 2015 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2276632
- MDP / Diversification Ratio (Choueifaty & Coignard, TOBAM — *not* Martellini), *JPM* 2008, 35(1):40–51 — https://www.tobam.fr/wp-content/uploads/2014/12/TOBAM-JoPM-Maximum-Div-2008.pdf · follow-up https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1895459

**Classical / efficient benchmarks & expected returns (Agents 3–4)**
- Martellini, "Toward the Design of Better Equity Benchmarks," *JPM* 2008, 34(4):34–41 — https://jpm.pm-research.com/content/34/4/34
- Amenc, Goltz, Martellini & Retkowsky, "Efficient Indexation," *JOIM* 2011 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2066083

**Ensemble construction (Agent 4 `ensemble.py`)**
- Amenc, Goltz, Lodh & Martellini, "Diversifying the Diversifiers and Tracking the Tracking Error," *JPM* 2012, 38(3):72–88 — https://jpm.pm-research.com/content/38/3/72
- Smart Beta 2.0: Amenc & Goltz, *J. Index Investing* 2013, 4(3):15–23 *(roadmap)* — https://jii.pm-research.com/content/4/3/15
- Towards Smart Equity Factor Indices: Amenc, Goltz, Lodh & Martellini, *JPM* 2014, 40(4):106–122 *(roadmap)* — https://jpm.pm-research.com/content/40/4/106
- Robustness of Smart Beta Strategies: Amenc, Goltz, Sivasubramanian & Lodh, *JII* 2015, 6(1):17–38 *(roadmap)* — https://jii.pm-research.com/content/6/1/17
- Diversified or Concentrated Factor Tilts?: Amenc, Ducoulombier, Goltz, Lodh & Sivasubramanian, *JPM* 2016, 42(2):64–76 *(roadmap)* — https://jpm.pm-research.com/content/42/2/64

**Conditional risk budgeting (Agent 4 `conditional.py`)**
- Martellini, Milhau & Tarelli, "Toward Conditional Risk Parity," *J. Alternative Investments* 2015, 18(1):48–64 — https://jai.pm-research.com/content/18/1/48

**Higher-moment estimation (Agent 3, advanced/stretch)**
- Martellini & Ziemann, "Improved Estimates of Higher-Order Comoments and Implications for Portfolio Selection," *Review of Financial Studies* 2010, 23(4):1467–1502 — https://academic.oup.com/rfs/article-abstract/23/4/1467/1591232
- Application: Hitaj, Martellini & Zambruno, "Optimal Hedge Fund Allocation…," EDHEC/JAI 2010 — https://www.top1000funds.com/wp-content/uploads/2012/02/EDHEC_Publication_Optimal_HF_Allocation.pdf

**Dynamic allocation / LDI / fund separation (Agent 8 + roadmap)**
- Martellini & Milhau, "Dynamic Allocation Decisions in the Presence of Funding Ratio Constraints," *J. Pension Economics & Finance* 2012, 11(4):549–580 — https://www.cambridge.org/core/journals/journal-of-pension-economics-and-finance
- Amenc, Martellini, Milhau & Ziemann, "Asset-Liability Management in Private Wealth Management," *JPM* 2009, 36(1):100–120 — https://jpm.pm-research.com/content/36/1/100
- Coqueret, Martellini & Milhau, "Equity Portfolios with Improved Liability-Hedging Benefits," *JPM* 2017, 43(2):37–49 — https://jpm.pm-research.com/content/43/2/37
- Deguest, Martellini & Milhau, "Hedging vs. Insurance: Long-Horizon Investing with Short-Term Constraints," *Bankers, Markets & Investors* 2014 *(roadmap)*

**Goal-based investing (roadmap)**
- Deguest, Martellini, Milhau, Suri & Wang, "Introducing a Comprehensive Investment Framework for Goals-Based Wealth Management," EDHEC 2015 — https://www.globenewswire.com/news-release/2015/11/09/785040/0/en/New-conceptual-framework-to-better-achieve-individual-investors-goals.html
- Deguest, Martellini & Milhau, *Goal-Based Investing: Theory and Practice*, World Scientific 2021 — https://ideas.repec.org/b/wsi/wsbook/12386.html
- Das, Ostrov, Radhakrishnan & Srivastav, "Dynamic Portfolio Allocation in Goals-Based Wealth Management," *Computational Management Science* 2020, 17:613–640 (non-Martellini; the computational DP method) — https://link.springer.com/article/10.1007/s10287-019-00351-7

*Verification caveat:* every publisher/SSRN/EDHEC domain returned HTTP 403 to the
research agents' fetchers; citations were confirmed via agreement across two or more
independent indexers (journal TOC pages, RePEc/IDEAS, Semantic Scholar, SciRP
reference records), not by reading the rendered pages. No citation is fabricated;
the few items resting only on a secondary index are flagged in the agent reports.

**Reference implementations (algorithmic provenance).** This environment's network
allowlist blocks publisher PDFs but permits GitHub/PyPI, so the *algorithms* in §12
were read directly from these peer-reviewed open-source implementations (commit
pinned; read-only). These are the source of the concrete update equations, not of
the financial theory (that is cited above).
- `convexfi/riskparity.py` @ `39e6120` — Spinu CCD + Feng–Palomar SCA. Papers in-repo: Spinu (2013) SSRN 2297383; Griveau-Billion–Richard–Roncalli (2013) arXiv:1311.4057; Feng–Palomar (2015) IEEE TSP 63(19); Choi–Chen (2022).
- `jcrichard/pyrb` @ `250054e` — constrained risk budgeting. In-repo: Richard–Roncalli (2019) SSRN 3331184; Roncalli (2015) expected-returns extension.
- `reckziegel/uncorbets` @ `e128271` — minimum-torsion + ENB. In-repo: Meucci–Santangelo–Deguest, SSRN 2276632.
- `dcajasn/Riskfolio-Lib` @ `2ed0167` — HRP/NCO/HERC + risk-measure set. In-repo: López de Prado (2016, 2019); Raffinot (2017, 2018); Pfitzinger–Katzke (2019).
- `robertmartin8/PyPortfolioOpt` @ `c524c6e` — efficient frontier, CLA, Black-Litterman, estimators. In-repo: Cornuéjols–Tütüncü (2006); Ledoit–Wolf (2003, 2001); Chen et al. (2010) OAS; He–Litterman (1999/2002); Idzorek (2007); Bailey–López de Prado (2013).

## 12. Numerical methods appendix (reference-implementation-derived)

Concrete formulas extracted from the implementations in §11. Agents must reproduce
these and cross-check against the cited reference where possible.

**12.1 Spinu log-barrier CCD (default risk-budget solver — `optimize/ccd.py`).**
Objective over `x>0`: `f(x)=½xᵀΣx − Σᵢ bᵢ ln xᵢ`; final weights `w=x/Σx`.
Per-coordinate update (positive root of the scalar quadratic `Σᵢᵢxᵢ² + (Σx)₋ᵢxᵢ − bᵢ = 0`):
`xᵢ ← (aux + √(aux² + 4·Σᵢᵢ·bᵢ)) / (2·Σᵢᵢ)`, where `aux = xᵢΣᵢᵢ − (Σx)ᵢ`.
Maintain `Σx` by rank-1 update after each coordinate moves. Init `x = √(1/Σ.sum())·1`.
Stop when `maxᵢ |RCᵢ/ΣRC − bᵢ| < 1e-8` (`RCᵢ = xᵢ(Σx)ᵢ`), `maxiter≈200`. Optional
expected-return tilt → use the std-dev-measure variant `f = c·σ(x) − πᵀx − λ Σ bᵢ ln xᵢ`.

**12.2 Max-Sharpe via Cornuéjols–Tütüncü substitution (`optimize/classical.py`).**
Do NOT maximize the fractional Sharpe directly. Solve the convex QP: `min wᵀΣw` s.t.
`(μ−r_f)ᵀw = 1`, `Σw = k`, `k ≥ 0`, with all linear constraints scaled by `k`;
recover `w_real = w/k`. Requires `max(μ) > r_f`. GMV sanity value: `σ_gmv = √(1/Σ pinv(Σ))`.

**12.3 Nearest-PSD fix (`riskmodel/psd.py`, applied to every covariance output).**
PSD test = Cholesky of `Σ + 1e-16·I`. Spectral repair: `Σ = V·diag(max(λ,0))·Vᵀ`
from `eigh(Σ)`. This is mandatory before any inversion (GMV/MSR/BL) or QP solve.

**12.4 Minimum-torsion transform (`diversification/torsion.py`).**
Let `s=√diag(Σ)`, correlation `C`, and `c = sqrtm(C)` the **symmetric** (eigendecomp)
root — not Cholesky. Polar fixed-point iteration: init `d=1`; repeat
`U=diag(d)·C·diag(d)`, `u=sqrtm(U)`, `q=u⁻¹·diag(d)·c`, `d=diag(q·c)`, `π=diag(d)·q`;
stop when `|‖c−π‖_F change| / ‖c−π‖_F / n ≤ 1e-8`. Torsion `t = diag(s)·(π·c⁻¹)·diag(1/s)`.
(Approximate one-shot: `t = diag(s)·C^{-1/2}·diag(1/s)`.)

**12.5 Effective Number of Bets (`diversification/metrics.py`).**
Diversification distribution `p = (tᵀ)⁻¹b ⊙ (t·Σ·b) / (bᵀΣb)` (so `Σpᵢ=1`);
`ENB = exp(−Σ pᵢ ln pᵢ)`. Entropy guard: replace `pᵢ ln pᵢ` with
`pᵢ ln(1 + (pᵢ−1)·[pᵢ>1e-5])` to avoid `ln(0)`. Sanity: `1/N` over `N` uncorrelated
unit-variance assets ⇒ `ENB≈N`; fully correlated ⇒ `ENB≈1`.

**12.6 HRP (`clustered/hrp.py`).** Distance `Dᵢⱼ=√(½(1−ρᵢⱼ))` → `scipy.cluster.hierarchy.linkage`
→ `leaves_list` quasi-diagonalization → recursive bisection splitting capital
inversely to cluster risk, inverse-variance weights within clusters. No matrix
inversion (works on singular Σ).

**12.7 Black-Litterman (`riskmodel/black_litterman.py`).**
- Prior (reverse optimization): `Π = δ·Σ·w_mkt` (+`r_f`); market-implied
  `δ = (E[R_m] − r_f)/σ_m²`.
- Posterior returns (He–Litterman, solved as a linear system, NOT by inverting):
  let `τΣP = τ·Σ·Pᵀ`, `A = P·(τΣP) + Ω`, `b = Q − P·Π`; then
  `E(R) = Π + (τΣP)·solve(A, b)` (fall back to `lstsq` if `A` singular).
- Posterior covariance: `Σ_post = Σ + [τΣ − (τΣP)·solve(A, (τΣP)ᵀ)]`.
- Default view uncertainty: `Ω = diag(diag(τ·P·Σ·Pᵀ))` (τ cancels); Idzorek option
  maps a per-view confidence∈[0,1] to `Ω`. Implied weights `w = (δΣ)⁻¹·E(R)`,
  normalized. Defaults `τ=0.05`. Absolute-views dict → one-hot `P`, `Q`.
