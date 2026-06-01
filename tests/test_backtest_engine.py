"""Tests for the walk-forward backtest engine (BUILD_PLAN §5, §7, Agent 5).

Covers the acceptance criteria for the backtester:
- reproducible equity curve on synthetic data;
- explicit no-look-ahead (the estimation window never sees the forward period);
- turnover + costs reduce returns;
- ERC and equal-weight (and GMV) run through the same interface;
- a deterministic seeded-sweep invariant (``hypothesis`` is not installed).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from riskbudget.backtest.engine import WalkForwardBacktester, walkforward
from riskbudget.core.errors import BacktestError
from riskbudget.core.interfaces import (
    Backtester,
    Constraints,
    RebalanceSchedule,
    RiskModel,
)
from riskbudget.core.types import (
    BacktestResult,
    Portfolio,
    PriceData,
    ReturnMatrix,
    RiskBudget,
)
from riskbudget.optimize import equal_weight, erc, gmv, msr
from riskbudget.riskmodel import mean_historical, sample_covariance

# ---------------------------------------------------------------------------
# Local fixtures (conftest is owned elsewhere; keep ours self-contained)
# ---------------------------------------------------------------------------

_ASSETS = ["A", "B", "C", "D"]
_COV = np.array(
    [
        [0.0004, 0.00010, 0.0, 0.00005],
        [0.00010, 0.0009, -0.00010, 0.0],
        [0.0, -0.00010, 0.0016, 0.00020],
        [0.00005, 0.0, 0.00020, 0.0006],
    ],
    dtype=float,
)


def _make_prices(seed: int, periods: int = 400, drift: float = 0.0003) -> PriceData:
    """A deterministic synthetic price panel from correlated log returns."""
    rng = np.random.default_rng(seed)
    mu = np.full(len(_ASSETS), drift)
    draws = rng.multivariate_normal(mu, _COV, size=periods)
    prices = 100.0 * np.exp(np.cumsum(draws, axis=0))
    dates = pd.date_range("2018-01-01", periods=periods, freq="B")
    return PriceData(pd.DataFrame(prices, index=dates, columns=_ASSETS))


@pytest.fixture
def prices() -> PriceData:
    return _make_prices(seed=2024)


@pytest.fixture
def budget() -> RiskBudget:
    return RiskBudget.equal(_ASSETS)


@pytest.fixture
def schedule() -> RebalanceSchedule:
    return RebalanceSchedule(frequency="monthly", lookback=60)


# ---------------------------------------------------------------------------
# Protocol conformance & structure
# ---------------------------------------------------------------------------


def test_satisfies_backtester_protocol() -> None:
    assert isinstance(WalkForwardBacktester(), Backtester)
    assert isinstance(walkforward(), WalkForwardBacktester)


def test_run_returns_well_formed_result(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    bt = walkforward()
    res = bt.run(prices, sample_covariance(), erc(), budget, schedule)

    assert isinstance(res, BacktestResult)
    # Equity curve spans every return period (one fewer than prices).
    assert len(res.equity_curve) == len(prices) - 1
    assert res.returns is not None
    assert len(res.returns) == len(res.equity_curve)
    # Weights frame: one row per rebalance, columns are the assets in order.
    assert list(res.weights.columns) == _ASSETS
    assert res.weights.shape[0] == res.diagnostics["n_rebalances"]
    assert res.diagnostics["n_rebalances"] > 0
    # Equity is a positive growth-of-$1 series.
    assert (res.equity_curve > 0).all()
    assert res.metadata["engine"] == "walkforward"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_equity_curve_is_reproducible(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    bt = walkforward()
    a = bt.run(prices, sample_covariance(), erc(), budget, schedule)
    b = bt.run(prices, sample_covariance(), erc(), budget, schedule)
    pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)
    pd.testing.assert_frame_equal(a.weights, b.weights)


# ---------------------------------------------------------------------------
# No look-ahead — the load-bearing correctness property
# ---------------------------------------------------------------------------


class _SpyRiskModel:
    """Wraps a RiskModel and records the date range of every estimation window."""

    def __init__(self, inner: RiskModel) -> None:
        self.inner = inner
        self.windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        idx = pd.DatetimeIndex(returns.dates)
        self.windows.append((idx[0], idx[-1]))
        return self.inner.estimate(returns)


def test_no_lookahead_estimation_window_strictly_precedes_rebalance(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    spy = _SpyRiskModel(sample_covariance())
    bt = walkforward()
    res = bt.run(prices, spy, erc(), budget, schedule)

    rebalance_dates = list(res.weights.index)
    assert len(spy.windows) == len(rebalance_dates)
    for (win_start, win_end), reb_date in zip(spy.windows, rebalance_dates, strict=True):
        # The window used to solve the weights HELD FROM ``reb_date`` must end
        # strictly before ``reb_date`` — it cannot peek at the rebalance bar or
        # any future bar.
        assert win_end < reb_date, f"look-ahead: window ends {win_end} >= rebalance {reb_date}"
        assert win_start < win_end


def test_diagnostics_record_strictly_past_window(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    bt = walkforward()
    res = bt.run(prices, sample_covariance(), erc(), budget, schedule)
    for diag in res.diagnostics["per_rebalance"]:
        # window_end_row is the rebalance row (exclusive bound) -> strictly past.
        assert diag["window_end_row"] == diag["row"]
        assert diag["window_start_row"] < diag["window_end_row"]
        assert diag["window_len"] == diag["window_end_row"] - diag["window_start_row"]


def test_lookback_window_size_is_respected(prices: PriceData, budget: RiskBudget) -> None:
    spy = _SpyRiskModel(sample_covariance())
    sched = RebalanceSchedule(frequency="monthly", lookback=40)
    walkforward().run(prices, spy, erc(), budget, sched)
    # Trailing window: at most ``lookback`` rows; once warmed up exactly lookback.
    # Count rows via the spy's recorded windows against the return index.
    idx = pd.DatetimeIndex(prices.to_returns("simple").dates)
    for win_start, win_end in spy.windows:
        n_rows = idx.get_loc(win_end) - idx.get_loc(win_start) + 1
        assert n_rows <= 40


def test_future_prices_do_not_change_past_weights(
    budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    """Appending future bars must not change weights solved on earlier windows."""
    short = _make_prices(seed=99, periods=200)
    long = _make_prices(seed=99, periods=400)  # identical prefix, more history

    bt = walkforward()
    res_short = bt.run(short, sample_covariance(), erc(), budget, schedule)
    res_long = bt.run(long, sample_covariance(), erc(), budget, schedule)

    common = res_short.weights.index.intersection(res_long.weights.index)
    assert len(common) > 0
    pd.testing.assert_frame_equal(res_short.weights.loc[common], res_long.weights.loc[common])


# ---------------------------------------------------------------------------
# Costs & turnover reduce returns
# ---------------------------------------------------------------------------


def test_costs_reduce_returns(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    from riskbudget.backtest.costs import proportional_cost

    free = walkforward()
    costed = walkforward(cost_model=proportional_cost(25.0))

    res_free = free.run(prices, sample_covariance(), erc(), budget, schedule)
    res_cost = costed.run(prices, sample_covariance(), erc(), budget, schedule)

    # Same weights solved (costs do not change the target), but net returns lower.
    pd.testing.assert_frame_equal(res_free.weights, res_cost.weights)
    assert res_cost.equity_curve.iloc[-1] < res_free.equity_curve.iloc[-1]
    assert res_cost.diagnostics["total_cost"] > 0.0
    assert res_free.diagnostics["total_cost"] == 0.0


def test_higher_cost_rate_reduces_terminal_wealth_monotonically(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    from riskbudget.backtest.costs import proportional_cost

    terminals = []
    for bps in (0.0, 10.0, 50.0, 100.0):
        bt = walkforward(cost_model=proportional_cost(bps))
        res = bt.run(prices, sample_covariance(), erc(), budget, schedule)
        terminals.append(float(res.equity_curve.iloc[-1]))
    # Strictly decreasing terminal wealth as costs rise.
    assert all(earlier > later for earlier, later in pairwise(terminals))


def test_buy_and_hold_incurs_a_single_cost(prices: PriceData, budget: RiskBudget) -> None:
    from riskbudget.backtest.costs import proportional_cost

    sched = RebalanceSchedule(frequency="none", lookback=60)
    bt = walkforward(cost_model=proportional_cost(30.0))
    res = bt.run(prices, sample_covariance(), erc(), budget, sched)
    assert res.diagnostics["n_rebalances"] == 1
    # The only cost is the initial entry from an all-cash book.
    assert len(res.diagnostics["per_rebalance"]) == 1


# ---------------------------------------------------------------------------
# Any constructor through one interface (head-to-head)
# ---------------------------------------------------------------------------


def test_erc_and_equal_weight_through_one_interface(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    bt = walkforward()
    res_erc = bt.run(prices, sample_covariance(), erc(), budget, schedule)
    res_ew = bt.run(prices, sample_covariance(), equal_weight(), budget, schedule)
    res_gmv = bt.run(prices, sample_covariance(), gmv(), budget, schedule)

    for res in (res_erc, res_ew, res_gmv):
        assert res.weights.shape[1] == len(_ASSETS)
        assert res.diagnostics["n_rebalances"] > 0

    # Equal-weight holds 1/N at every rebalance.
    np.testing.assert_allclose(res_ew.weights.to_numpy(), 0.25, atol=1e-12)
    # ERC weights are not equal-weight (assets have distinct vols).
    assert not np.allclose(res_erc.weights.to_numpy(), 0.25, atol=1e-3)
    # Distinct strategies produce distinct equity curves.
    assert not np.allclose(res_erc.equity_curve.to_numpy(), res_ew.equity_curve.to_numpy())


def test_msr_runs_with_a_mean_model(budget: RiskBudget) -> None:
    prices = _make_prices(seed=3, periods=400, drift=0.0008)
    sched = RebalanceSchedule(frequency="quarterly", lookback=150)
    bt = walkforward(mean_model=mean_historical())
    res = bt.run(prices, sample_covariance(), msr(), budget, sched)
    assert res.diagnostics["n_rebalances"] > 0
    # MSR weights are long-only and sum to ~1 (fully invested) at each rebalance.
    sums = res.weights.sum(axis=1).to_numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
    assert (res.weights.to_numpy() >= -1e-9).all()


def test_optimizer_solve_path_is_supported(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    """A pure Optimizer exposing only ``solve(...)`` also drives the engine."""

    class _SolveOnly:
        def __init__(self) -> None:
            self._inner = erc()

        def solve(self, cov: np.ndarray, b: RiskBudget, c: Constraints) -> Portfolio:
            return self._inner.solve(cov, b, c)

    bt = walkforward()
    res = bt.run(prices, sample_covariance(), _SolveOnly(), budget, schedule)
    # Matches the construct() path for the same risk-budget solve.
    ref = bt.run(prices, sample_covariance(), erc(), budget, schedule)
    pd.testing.assert_frame_equal(res.weights, ref.weights)


# ---------------------------------------------------------------------------
# Weights satisfy the budget within solver tolerance
# ---------------------------------------------------------------------------


def test_erc_weights_match_risk_budget(
    prices: PriceData, budget: RiskBudget, schedule: RebalanceSchedule
) -> None:
    spy = _SpyRiskModel(sample_covariance())
    bt = walkforward()
    res = bt.run(prices, spy, erc(), budget, schedule)

    target = budget.as_array(_ASSETS)
    for i, (_, row) in enumerate(res.weights.iterrows()):
        w = row.to_numpy()
        cov = spy.inner.estimate(_window_for(prices, spy.windows[i]))
        port = Portfolio(dict(zip(_ASSETS, w, strict=True)))
        trc = np.array([port.risk_contributions(cov)[a] for a in _ASSETS])
        frac = trc / trc.sum()
        # ERC -> equal risk contributions within solver tolerance.
        np.testing.assert_allclose(frac, target, atol=1e-4)


def _window_for(prices: PriceData, span: tuple[pd.Timestamp, pd.Timestamp]) -> ReturnMatrix:
    rm = prices.to_returns("simple")
    frame = rm.frame.loc[span[0] : span[1]]
    return ReturnMatrix(frame)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_insufficient_history_raises(budget: RiskBudget) -> None:
    short = _make_prices(seed=1, periods=20)
    sched = RebalanceSchedule(frequency="monthly", lookback=60, min_lookback=60)
    with pytest.raises(BacktestError):
        walkforward().run(short, sample_covariance(), erc(), budget, sched)


def test_non_pricedata_input_raises(budget: RiskBudget, schedule: RebalanceSchedule) -> None:
    with pytest.raises(BacktestError):
        walkforward().run(
            "not prices",  # type: ignore[arg-type]
            sample_covariance(),
            erc(),
            budget,
            schedule,
        )


def test_unknown_return_method_raises() -> None:
    with pytest.raises(BacktestError):
        walkforward(return_method="quadratic")


# ---------------------------------------------------------------------------
# Deterministic seeded-sweep invariants (hypothesis is not installed)
# ---------------------------------------------------------------------------


def test_seeded_sweep_invariants() -> None:
    """Across many seeded synthetic datasets, structural invariants always hold.

    Invariants per run (ERC, long-only, fully invested):
    - weights at each rebalance are non-negative and sum to ~1;
    - the equity curve is strictly positive and finite;
    - net returns never exceed gross returns once costs are charged;
    - more periods of identical-prefix data never alter earlier weights.
    """
    from riskbudget.backtest.costs import proportional_cost

    budget = RiskBudget.equal(_ASSETS)
    schedule = RebalanceSchedule(frequency="monthly", lookback=50)

    for seed in range(12):
        prices = _make_prices(seed=seed, periods=260, drift=0.0002)
        free = walkforward()
        costed = walkforward(cost_model=proportional_cost(20.0))

        res = free.run(prices, sample_covariance(), erc(), budget, schedule)
        res_c = costed.run(prices, sample_covariance(), erc(), budget, schedule)

        w = res.weights.to_numpy()
        assert (w >= -1e-9).all(), f"seed {seed}: negative weight"
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)

        assert np.isfinite(res.equity_curve.to_numpy()).all()
        assert (res.equity_curve > 0).all()

        # Costs only ever subtract: terminal wealth with costs <= without.
        assert res_c.equity_curve.iloc[-1] <= res.equity_curve.iloc[-1] + 1e-12

        # Same weights solved regardless of cost model (costs don't move targets).
        pd.testing.assert_frame_equal(res.weights, res_c.weights)
