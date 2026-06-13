"""Regression tests for the correctness-review findings (H1, H2, M1, L1, L2).

Each test pins the FIXED behavior so the original bug cannot silently return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import riskbudget as rb
from riskbudget.backtest.costs import compute_turnover
from riskbudget.backtest.engine import WalkForwardBacktester
from riskbudget.optimize.voltarget import VolatilityTargetConstructor
from riskbudget.spec import StrategySpec

COV = [[4e-4, 1e-4, 0.0], [1e-4, 9e-4, 2e-4], [0.0, 2e-4, 1.6e-3]]


# -- H1: drift must preserve gross exposure (cash residual at rf=0) -----------


def test_drift_preserves_leverage_on_zero_returns() -> None:
    drift = WalkForwardBacktester._drift
    # a 2x book with zero returns stays 2x (the old bug renormalized to 1x)
    out = drift(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert out.sum() == pytest.approx(2.0)
    # a half-cash book stays half-cash (the old bug levered it up to 1x)
    out = drift(np.array([0.25, 0.25]), np.array([0.0, 0.0]))
    assert out.sum() == pytest.approx(0.5)


def test_drift_unlevered_book_unchanged_by_fix() -> None:
    drift = WalkForwardBacktester._drift
    w = np.array([0.6, 0.4])
    r = np.array([0.02, -0.01])
    out = drift(w, r)
    grown = w * (1 + r)
    np.testing.assert_allclose(out, grown / grown.sum())  # old formula, Σw == 1
    assert out.sum() == pytest.approx(grown.sum() / (1 + w @ r))


def test_drift_levered_book_grows_with_portfolio_return() -> None:
    drift = WalkForwardBacktester._drift
    w = np.array([1.2, 0.8])  # 2x gross, financed at rf=0
    r = np.array([0.01, 0.01])
    out = drift(w, r)
    # portfolio grew 1 + w·r = 1.02; positions grew 1.01 -> gross 2*1.01/1.02
    assert out.sum() == pytest.approx(2.0 * 1.01 / 1.02)


def test_voltarget_leverage_persists_between_monthly_rebalances() -> None:
    """End-to-end: a levered vol-target book must HOLD its leverage between
    monthly rebalances (the H1 bug snapped it back to 1x after one day)."""
    spec = StrategySpec(
        name="vt",
        assets=["A", "B", "C"],
        data_source={"name": "synthetic", "params": {"cov": COV}},
        risk_model="sample",
        schedule={"frequency": "monthly", "lookback": 252, "min_lookback": 252},
        periods_per_year=252,
        target_volatility=0.10,
        target_vol_max_leverage=3.0,
    )
    res = rb.backtest(spec)
    solved_lev = np.array([d["leverage"] for d in res.diagnostics["per_rebalance"]])
    realized = float(res.returns.std() * math.sqrt(252))
    # These assets have 32-64% annualized vols, so the vol-target DE-levers the
    # book to ~0.36x. With the drift fix that partial-cash book persists between
    # monthly rebalances and realized vol tracks the 10% target; with the H1 bug
    # the book silently re-levered itself to 1x after one day and realized ~3x
    # the target.
    assert solved_lev.mean() < 0.5
    assert 0.07 < realized < 0.13


# -- H2: mean model gets the same periods_per_year as the risk model ----------


def test_mean_model_periods_per_year_propagated() -> None:
    spec = StrategySpec(
        name="msr",
        assets=["A", "B"],
        data_source={"name": "synthetic", "params": {"cov": [[4e-4, 0.0], [0.0, 9e-4]]}},
        method="msr",
        risk_model="sample",
        mean_model="mean_historical",
        periods_per_year=12,
    )
    assert spec.build_risk_model().periods_per_year == 12.0
    assert spec.build_mean_model().periods_per_year == 12.0  # was 252 (bug H2)


def test_mean_model_explicit_param_wins() -> None:
    spec = StrategySpec(
        name="msr",
        assets=["A", "B"],
        data_source={"name": "synthetic", "params": {"cov": [[4e-4, 0.0], [0.0, 9e-4]]}},
        method="msr",
        risk_model="sample",
        mean_model="mean_historical",
        mean_model_params={"periods_per_year": 52},
        periods_per_year=12,
    )
    assert spec.build_mean_model().periods_per_year == 52.0


# -- M1: one-way turnover counts the larger traded leg ------------------------


def test_turnover_gross_preserving_matches_half_sum() -> None:
    prev = np.array([0.6, 0.4])
    new = np.array([0.4, 0.6])
    assert compute_turnover(prev, new) == pytest.approx(0.2)  # == 0.5 * Σ|Δw|


def test_turnover_lever_up_counts_full_buy_leg() -> None:
    prev = np.array([0.5, 0.5])  # 1x
    new = np.array([1.0, 1.0])  # 2x: buys 1.0 NAV, sells nothing
    assert compute_turnover(prev, new) == pytest.approx(1.0)  # was 0.5 (bug M1)


def test_turnover_initial_deployment_counts_full_notional() -> None:
    prev = np.zeros(2)
    new = np.array([0.5, 0.5])
    assert compute_turnover(prev, new) == pytest.approx(1.0)


# -- L1/L2 spot checks ---------------------------------------------------------


def test_voltarget_zero_vol_goes_to_cash() -> None:
    class _EW:
        def construct(self, cov, *, mu=None, budget=None, constraints):
            from riskbudget.core.types import Portfolio

            return Portfolio({"A": 0.5, "B": 0.5})

    vt = VolatilityTargetConstructor(_EW(), target_volatility=0.10)
    port = vt.construct(np.zeros((2, 2)), constraints=None)
    assert port.leverage == pytest.approx(0.0)


def test_stocks_tr_missing_dividend_keeps_price_return() -> None:
    import pandas as pd
    from riskbudget.data.providers.daily import _stocks_tr_index

    dates = pd.date_range("2000-01-03", periods=4, freq="B")
    stocks = pd.Series([100.0, 110.0, 110.0, 121.0], index=dates)
    # dividend yield series starts AFTER the second price date
    div_yield = pd.Series([0.02], index=[dates[2]])
    tr = _stocks_tr_index(stocks, div_yield)
    # the +10% price move on day 2 must be reflected even with no dividend data
    assert tr.iloc[1] / tr.iloc[0] == pytest.approx(1.10)  # was 1.0 (bug L2)
