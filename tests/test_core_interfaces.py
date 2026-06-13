"""Tests for the core interface contracts (BUILD_PLAN §5).

Verifies the config value types validate sensibly and that minimal in-test
implementations satisfy the runtime-checkable protocols.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import ConfigurationError
from riskbudget.core.interfaces import (
    Backtester,
    Constraints,
    DataSource,
    Optimizer,
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

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_constraints_default_is_long_only_fully_invested() -> None:
    c = Constraints()
    assert c.long_only is True
    assert c.leverage == 1.0
    assert Constraints.long_only_fully_invested() == c


def test_constraints_reject_bad_leverage() -> None:
    with pytest.raises(ConfigurationError):
        Constraints(leverage=0.0)


def test_constraints_reject_negative_turnover() -> None:
    with pytest.raises(ConfigurationError):
        Constraints(max_turnover=-0.1)


def test_constraints_reject_min_above_max() -> None:
    with pytest.raises(ConfigurationError):
        Constraints(min_weight=0.5, max_weight=0.2)


def test_constraints_reject_negative_group_cap() -> None:
    with pytest.raises(ConfigurationError):
        Constraints(group_caps={"tech": -0.1})


# ---------------------------------------------------------------------------
# RebalanceSchedule
# ---------------------------------------------------------------------------


def test_rebalance_schedule_defaults() -> None:
    s = RebalanceSchedule()
    assert s.frequency == "monthly"
    assert s.effective_min_lookback == 2


def test_rebalance_schedule_effective_min_lookback_uses_lookback() -> None:
    assert RebalanceSchedule(lookback=60).effective_min_lookback == 60
    assert RebalanceSchedule(lookback=60, min_lookback=30).effective_min_lookback == 30


def test_rebalance_schedule_rejects_bad_lookback() -> None:
    with pytest.raises(ConfigurationError):
        RebalanceSchedule(lookback=0)


def test_rebalance_schedule_rejects_start_after_end() -> None:
    with pytest.raises(ConfigurationError):
        RebalanceSchedule(start=date(2021, 1, 1), end=date(2020, 1, 1))


# ---------------------------------------------------------------------------
# Protocol conformance (structural typing, runtime-checkable)
# ---------------------------------------------------------------------------


class _StubDataSource:
    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        idx = pd.date_range(start, periods=3, freq="D")
        frame = pd.DataFrame({a: [100.0, 101.0, 102.0] for a in assets}, index=idx)
        return PriceData(frame)


class _StubRiskModel:
    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        return np.cov(returns.values, rowvar=False)


class _StubOptimizer:
    def solve(self, cov: np.ndarray, budget: RiskBudget, constraints: Constraints) -> Portfolio:
        n = len(budget)
        return Portfolio(dict.fromkeys(budget.assets, 1.0 / n))


class _StubBacktester:
    def run(
        self,
        data: PriceData,
        model: RiskModel,
        optimizer: Optimizer,
        budget: RiskBudget,
        schedule: RebalanceSchedule,
    ) -> BacktestResult:
        eq = pd.Series([1.0], index=data.dates[:1])
        weights = pd.DataFrame(
            {a: [1.0 / len(budget)] for a in budget.assets}, index=data.dates[:1]
        )
        return BacktestResult(equity_curve=eq, weights=weights)


def test_stubs_satisfy_protocols() -> None:
    assert isinstance(_StubDataSource(), DataSource)
    assert isinstance(_StubRiskModel(), RiskModel)
    assert isinstance(_StubOptimizer(), Optimizer)
    assert isinstance(_StubBacktester(), Backtester)


def test_stub_pipeline_runs_end_to_end() -> None:
    ds: DataSource = _StubDataSource()
    model: RiskModel = _StubRiskModel()
    opt: Optimizer = _StubOptimizer()
    bt: Backtester = _StubBacktester()

    assets = ["X", "Y"]
    prices = ds.get_prices(assets, date(2020, 1, 1), date(2020, 1, 3))
    budget = RiskBudget.equal(assets)
    cov = model.estimate(prices.to_returns())
    pf = opt.solve(cov, budget, Constraints())
    assert pf.assets == assets

    result = bt.run(prices, model, opt, budget, RebalanceSchedule())
    assert isinstance(result, BacktestResult)
