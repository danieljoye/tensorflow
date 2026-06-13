"""Schedule, frequency, return-method, and registry tests for the backtester."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from riskbudget.backtest import (
    BACKTESTER_FACTORIES,
    COST_MODEL_FACTORIES,
    register,
    walkforward,
)
from riskbudget.core.errors import BacktestError
from riskbudget.core.interfaces import RebalanceSchedule
from riskbudget.core.types import PriceData, RiskBudget
from riskbudget.optimize import erc
from riskbudget.riskmodel import sample_covariance

_ASSETS = ["A", "B", "C"]
_COV = np.array(
    [[0.0004, 0.0001, 0.0], [0.0001, 0.0009, -0.0001], [0.0, -0.0001, 0.0016]],
    dtype=float,
)


def _prices(seed: int = 5, periods: int = 300) -> PriceData:
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.full(3, 0.0003), _COV, size=periods)
    prices = 100.0 * np.exp(np.cumsum(draws, axis=0))
    dates = pd.date_range("2019-01-01", periods=periods, freq="B")
    return PriceData(pd.DataFrame(prices, index=dates, columns=_ASSETS))


@pytest.fixture
def budget() -> RiskBudget:
    return RiskBudget.equal(_ASSETS)


@pytest.mark.parametrize(
    ("frequency", "expected_order"),
    [
        ("daily", "most"),
        ("weekly", "many"),
        ("monthly", "some"),
        ("quarterly", "few"),
        ("annual", "fewest"),
        ("none", "one"),
    ],
)
def test_rebalance_count_orders_by_frequency(
    budget: RiskBudget, frequency: str, expected_order: str
) -> None:
    sched = RebalanceSchedule(frequency=frequency, lookback=40)  # type: ignore[arg-type]
    res = walkforward().run(_prices(), sample_covariance(), erc(), budget, sched)
    n = res.diagnostics["n_rebalances"]
    assert n >= 1
    if frequency == "none":
        assert n == 1
    if frequency == "daily":
        # Daily rebalances on (almost) every available row.
        assert n > 200


def test_frequencies_are_monotone_in_count(budget: RiskBudget) -> None:
    counts = {}
    for freq in ("annual", "quarterly", "monthly", "weekly", "daily"):
        sched = RebalanceSchedule(frequency=freq, lookback=40)  # type: ignore[arg-type]
        res = walkforward().run(_prices(), sample_covariance(), erc(), budget, sched)
        counts[freq] = res.diagnostics["n_rebalances"]
    assert (
        counts["annual"]
        <= counts["quarterly"]
        <= counts["monthly"]
        <= counts["weekly"]
        <= counts["daily"]
    )


def test_log_and_simple_return_methods_both_run(budget: RiskBudget) -> None:
    sched = RebalanceSchedule(frequency="monthly", lookback=40)
    prices = _prices()
    simple = walkforward(return_method="simple").run(
        prices, sample_covariance(), erc(), budget, sched
    )
    log = walkforward(return_method="log").run(prices, sample_covariance(), erc(), budget, sched)
    assert simple.metadata["return_method"] == "simple"
    assert log.metadata["return_method"] == "log"
    # Both produce positive equity curves of equal length.
    assert len(simple.equity_curve) == len(log.equity_curve)
    assert (simple.equity_curve > 0).all()
    assert (log.equity_curve > 0).all()


def test_start_end_window_clips_rebalances(budget: RiskBudget) -> None:
    prices = _prices(periods=300)
    full = RebalanceSchedule(frequency="monthly", lookback=40)
    clipped = RebalanceSchedule(
        frequency="monthly",
        lookback=40,
        start=dt.date(2019, 6, 1),
        end=dt.date(2019, 10, 1),
    )
    res_full = walkforward().run(prices, sample_covariance(), erc(), budget, full)
    res_clip = walkforward().run(prices, sample_covariance(), erc(), budget, clipped)
    assert res_clip.diagnostics["n_rebalances"] < res_full.diagnostics["n_rebalances"]
    for d in res_clip.weights.index:
        assert pd.Timestamp("2019-06-01") <= d <= pd.Timestamp("2019-10-01")


def test_empty_window_raises(budget: RiskBudget) -> None:
    # A start/end window beyond the data leaves no valid rebalance dates.
    sched = RebalanceSchedule(
        frequency="monthly",
        lookback=40,
        start=dt.date(2099, 1, 1),
        end=dt.date(2099, 12, 31),
    )
    with pytest.raises(BacktestError):
        walkforward().run(_prices(), sample_covariance(), erc(), budget, sched)


# -- registry hook ----------------------------------------------------------


class _Registry:
    def __init__(self) -> None:
        self.backtesters: dict[str, object] = {}
        self.cost_models: dict[str, object] = {}

    def register_backtester(self, name: str, factory: object) -> None:
        self.backtesters[name] = factory

    def register_cost_model(self, name: str, factory: object) -> None:
        self.cost_models[name] = factory


def test_register_hook_with_typed_registry() -> None:
    reg = _Registry()
    register(reg)
    assert "walkforward" in reg.backtesters
    assert reg.backtesters["walkforward"] is BACKTESTER_FACTORIES["walkforward"]
    assert "proportional" in reg.cost_models
    assert "none" in reg.cost_models
    assert reg.cost_models["proportional"] is COST_MODEL_FACTORIES["proportional"]


class _GenericRegistry:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, object]] = []

    def register(self, kind: str, name: str, factory: object) -> None:
        self.items.append((kind, name, factory))


def test_register_hook_falls_back_to_generic() -> None:
    reg = _GenericRegistry()
    register(reg)
    kinds = {kind for kind, _, _ in reg.items}
    assert kinds == {"backtester", "cost_model"}
    names = {name for _, name, _ in reg.items}
    assert {"walkforward", "proportional", "none"} <= names
