"""Tests for the proportional transaction-cost model (BUILD_PLAN §7, Agent 5)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.backtest.costs import (
    CostModel,
    ProportionalCost,
    compute_turnover,
    no_cost,
    proportional_cost,
)
from riskbudget.core.errors import BacktestError


def test_turnover_one_way_is_half_two_way() -> None:
    prev = np.array([0.5, 0.5, 0.0])
    new = np.array([0.0, 0.5, 0.5])
    two_way = compute_turnover(prev, new, one_way=False)
    one_way = compute_turnover(prev, new, one_way=True)
    assert two_way == pytest.approx(1.0)
    assert one_way == pytest.approx(0.5)


def test_turnover_zero_for_identical_books() -> None:
    w = np.array([0.2, 0.3, 0.5])
    assert compute_turnover(w, w) == 0.0


def test_turnover_rejects_misaligned_vectors() -> None:
    with pytest.raises(BacktestError):
        compute_turnover(np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0]))


def test_turnover_rejects_non_finite() -> None:
    with pytest.raises(BacktestError):
        compute_turnover(np.array([np.nan, 0.0]), np.array([0.0, 1.0]))


def test_proportional_cost_is_rate_times_turnover() -> None:
    model = ProportionalCost(bps=10.0)  # 10 bps == 1e-3
    prev = np.array([0.5, 0.5])
    new = np.array([0.0, 1.0])
    # one-way turnover = 0.5; cost = 1e-3 * 0.5 = 5e-4
    assert model.cost(prev, new) == pytest.approx(5e-4)
    assert model.rate == pytest.approx(1e-3)


def test_cost_monotonic_in_bps_and_turnover() -> None:
    prev = np.array([0.5, 0.5])
    small = np.array([0.4, 0.6])
    large = np.array([0.0, 1.0])
    cheap = ProportionalCost(bps=5.0)
    pricey = ProportionalCost(bps=25.0)
    # More turnover costs more at a fixed rate.
    assert cheap.cost(prev, small) < cheap.cost(prev, large)
    # Higher rate costs more at fixed turnover.
    assert cheap.cost(prev, large) < pricey.cost(prev, large)


def test_no_cost_is_zero() -> None:
    model = no_cost()
    assert model.cost(np.array([0.5, 0.5]), np.array([0.0, 1.0])) == 0.0


def test_negative_bps_rejected() -> None:
    with pytest.raises(BacktestError):
        ProportionalCost(bps=-1.0)


def test_factory_and_protocol() -> None:
    model = proportional_cost(7.5)
    assert isinstance(model, CostModel)
    assert model.bps == 7.5
