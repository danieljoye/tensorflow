"""Tests for the dynamic allocators and bt_mix driver (Agent 8).

Covers fixed-mix, glidepath (calendar + state-dependent interpolation), floor and
drawdown allocators (limit respected), the Allocator protocol, and a bt_mix run
over GBM scenarios with a sane terminal-wealth distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.interfaces import Allocator
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix
from riskbudget.dynamic.allocators import (
    DrawdownAllocator,
    FixedMixAllocator,
    bt_mix,
    drawdown,
    fixed_mix,
    floor,
    glidepath,
)
from riskbudget.simulate.gbm import gbm


def _risky(returns: np.ndarray) -> ReturnMatrix:
    idx = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    return ReturnMatrix(pd.DataFrame({"RISKY": np.asarray(returns, dtype=float)}, index=idx))


def test_all_factories_satisfy_allocator_protocol() -> None:
    assert isinstance(fixed_mix(), Allocator)
    assert isinstance(glidepath(), Allocator)
    assert isinstance(floor(), Allocator)
    assert isinstance(drawdown(max_drawdown=0.2), Allocator)


def test_fixed_mix_holds_constant_weight() -> None:
    alloc = fixed_mix(risky_weight=0.6)
    res = alloc.allocate(_risky(np.full(50, 0.001)), 0.0, AllocatorParams())
    assert isinstance(res, BacktestResult)
    risky_col = res.metadata["risky_asset"]
    assert np.allclose(res.weights[risky_col].to_numpy(), 0.6)
    # 60/40 of a +0.1% risky / 0% safe path -> +0.06% per step.
    assert res.returns is not None
    assert res.returns.iloc[0] == pytest.approx(0.6 * 0.001)


def test_fixed_mix_rejects_weight_above_cap() -> None:
    with pytest.raises(ValidationError):
        FixedMixAllocator(risky_weight=1.5, leverage_cap=1.0)


def test_glidepath_calendar_interpolates_linearly() -> None:
    alloc = glidepath(start_weight=0.9, end_weight=0.1)
    res = alloc.allocate(_risky(np.zeros(11)), 0.0, AllocatorParams())
    weights = res.diagnostics["risky_weight"].to_numpy()
    assert weights[0] == pytest.approx(0.9)
    assert weights[-1] == pytest.approx(0.1)
    assert weights[5] == pytest.approx(0.5)  # midpoint of 11 evenly spaced points
    # Strictly decreasing linear ramp.
    assert np.all(np.diff(weights) < 0)


def test_glidepath_single_period_uses_start_weight() -> None:
    res = glidepath(start_weight=0.7, end_weight=0.2).allocate(
        _risky(np.zeros(1)), 0.0, AllocatorParams()
    )
    assert res.diagnostics["risky_weight"].iloc[0] == pytest.approx(0.7)


def test_glidepath_state_dependent_scales_with_cushion() -> None:
    # State-dependent: weight = start + (end-start)*clip(cushion,0,1). With
    # start=0.0, end=1.0 the weight equals the cushion. Start at full cushion
    # (floor 0) on a flat path -> cushion 1 -> weight 1.
    alloc = glidepath(start_weight=0.0, end_weight=1.0, state_dependent=True)
    res = alloc.allocate(_risky(np.zeros(20)), 0.0, AllocatorParams(floor=0.0))
    assert res.metadata["state_dependent"] is True
    assert res.diagnostics["risky_weight"].iloc[0] == pytest.approx(1.0)


def test_glidepath_state_dependent_derisks_as_cushion_thins() -> None:
    # On a falling path with a floor, the cushion shrinks, so the cushion-driven
    # weight should decline over time.
    alloc = glidepath(start_weight=0.0, end_weight=1.0, state_dependent=True)
    res = alloc.allocate(_risky(np.full(100, -0.01)), 0.0, AllocatorParams(floor=0.8))
    weights = res.diagnostics["risky_weight"].to_numpy()
    assert weights[-1] < weights[0]


def test_floor_allocator_respects_floor() -> None:
    res = floor().allocate(
        _risky(np.full(300, -0.02)), 0.0, AllocatorParams(multiplier=3.0, floor=0.8)
    )
    assert np.all(res.equity_curve.to_numpy() >= 0.8 - 1e-9)
    assert res.metadata["strategy"] == "floor"
    assert res.metrics["floor_breached"] == 0.0


def test_drawdown_allocator_respects_limit() -> None:
    # Drawdown limit 15%: the account must stay above 85% of its running peak.
    res = drawdown(max_drawdown=0.15).allocate(
        _risky(np.full(250, -0.015)), 0.0, AllocatorParams(multiplier=3.0, floor=0.0)
    )
    account = res.equity_curve.to_numpy()
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], account)))[1:]
    assert np.all(account >= 0.85 * running_peak - 1e-9)
    assert res.metadata["max_drawdown"] == pytest.approx(0.15)


def test_drawdown_limit_can_come_from_params() -> None:
    res = drawdown().allocate(
        _risky(np.full(120, -0.01)),
        0.0,
        AllocatorParams(multiplier=3.0, floor=0.0, max_drawdown=0.2),
    )
    account = res.equity_curve.to_numpy()
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], account)))[1:]
    assert np.all(account >= 0.80 * running_peak - 1e-9)


def test_drawdown_allocator_requires_a_limit() -> None:
    with pytest.raises(BacktestError):
        DrawdownAllocator().allocate(_risky(np.zeros(10)), 0.0, AllocatorParams())


def test_bt_mix_over_gbm_scenarios_sane_terminal_wealth() -> None:
    paths = gbm(
        n_years=5.0,
        steps_per_year=12,
        mu=0.06,
        sigma=0.15,
        n_scenarios=200,
        s_0=100.0,
        seed=20240517,
        prices=True,
    )
    out = bt_mix(
        fixed_mix(risky_weight=1.0),
        paths,
        0.0,
        AllocatorParams(multiplier=3.0, floor=0.0, start_value=1.0),
        is_prices=True,
        periods_per_year=12,
    )
    terminal = out["terminal_wealth"]
    assert isinstance(terminal, np.ndarray)
    assert terminal.shape == (200,)
    # All-risky fixed mix should roughly track the underlying GBM growth: positive,
    # finite, with a wide but bounded spread.
    assert np.all(np.isfinite(terminal))
    assert np.all(terminal > 0)
    summary = out["summary"]
    assert summary["n_scenarios"] == 200.0
    assert 0.5 < summary["median"] < 5.0
    assert summary["min"] <= summary["median"] <= summary["max"]
    assert out["mean_equity_curve"].shape[0] == paths.shape[0] - 1


def test_bt_mix_cppi_floor_reduces_breach_probability() -> None:
    # A CPPI floor should produce far fewer floor breaches than an all-risky mix on
    # the same GBM scenarios.
    from riskbudget.dynamic.cppi import cppi

    paths = gbm(
        n_years=5.0,
        steps_per_year=12,
        mu=0.0,
        sigma=0.30,
        n_scenarios=300,
        s_0=100.0,
        seed=42,
        prices=True,
    )
    params = AllocatorParams(multiplier=3.0, floor=0.8, start_value=1.0)
    cppi_out = bt_mix(cppi(), paths, 0.0, params, is_prices=True, periods_per_year=12)
    mix_out = bt_mix(fixed_mix(risky_weight=1.0), paths, 0.0, params, is_prices=True)
    assert cppi_out["summary"]["p_breach_floor"] <= mix_out["summary"]["p_breach_floor"]


def test_bt_mix_rejects_empty_frame() -> None:
    with pytest.raises(BacktestError):
        bt_mix(fixed_mix(), pd.DataFrame(), 0.0, AllocatorParams())


def test_bt_mix_rejects_non_allocator() -> None:
    paths = gbm(n_years=1.0, steps_per_year=12, n_scenarios=2, seed=1)
    with pytest.raises(BacktestError):
        bt_mix(object(), paths, 0.0, AllocatorParams())


def test_bt_mix_seeded_determinism() -> None:
    def run() -> np.ndarray:
        paths = gbm(n_years=2.0, steps_per_year=12, n_scenarios=50, seed=2024)
        out = bt_mix(fixed_mix(risky_weight=0.7), paths, 0.0, AllocatorParams(floor=0.0))
        return np.asarray(out["terminal_wealth"])

    assert np.array_equal(run(), run())
