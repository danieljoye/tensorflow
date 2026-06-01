"""Tests for CPPI (Agent 8, ``riskbudget/dynamic/cppi.py``).

Covers the floor invariant on monotone-down paths, a hand-worked cushion/multiplier
step, the drawdown-floor variant, the Allocator protocol, error taxonomy, and a
deterministic seeded sweep (hypothesis is not installed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.interfaces import Allocator
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix
from riskbudget.dynamic.cppi import CPPI, cppi, run_cppi_path


def _risky(returns: np.ndarray) -> ReturnMatrix:
    idx = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    return ReturnMatrix(pd.DataFrame({"RISKY": np.asarray(returns, dtype=float)}, index=idx))


def test_cppi_satisfies_allocator_protocol() -> None:
    assert isinstance(cppi(), Allocator)
    assert isinstance(CPPI(), Allocator)


def test_factory_returns_cppi() -> None:
    alloc = cppi(leverage_cap=1.0, periods_per_year=252)
    assert isinstance(alloc, CPPI)


def test_never_breaches_floor_on_monotone_down_path() -> None:
    # A long, steep, monotone-down risky path. With a fixed floor and zero safe
    # return, CPPI must keep the account at or above the floor every step.
    risky = np.full(400, -0.03)
    safe = np.zeros_like(risky)
    hist = run_cppi_path(
        risky, safe, multiplier=3.0, floor_fraction=0.8, start_value=1.0, max_drawdown=None
    )
    floor_level = 0.8
    assert np.all(hist.account >= floor_level - 1e-9)
    # As the account falls toward the floor, the cushion (and risky weight) shrink.
    assert hist.risky_weight[-1] <= hist.risky_weight[0]
    assert hist.account[-1] >= floor_level - 1e-9


def test_hand_worked_cushion_and_multiplier_step() -> None:
    # One step: start_value=1.0, floor=0.8 -> cushion=(1-0.8)/1=0.2,
    # risky_weight=clip(3*0.2,0,1)=0.6. Risky +10%, safe 0% ->
    # step_return=0.6*0.10=0.06 -> account=1.06.
    risky = np.array([0.10])
    safe = np.array([0.0])
    hist = run_cppi_path(
        risky, safe, multiplier=3.0, floor_fraction=0.8, start_value=1.0, max_drawdown=None
    )
    assert hist.cushion[0] == pytest.approx(0.2)
    assert hist.risky_weight[0] == pytest.approx(0.6)
    assert hist.account[0] == pytest.approx(1.06)


def test_risky_weight_clips_at_leverage_cap() -> None:
    # Large multiplier -> m*cushion exceeds the cap -> weight clamps to cap.
    hist = run_cppi_path(
        np.array([0.05]),
        np.array([0.0]),
        multiplier=100.0,
        floor_fraction=0.5,
        start_value=1.0,
        leverage_cap=1.0,
    )
    assert hist.risky_weight[0] == pytest.approx(1.0)


def test_drawdown_floor_caps_drawdown() -> None:
    # Drawdown floor at 1 - 0.10 = 0.90 of the running peak: the account should
    # never fall below 90% of its peak (gap risk aside) on a smooth down path.
    risky = np.full(300, -0.01)
    safe = np.zeros_like(risky)
    hist = run_cppi_path(
        risky, safe, multiplier=3.0, floor_fraction=0.0, start_value=1.0, max_drawdown=0.10
    )
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], hist.account)))[1:]
    assert np.all(hist.account >= 0.90 * running_peak - 1e-9)


def test_allocate_builds_consumable_backtest_result() -> None:
    rng = np.random.default_rng(7)
    risky = _risky(rng.normal(0.0005, 0.012, size=250))
    result = cppi().allocate(risky, 0.0, AllocatorParams(multiplier=3.0, floor=0.8))
    assert isinstance(result, BacktestResult)
    # Weights sum to one each period (risky + safe).
    row_sums = result.weights.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0)
    assert result.equity_curve.shape[0] == 250
    assert "cushion" in result.diagnostics
    assert result.metadata["strategy"] == "cppi"
    assert result.metrics["floor_breached"] == 0.0


def test_scalar_safe_rate_is_deannualized() -> None:
    # The allocator must de-annualize a scalar safe rate by periods_per_year: a
    # scalar 5% annual rate and an explicit per-period series compounding to 5%
    # over a year must give the *same* CPPI run.
    n = 252
    safe_rate = 0.05
    per_period = (1.0 + safe_rate) ** (1.0 / n) - 1.0
    risky = _risky(np.full(n, 0.0003))
    params = AllocatorParams(multiplier=3.0, floor=0.8)

    scalar_run = cppi(periods_per_year=n).allocate(risky, safe_rate, params)

    safe_series = ReturnMatrix(pd.DataFrame({"S": np.full(n, per_period)}, index=risky.dates))
    series_run = cppi(periods_per_year=n).allocate(risky, safe_series, params)

    assert np.allclose(
        scalar_run.equity_curve.to_numpy(), series_run.equity_curve.to_numpy(), rtol=1e-12
    )


def test_multi_column_risky_rejected() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    two = ReturnMatrix(pd.DataFrame({"A": np.zeros(5), "B": np.zeros(5)}, index=idx))
    with pytest.raises(BacktestError):
        cppi().allocate(two, 0.0, AllocatorParams())


def test_safe_series_length_mismatch_raises() -> None:
    risky = _risky(np.zeros(10))
    bad_safe = ReturnMatrix(
        pd.DataFrame({"S": np.zeros(5)}, index=pd.date_range("2020-01-01", periods=5, freq="B"))
    )
    with pytest.raises(BacktestError):
        cppi().allocate(risky, bad_safe, AllocatorParams())


def test_invalid_params_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        run_cppi_path(np.array([0.0]), np.array([0.0]), multiplier=-1.0, floor_fraction=0.8)
    with pytest.raises(ValidationError):
        run_cppi_path(np.array([0.0]), np.array([0.0]), multiplier=3.0, floor_fraction=1.5)


def test_seeded_sweep_floor_invariant() -> None:
    # Deterministic seeded sweep (hypothesis not installed): across many random
    # paths and parameter combos, the fixed-floor CPPI never breaches its floor.
    rng = np.random.default_rng(123456)
    for _ in range(200):
        n = int(rng.integers(20, 200))
        mu = rng.uniform(-0.01, 0.01)
        sigma = rng.uniform(0.005, 0.05)
        risky = rng.normal(mu, sigma, size=n)
        m = float(rng.uniform(1.0, 5.0))
        floor_frac = float(rng.uniform(0.0, 0.95))
        safe_rate = float(rng.uniform(0.0, 0.05))
        per_period_safe = (1.0 + safe_rate) ** (1.0 / 252) - 1.0
        hist = run_cppi_path(
            risky,
            np.full(n, per_period_safe),
            multiplier=m,
            floor_fraction=floor_frac,
            start_value=1.0,
            max_drawdown=None,
        )
        # Single-step jumps cannot exceed 1/m for the cushion linearization to
        # hold; the smooth synthetic returns above stay well inside that bound, so
        # the floor must hold to numerical tolerance.
        assert np.all(hist.account >= floor_frac - 1e-8)


def test_seeded_sweep_is_deterministic() -> None:
    def run() -> np.ndarray:
        rng = np.random.default_rng(99)
        risky = rng.normal(0.0, 0.02, size=128)
        return run_cppi_path(risky, np.zeros(128), multiplier=3.0, floor_fraction=0.7).account

    assert np.array_equal(run(), run())
