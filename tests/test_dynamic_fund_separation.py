"""Tests for the three-fund PSP/LHP/safe allocator (Agent 8, Martellini–Milhau 2012).

Covers the funding-ratio floor invariant on synthetic paths, the cushion/multiplier
math, the Allocator protocol, error taxonomy, and a deterministic seeded sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.interfaces import Allocator
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix
from riskbudget.dynamic.fund_separation import (
    FundSeparationAllocator,
    fund_separation,
    run_fund_separation_path,
)


def _psp(returns: np.ndarray) -> ReturnMatrix:
    idx = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    return ReturnMatrix(pd.DataFrame({"PSP": np.asarray(returns, dtype=float)}, index=idx))


def test_satisfies_allocator_protocol() -> None:
    assert isinstance(fund_separation(funding_floor=0.9), Allocator)
    assert isinstance(FundSeparationAllocator(funding_floor=0.9), Allocator)


def test_hand_worked_cushion_step() -> None:
    # start_assets=1.2, start_liab=1.0 -> F=1.2, floor=0.9 ->
    # cushion = 1 - 0.9/1.2 = 0.25; psp_weight = clip(2*0.25,0,1) = 0.5.
    hist = run_fund_separation_path(
        psp_returns=np.array([0.0]),
        lhp_returns=np.array([0.0]),
        liability_returns=np.array([0.0]),
        safe_returns=np.array([0.0]),
        multiplier=2.0,
        funding_floor=0.9,
        start_assets=1.2,
        start_liabilities=1.0,
    )
    assert hist.cushion[0] == pytest.approx(0.25)
    assert hist.psp_weight[0] == pytest.approx(0.5)


def test_funding_ratio_stays_above_floor_on_down_path() -> None:
    # PSP keeps falling; LHP tracks liabilities (both 0 here). The funding-ratio
    # floor logic must keep F >= floor (gap risk aside) on a smooth path.
    n = 300
    hist = run_fund_separation_path(
        psp_returns=np.full(n, -0.02),
        lhp_returns=np.zeros(n),
        liability_returns=np.zeros(n),
        safe_returns=np.zeros(n),
        multiplier=3.0,
        funding_floor=0.9,
        start_assets=1.3,
        start_liabilities=1.0,
    )
    assert np.all(hist.funding_ratio >= 0.9 - 1e-9)
    # De-risking: PSP weight shrinks as the cushion thins toward the floor.
    assert hist.psp_weight[-1] <= hist.psp_weight[0]


def test_funding_ratio_floor_with_growing_liabilities() -> None:
    # Liabilities grow each period; the LHP grows with them so the funding ratio is
    # protected. PSP is flat. Floor must still hold.
    n = 200
    hist = run_fund_separation_path(
        psp_returns=np.zeros(n),
        lhp_returns=np.full(n, 0.01),
        liability_returns=np.full(n, 0.01),
        safe_returns=np.zeros(n),
        multiplier=3.0,
        funding_floor=0.95,
        start_assets=1.2,
        start_liabilities=1.0,
    )
    assert np.all(hist.funding_ratio >= 0.95 - 1e-9)


def test_allocate_builds_consumable_result() -> None:
    rng = np.random.default_rng(11)
    psp = _psp(rng.normal(0.0005, 0.012, size=250))
    alloc = fund_separation(lhp=0.0, liabilities=0.0, funding_floor=0.9)
    res = alloc.allocate(psp, 0.0, AllocatorParams(multiplier=3.0, start_value=1.2))
    assert isinstance(res, BacktestResult)
    assert np.allclose(res.weights.sum(axis=1).to_numpy(), 1.0)
    assert "funding_ratio" in res.diagnostics
    assert res.metadata["strategy"] == "fund_separation"
    assert res.metrics["funding_floor_breached"] == 0.0
    assert res.metrics["min_funding_ratio"] >= 0.9 - 1e-9


def test_lhp_safe_blend_in_hedging_sleeve() -> None:
    # With lhp_safe_weight=1.0 the hedging sleeve is pure safe; PSP flat, safe
    # accrues -> assets grow only via the safe leg of the hedge sleeve.
    n = 50
    hist = run_fund_separation_path(
        psp_returns=np.zeros(n),
        lhp_returns=np.zeros(n),
        liability_returns=np.zeros(n),
        safe_returns=np.full(n, 0.001),
        multiplier=3.0,
        funding_floor=0.9,
        start_assets=1.2,
        start_liabilities=1.0,
        lhp_safe_weight=1.0,
    )
    # Hedging sleeve earns the safe rate on its (1 - psp_weight) share.
    assert hist.assets[-1] > 1.2


def test_initial_funding_ratio_below_floor_raises() -> None:
    with pytest.raises(ValidationError):
        run_fund_separation_path(
            psp_returns=np.zeros(5),
            lhp_returns=np.zeros(5),
            liability_returns=np.zeros(5),
            safe_returns=np.zeros(5),
            multiplier=2.0,
            funding_floor=1.5,
            start_assets=1.0,
            start_liabilities=1.0,
        )


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValidationError):
        run_fund_separation_path(
            psp_returns=np.zeros(5),
            lhp_returns=np.zeros(4),
            liability_returns=np.zeros(5),
            safe_returns=np.zeros(5),
            multiplier=2.0,
            funding_floor=0.9,
        )


def test_multi_column_psp_rejected() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    two = ReturnMatrix(pd.DataFrame({"A": np.zeros(5), "B": np.zeros(5)}, index=idx))
    with pytest.raises(BacktestError):
        fund_separation(funding_floor=0.9).allocate(two, 0.0, AllocatorParams(start_value=1.2))


def test_seeded_sweep_funding_floor_invariant() -> None:
    # Deterministic seeded sweep (hypothesis not installed): across many random PSP
    # paths and parameter combos, the funding ratio never breaches its floor when
    # liabilities/LHP are flat and the PSP path is smooth.
    rng = np.random.default_rng(987654)
    for _ in range(200):
        n = int(rng.integers(20, 200))
        sigma = rng.uniform(0.005, 0.04)
        psp = rng.normal(rng.uniform(-0.005, 0.005), sigma, size=n)
        m = float(rng.uniform(1.0, 5.0))
        floor_f = float(rng.uniform(0.5, 0.95))
        # Start strictly above the floor.
        start_assets = float(floor_f + rng.uniform(0.1, 0.5))
        hist = run_fund_separation_path(
            psp_returns=psp,
            lhp_returns=np.zeros(n),
            liability_returns=np.zeros(n),
            safe_returns=np.zeros(n),
            multiplier=m,
            funding_floor=floor_f,
            start_assets=start_assets,
            start_liabilities=1.0,
        )
        assert np.all(hist.funding_ratio >= floor_f - 1e-8)


def test_seeded_sweep_is_deterministic() -> None:
    def run() -> np.ndarray:
        rng = np.random.default_rng(303)
        psp = rng.normal(0.0, 0.02, size=128)
        return run_fund_separation_path(
            psp_returns=psp,
            lhp_returns=np.zeros(128),
            liability_returns=np.zeros(128),
            safe_returns=np.zeros(128),
            multiplier=3.0,
            funding_floor=0.9,
            start_assets=1.3,
            start_liabilities=1.0,
        ).funding_ratio

    assert np.array_equal(run(), run())
