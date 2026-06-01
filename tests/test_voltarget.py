"""Tests for the volatility-targeting overlay (constructor + spec wiring)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import riskbudget as rb
from riskbudget.core.errors import OptimizationError
from riskbudget.optimize.voltarget import VolatilityTargetConstructor
from riskbudget.spec import StrategySpec

# A 3-asset covariance with very different vols (per-period); annualized ×12.
COV = np.array([[4e-4, 1e-4, 0.0], [1e-4, 9e-4, 2e-4], [0.0, 2e-4, 1.6e-3]])


class _EqualWeight:
    """Minimal inner constructor: equal long-only weights."""

    def construct(self, cov, *, mu=None, budget=None, constraints):
        n = cov.shape[0]
        from riskbudget.core.types import Portfolio

        return Portfolio({f"A{i}": 1.0 / n for i in range(n)})


def test_leverage_hits_target_exante() -> None:
    vt = VolatilityTargetConstructor(_EqualWeight(), target_volatility=0.10, periods_per_year=12)
    port = vt.construct(COV, constraints=None)
    # the scaled book's annualized ex-ante vol equals the 10% target
    sigma_annual = port.volatility(COV) * math.sqrt(12)
    assert sigma_annual == pytest.approx(0.10, rel=1e-6)
    assert port.leverage == pytest.approx(sum(abs(w) for w in port.weights.values()))


def test_leverage_capped() -> None:
    # tiny covariance -> would need huge leverage; cap binds
    tiny = COV * 1e-6
    vt = VolatilityTargetConstructor(
        _EqualWeight(), target_volatility=0.10, periods_per_year=12, max_leverage=2.5
    )
    port = vt.construct(tiny, constraints=None)
    assert port.leverage == pytest.approx(2.5, rel=1e-9)


def test_invalid_params_raise() -> None:
    with pytest.raises(OptimizationError):
        VolatilityTargetConstructor(_EqualWeight(), target_volatility=0.0, periods_per_year=12)
    with pytest.raises(OptimizationError):
        VolatilityTargetConstructor(_EqualWeight(), target_volatility=0.1, periods_per_year=0)


def test_spec_backtest_applies_targeting() -> None:
    common: dict = {
        "assets": ["A", "B", "C"],
        "data_source": {"name": "synthetic", "params": {"cov": COV.tolist()}},
        "risk_model": "sample",
        "schedule": {"frequency": "monthly", "lookback": 36},
        "periods_per_year": 12,
    }
    base = rb.backtest(StrategySpec(name="ERC", method="erc", **common))
    vt = rb.backtest(StrategySpec(name="ERC-vt", method="erc", target_volatility=0.10, **common))

    # targeting actively re-levers off 1.0 at some rebalance ...
    levs = [d["leverage"] for d in vt.diagnostics["per_rebalance"]]
    assert any(abs(lv - 1.0) > 1e-6 for lv in levs)
    # ... and changes the realized return series vs the untargeted run.
    assert not np.allclose(base.returns.to_numpy(), vt.returns.to_numpy())


def test_spec_without_target_is_fully_invested() -> None:
    spec = StrategySpec(
        name="ERC",
        method="erc",
        assets=["A", "B", "C"],
        data_source={"name": "synthetic", "params": {"cov": COV.tolist()}},
        risk_model="sample",
        schedule={"frequency": "monthly", "lookback": 36},
        periods_per_year=12,
    )
    res = rb.backtest(spec)
    levs = [d["leverage"] for d in res.diagnostics["per_rebalance"]]
    assert all(abs(lv - 1.0) < 1e-6 for lv in levs)
