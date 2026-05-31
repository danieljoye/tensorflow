"""Tests for the expected-return estimators (Agent 3, ``returns_model.py``).

Each estimator implements ``MeanModel``; tests check protocol conformance,
asset alignment, the known-drift recovery on synthetic data, and the documented
annualization conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import RiskModelError
from riskbudget.core.interfaces import MeanModel
from riskbudget.core.types import ExpectedReturns, ReturnMatrix
from riskbudget.riskmodel import (
    MEAN_MODEL_FACTORIES,
    capm,
    ewma_mean,
    mean_historical,
    risk_based,
)


def _constant_drift_returns(
    drifts: dict[str, float], n_periods: int = 4000, scale: float = 0.001
) -> ReturnMatrix:
    """Synthetic returns whose per-asset mean equals the given (per-period) drift.

    Low noise (``scale``) and a long series so the sample / EWMA mean recovers the
    drift tightly enough for an exact-ish assertion.
    """
    rng = np.random.default_rng(42)
    cols = list(drifts)
    data = np.column_stack(
        [rng.normal(loc=d, scale=scale, size=n_periods) for d in drifts.values()]
    )
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="B")
    return ReturnMatrix(pd.DataFrame(data, index=dates, columns=cols))


def test_mean_models_satisfy_protocol() -> None:
    for factory in MEAN_MODEL_FACTORIES.values():
        assert isinstance(factory(), MeanModel)


def test_historical_arithmetic_recovers_drift() -> None:
    drifts = {"A": 0.0004, "B": 0.0010, "C": -0.0002}
    rmat = _constant_drift_returns(drifts)
    er = mean_historical(method="arithmetic", periods_per_year=252).estimate(rmat)
    assert er.assets == ["A", "B", "C"]
    got = er.as_array(["A", "B", "C"])
    expected = np.array([d * 252 for d in drifts.values()])
    assert np.allclose(got, expected, atol=0.02)


def test_historical_geometric_is_below_arithmetic_for_positive_vol() -> None:
    # Meaningful volatility so the drag (geo < arith) is clearly visible.
    drifts = {"A": 0.001, "B": 0.001}
    rmat = _constant_drift_returns(drifts, scale=0.02)
    arith = mean_historical(method="arithmetic", periods_per_year=252).estimate(rmat)
    geo = mean_historical(method="geometric", periods_per_year=252).estimate(rmat)
    # Geometric (CAGR) < arithmetic when volatility > 0 (volatility drag).
    assert np.all(geo.as_array() < arith.as_array())


def test_ewma_mean_recovers_drift_and_is_aligned() -> None:
    drifts = {"X": 0.0006, "Y": -0.0003}
    rmat = _constant_drift_returns(drifts)
    er = ewma_mean(span=2000, periods_per_year=252).estimate(rmat)
    assert isinstance(er, ExpectedReturns)
    assert np.allclose(er.as_array(["X", "Y"]), np.array([0.0006, -0.0003]) * 252, atol=0.04)


def test_capm_equal_weight_proxy_runs(return_matrix: ReturnMatrix) -> None:
    er = capm(risk_free_rate=0.02, periods_per_year=252).estimate(return_matrix)
    assert er.assets == return_matrix.assets
    assert np.isfinite(er.as_array()).all()


def test_capm_with_named_market() -> None:
    rng = np.random.default_rng(5)
    n = 1000
    market = rng.normal(0.0005, 0.01, n)
    # asset A has beta ~2 to the market, B has beta ~0.5, plus noise.
    a = 2.0 * market + rng.normal(0, 0.002, n)
    b = 0.5 * market + rng.normal(0, 0.002, n)
    frame = pd.DataFrame(
        {"A": a, "B": b, "MKT": market},
        index=pd.date_range("2015-01-01", periods=n, freq="B"),
    )
    rmat = ReturnMatrix(frame)
    er = capm(risk_free_rate=0.0, market="MKT", periods_per_year=252).estimate(rmat)
    mu = dict(zip(er.assets, er.as_array(), strict=True))
    # Higher-beta asset has the larger market-implied return.
    assert mu["A"] > mu["B"]


def test_capm_unknown_market_raises(return_matrix: ReturnMatrix) -> None:
    with pytest.raises(RiskModelError):
        capm(market="ZZZ").estimate(return_matrix)


def test_risk_based_orders_by_volatility() -> None:
    rng = np.random.default_rng(9)
    n = 2000
    # C is most volatile, A least.
    frame = pd.DataFrame(
        {
            "A": rng.normal(0, 0.005, n),
            "B": rng.normal(0, 0.010, n),
            "C": rng.normal(0, 0.020, n),
        },
        index=pd.date_range("2012-01-01", periods=n, freq="B"),
    )
    rmat = ReturnMatrix(frame)
    er = risk_based(proxy="volatility", target_return=0.1, periods_per_year=252).estimate(rmat)
    mu = er.as_array(["A", "B", "C"])
    assert mu[0] < mu[1] < mu[2]
    # Scaled so the cross-sectional mean equals target_return.
    assert np.isclose(mu.mean(), 0.1, atol=1e-9)


def test_risk_based_semideviation_proxy(return_matrix: ReturnMatrix) -> None:
    er = risk_based(proxy="semideviation", target_return=0.08).estimate(return_matrix)
    assert np.isclose(er.as_array().mean(), 0.08, atol=1e-9)


def test_invalid_method_raises() -> None:
    with pytest.raises(RiskModelError):
        mean_historical(method="bogus")
    with pytest.raises(RiskModelError):
        risk_based(proxy="bogus")
