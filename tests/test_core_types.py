"""Tests for the core data types (BUILD_PLAN §5).

The headline contract is the risk-contribution identity: total risk
contributions must sum to portfolio volatility ``Σᵢ TRCᵢ == σ(w)``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import (
    BacktestResult,
    Portfolio,
    PriceData,
    ReturnMatrix,
    RiskBudget,
)

# ---------------------------------------------------------------------------
# Risk contributions / volatility — the load-bearing math
# ---------------------------------------------------------------------------


def test_risk_contributions_sum_to_volatility(assets: list[str], cov_3: np.ndarray) -> None:
    pf = Portfolio({assets[0]: 0.5, assets[1]: 0.3, assets[2]: 0.2})
    rc = pf.risk_contributions(cov_3)
    total = sum(rc.values())
    assert total == pytest.approx(pf.volatility(cov_3), rel=1e-12, abs=1e-12)


def test_risk_contributions_match_closed_form(assets: list[str], cov_3: np.ndarray) -> None:
    w = np.array([0.5, 0.3, 0.2])
    vol = math.sqrt(float(w @ cov_3 @ w))
    expected = w * (cov_3 @ w) / vol
    pf = Portfolio(dict(zip(assets, w, strict=True)))
    rc = pf.risk_contributions(cov_3)
    np.testing.assert_allclose([rc[a] for a in assets], expected, rtol=1e-12)


def test_equal_weight_erc_known_2x2() -> None:
    # For a 2-asset equal-weight book with equal variances and zero correlation,
    # risk contributions are equal by symmetry.
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    pf = Portfolio({"X": 0.5, "Y": 0.5})
    rc = pf.risk_contributions(cov)
    assert rc["X"] == pytest.approx(rc["Y"])
    assert sum(rc.values()) == pytest.approx(pf.volatility(cov))


def test_zero_volatility_portfolio_returns_zero_contributions() -> None:
    cov = np.zeros((2, 2))
    pf = Portfolio({"X": 0.5, "Y": 0.5})
    rc = pf.risk_contributions(cov)
    assert rc == {"X": 0.0, "Y": 0.0}


def test_risk_contributions_respect_explicit_ordering(assets: list[str], cov_3: np.ndarray) -> None:
    pf = Portfolio({assets[0]: 0.5, assets[1]: 0.3, assets[2]: 0.2})
    reordered = list(reversed(assets))
    # Reorder cov to match the reordered asset list.
    idx = [assets.index(a) for a in reordered]
    cov_re = cov_3[np.ix_(idx, idx)]
    rc_default = pf.risk_contributions(cov_3)
    rc_re = pf.risk_contributions(cov_re, assets=reordered)
    for a in assets:
        assert rc_re[a] == pytest.approx(rc_default[a])


def test_volatility_rejects_non_symmetric_cov() -> None:
    pf = Portfolio({"X": 0.5, "Y": 0.5})
    bad = np.array([[0.04, 0.01], [0.02, 0.04]])
    with pytest.raises(ValidationError):
        pf.volatility(bad)


def test_volatility_rejects_wrong_shape() -> None:
    pf = Portfolio({"X": 0.5, "Y": 0.5})
    with pytest.raises(ValidationError):
        pf.volatility(np.eye(3))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def test_portfolio_leverage_and_net_exposure() -> None:
    pf = Portfolio({"X": 0.7, "Y": -0.3})
    assert pf.leverage == pytest.approx(1.0)
    assert pf.net_exposure == pytest.approx(0.4)


def test_portfolio_as_array_ordering() -> None:
    pf = Portfolio({"X": 0.7, "Y": 0.3})
    np.testing.assert_array_equal(pf.as_array(["Y", "X"]), np.array([0.3, 0.7]))


def test_portfolio_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        Portfolio({})


def test_portfolio_rejects_nonfinite() -> None:
    with pytest.raises(ValidationError):
        Portfolio({"X": float("nan")})


def test_portfolio_as_array_rejects_subset() -> None:
    pf = Portfolio({"X": 0.5, "Y": 0.5})
    with pytest.raises(ValidationError):
        pf.as_array(["X"])


# ---------------------------------------------------------------------------
# RiskBudget
# ---------------------------------------------------------------------------


def test_risk_budget_equal(assets: list[str]) -> None:
    b = RiskBudget.equal(assets)
    assert b.assets == assets
    for v in b.budgets.values():
        assert v == pytest.approx(1.0 / len(assets))


def test_risk_budget_requires_sum_to_one() -> None:
    # The contract requires budgets that already sum to 1.0; non-unit sums are
    # rejected rather than silently rescaled.
    with pytest.raises(ValidationError):
        RiskBudget({"X": 2.0, "Y": 2.0})


def test_risk_budget_accepts_unit_sum() -> None:
    b = RiskBudget({"X": 0.5, "Y": 0.5})
    assert sum(b.budgets.values()) == pytest.approx(1.0)
    assert b.budgets["X"] == pytest.approx(0.5)


def test_risk_budget_from_weights_normalizes() -> None:
    b = RiskBudget.from_weights({"X": 3.0, "Y": 1.0})
    assert b.budgets["X"] == pytest.approx(0.75)
    assert b.budgets["Y"] == pytest.approx(0.25)
    assert sum(b.budgets.values()) == pytest.approx(1.0)


def test_risk_budget_from_weights_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        RiskBudget.from_weights({"X": -1.0, "Y": 2.0})


def test_risk_budget_rejects_nonpositive() -> None:
    with pytest.raises(ValidationError):
        RiskBudget({"X": 0.0, "Y": 1.0})


def test_risk_budget_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        RiskBudget({})


def test_risk_budget_equal_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        RiskBudget.equal(["X", "X"])


def test_risk_budget_as_array(assets: list[str]) -> None:
    b = RiskBudget.equal(assets)
    arr = b.as_array(list(reversed(assets)))
    np.testing.assert_allclose(arr, np.full(len(assets), 1.0 / len(assets)))


# ---------------------------------------------------------------------------
# PriceData / ReturnMatrix
# ---------------------------------------------------------------------------


def test_price_data_to_returns_log() -> None:
    frame = pd.DataFrame(
        {"X": [100.0, 110.0, 121.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    pd_obj = PriceData(frame)
    rm = pd_obj.to_returns("log")
    assert rm.shape == (2, 1)
    np.testing.assert_allclose(rm.values[:, 0], [math.log(1.1), math.log(1.1)])


def test_price_data_to_returns_simple() -> None:
    frame = pd.DataFrame(
        {"X": [100.0, 110.0, 121.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    rm = PriceData(frame).to_returns("simple")
    np.testing.assert_allclose(rm.values[:, 0], [0.1, 0.1])


def test_price_data_rejects_nonpositive() -> None:
    frame = pd.DataFrame({"X": [100.0, 0.0]})
    with pytest.raises(ValidationError):
        PriceData(frame)


def test_price_data_to_returns_unknown_method() -> None:
    frame = pd.DataFrame({"X": [100.0, 110.0]})
    with pytest.raises(ValidationError):
        PriceData(frame).to_returns("geometric")  # type: ignore[arg-type]


def test_return_matrix_rejects_nan() -> None:
    frame = pd.DataFrame({"X": [0.01, float("nan")]})
    with pytest.raises(ValidationError):
        ReturnMatrix(frame)


def test_return_matrix_select(return_matrix: ReturnMatrix, assets: list[str]) -> None:
    sub = return_matrix.select([assets[1]])
    assert sub.assets == [assets[1]]
    assert sub.shape == (len(return_matrix), 1)


def test_return_matrix_sorted_by_date() -> None:
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    frame = pd.DataFrame({"X": [0.3, 0.1, 0.2]}, index=idx)
    rm = ReturnMatrix(frame)
    assert list(rm.dates) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    np.testing.assert_allclose(rm.values[:, 0], [0.1, 0.2, 0.3])


def test_synthetic_return_matrix_recovers_cov(
    return_matrix: ReturnMatrix, cov_3: np.ndarray
) -> None:
    sample = np.cov(return_matrix.values, rowvar=False)
    # Loose tolerance: 512 samples of a 3-asset MVN.
    np.testing.assert_allclose(sample, cov_3, atol=0.02)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


def test_backtest_result_basic() -> None:
    eq = pd.Series([1.0, 1.1, 1.05], index=pd.date_range("2020-01-01", periods=3))
    weights = pd.DataFrame(
        {"X": [0.5, 0.5], "Y": [0.5, 0.5]},
        index=pd.date_range("2020-01-01", periods=2, freq="MS"),
    )
    res = BacktestResult(equity_curve=eq, weights=weights, metrics={"sharpe": 1.2})
    assert res.assets == ["X", "Y"]
    assert res.start == eq.index[0]
    assert res.end == eq.index[-1]
    assert res.metrics["sharpe"] == 1.2


def test_backtest_result_rejects_bad_equity_curve() -> None:
    with pytest.raises(ValidationError):
        BacktestResult(equity_curve=[1.0, 2.0], weights=pd.DataFrame({"X": [1.0]}))  # type: ignore[arg-type]
