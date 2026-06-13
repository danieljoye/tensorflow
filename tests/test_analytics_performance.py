"""Tests for analytics/performance.py against hand-computed references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.analytics.performance import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from riskbudget.core.errors import ValidationError


def test_annualized_return_compound_reference() -> None:
    # Two periods, ppy=2 ⇒ one "year": geometric annualization == cumulative growth.
    r = pd.Series([0.10, -0.05])
    growth = 1.10 * 0.95
    assert annualized_return(r, periods_per_year=2) == pytest.approx(growth - 1.0)


def test_annualized_return_grosses_up() -> None:
    # Constant 1% over 4 periods, ppy=4 (one year) ⇒ (1.01)^4 - 1.
    r = pd.Series([0.01] * 4)
    assert annualized_return(r, periods_per_year=4) == pytest.approx(1.01**4 - 1.0)


def test_annualized_volatility_reference() -> None:
    r = np.array([0.01, -0.01, 0.02, -0.02])
    expected = float(np.std(r)) * np.sqrt(12)
    assert annualized_volatility(r, periods_per_year=12) == pytest.approx(expected)


def test_sharpe_zero_rf_matches_ratio() -> None:
    r = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
    expected = annualized_return(r, 252) / annualized_volatility(r, 252)
    assert sharpe_ratio(r, 0.0, 252) == pytest.approx(expected)


def test_sharpe_nan_on_zero_vol() -> None:
    r = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(r))


def test_sortino_uses_downside_only() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    val = sortino_ratio(r, 0.0, 12)
    assert np.isfinite(val)


def test_sortino_nan_when_no_downside() -> None:
    r = pd.Series([0.01, 0.02, 0.0])
    assert np.isnan(sortino_ratio(r))


def test_calmar_ratio_reference() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    ann = annualized_return(r, 3)
    # Worst drawdown of wealth [1.1, 0.88, 0.924] from peak 1.1 ⇒ -0.2.
    assert calmar_ratio(r, 3) == pytest.approx(ann / 0.2)


def test_calmar_nan_without_drawdown() -> None:
    r = pd.Series([0.01, 0.02, 0.03])  # monotonically up
    assert np.isnan(calmar_ratio(r))


def test_hit_rate() -> None:
    r = pd.Series([0.01, -0.02, 0.0, 0.03])
    assert hit_rate(r) == pytest.approx(0.5)


def test_average_turnover_reference() -> None:
    w = pd.DataFrame(
        {"A": [0.5, 0.7], "B": [0.5, 0.3]},
        index=pd.RangeIndex(2),
    )
    # One-way turnover = 0.5*(|0.2|+|0.2|) = 0.2.
    assert average_turnover(w) == pytest.approx(0.2)


def test_average_turnover_single_row_is_zero() -> None:
    w = pd.DataFrame({"A": [1.0]})
    assert average_turnover(w) == 0.0


def test_average_turnover_treats_nan_as_zero() -> None:
    w = pd.DataFrame({"A": [1.0, np.nan], "B": [np.nan, 1.0]})
    # A: 1->0, B: 0->1 ⇒ one-way = 0.5*(1+1)=1.0.
    assert average_turnover(w) == pytest.approx(1.0)


def test_empty_returns_raises() -> None:
    with pytest.raises(ValidationError):
        annualized_return(pd.Series([], dtype=float))


def test_nonfinite_returns_raises() -> None:
    with pytest.raises(ValidationError):
        annualized_volatility(np.array([0.1, np.inf]))


def test_bad_ppy_raises() -> None:
    with pytest.raises(ValidationError):
        annualized_return(pd.Series([0.01]), periods_per_year=0)
