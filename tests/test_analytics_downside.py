"""Tests for analytics/downside.py: drawdown, semideviation, VaR/CVaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.analytics.downside import (
    cvar_historic,
    drawdown,
    max_drawdown,
    semideviation,
    var_gaussian,
    var_historic,
)
from riskbudget.core.errors import ValidationError


def test_drawdown_columns_and_values() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    dd = drawdown(r)
    assert list(dd.columns) == ["wealth", "peak", "drawdown"]
    # wealth = [1.1, 0.88, 0.924]; peak = [1.1, 1.1, 1.1].
    np.testing.assert_allclose(dd["wealth"].to_numpy(), [1.1, 0.88, 0.924])
    np.testing.assert_allclose(dd["peak"].to_numpy(), [1.1, 1.1, 1.1])
    assert dd["drawdown"].min() == pytest.approx((0.88 - 1.1) / 1.1)


def test_drawdown_preserves_index() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    dd = drawdown(pd.Series([0.01, -0.02, 0.0], index=idx))
    assert dd.index.equals(idx)


def test_max_drawdown_reference() -> None:
    r = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(r) == pytest.approx((0.88 - 1.1) / 1.1)


def test_max_drawdown_zero_when_monotone_up() -> None:
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_semideviation_negatives_only() -> None:
    r = pd.Series([0.05, -0.02, 0.03, -0.04])
    expected = float(np.std(np.array([-0.02, -0.04])))
    assert semideviation(r) == pytest.approx(expected)


def test_semideviation_zero_without_negatives() -> None:
    assert semideviation(pd.Series([0.01, 0.02])) == 0.0


def test_var_historic_reference() -> None:
    r = pd.Series(np.linspace(-0.10, 0.10, 101))
    # 5th percentile of a symmetric linspace ⇒ ~ -0.09; VaR is its negation.
    assert var_historic(r, 0.05) == pytest.approx(-np.percentile(r.to_numpy(), 5.0))


def test_cvar_ge_var() -> None:
    rng = np.random.default_rng(7)
    r = pd.Series(rng.standard_normal(1000) * 0.02)
    assert cvar_historic(r, 0.05) >= var_historic(r, 0.05)


def test_var_gaussian_unmodified_matches_normal_quantile() -> None:
    from scipy.stats import norm

    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    z = norm.ppf(0.05)
    expected = -(np.mean(r.to_numpy()) + z * np.std(r.to_numpy()))
    assert var_gaussian(r, 0.05, modified=False) == pytest.approx(expected)


def test_cornish_fisher_ge_gaussian_on_left_skewed_fat_tailed() -> None:
    rng = np.random.default_rng(123)
    # Left-skewed, fat-tailed: mostly small positives with occasional crashes.
    base = rng.standard_normal(5000) * 0.005
    crashes = rng.standard_normal(5000)
    base[crashes < -2.0] -= 0.08  # inject a left tail
    r = pd.Series(base)
    cf = var_gaussian(r, 0.05, modified=True)
    gauss = var_gaussian(r, 0.05, modified=False)
    assert cf >= gauss


def test_bad_level_raises() -> None:
    with pytest.raises(ValidationError):
        var_historic(pd.Series([0.01, 0.02]), level=1.5)


def test_bad_start_value_raises() -> None:
    with pytest.raises(ValidationError):
        drawdown(pd.Series([0.01]), start_value=-1.0)


def test_invariant_cornish_fisher_sweep() -> None:
    """Deterministic seeded sweep: on left-skewed fat-tailed draws, CF VaR >= Gaussian."""
    rng = np.random.default_rng(2024)
    for _ in range(50):
        n = int(rng.integers(2000, 6000))
        x = rng.standard_normal(n) * rng.uniform(0.003, 0.01)
        tail = rng.standard_normal(n)
        x[tail < -1.8] -= rng.uniform(0.03, 0.12)
        r = pd.Series(x)
        cf = var_gaussian(r, 0.05, modified=True)
        gauss = var_gaussian(r, 0.05, modified=False)
        assert cf >= gauss - 1e-12
