"""Tests for analytics/distribution.py: skewness, kurtosis, Jarque-Bera."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from riskbudget.analytics.distribution import is_normal, kurtosis, skewness
from riskbudget.core.errors import ValidationError


def test_skewness_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    r = rng.exponential(size=2000)  # right-skewed
    assert skewness(r) == pytest.approx(scipy.stats.skew(r), rel=1e-10)


def test_kurtosis_raw_matches_scipy_plus_three() -> None:
    rng = np.random.default_rng(2)
    r = rng.standard_normal(5000)
    # scipy.kurtosis default is Fisher (excess); raw == Fisher + 3.
    assert kurtosis(r, excess=False) == pytest.approx(scipy.stats.kurtosis(r) + 3.0, rel=1e-10)


def test_excess_kurtosis_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    r = rng.standard_normal(5000)
    assert kurtosis(r, excess=True) == pytest.approx(scipy.stats.kurtosis(r), rel=1e-10)


def test_symmetric_sample_has_zero_skew() -> None:
    r = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert skewness(r) == pytest.approx(0.0, abs=1e-12)


def test_is_normal_true_for_normal_sample() -> None:
    rng = np.random.default_rng(4)
    r = rng.standard_normal(2000)
    assert is_normal(r) is True


def test_is_normal_false_for_heavy_tailed() -> None:
    rng = np.random.default_rng(5)
    r = rng.standard_t(df=2, size=3000)  # fat tails ⇒ reject normality
    assert is_normal(r) is False


def test_constant_series_returns_nan_moment() -> None:
    assert np.isnan(skewness(pd.Series([1.0, 1.0, 1.0])))


def test_bad_level_raises() -> None:
    with pytest.raises(ValidationError):
        is_normal(pd.Series([0.1, 0.2]), level=2.0)


def test_empty_raises() -> None:
    with pytest.raises(ValidationError):
        kurtosis(pd.Series([], dtype=float))
