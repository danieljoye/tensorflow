"""Shared, deterministic test fixtures for the risk-budgeting system.

These fixtures are offline and reproducible (fixed seed / hand-written numbers)
so every agent's tests can build on a known synthetic covariance and return
matrix without a network or a data source implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.types import PriceData, ReturnMatrix

# A small, hand-chosen 3-asset covariance: symmetric, positive-definite, with
# distinct variances and non-trivial correlations. Kept tiny so risk-contribution
# math can be checked by hand if needed.
_COV_3 = np.array(
    [
        [0.04, 0.006, 0.0],
        [0.006, 0.09, -0.012],
        [0.0, -0.012, 0.16],
    ],
    dtype=float,
)
_ASSETS_3 = ["AAA", "BBB", "CCC"]


@pytest.fixture
def assets() -> list[str]:
    """The canonical 3-asset universe used across core tests."""
    return list(_ASSETS_3)


@pytest.fixture
def cov_3() -> np.ndarray:
    """A 3x3 symmetric positive-definite covariance matrix (annualized-ish)."""
    return _COV_3.copy()


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded NumPy generator for reproducible synthetic draws."""
    return np.random.default_rng(20240517)


@pytest.fixture
def return_matrix(assets: list[str], cov_3: np.ndarray, rng: np.random.Generator) -> ReturnMatrix:
    """A deterministic synthetic return matrix consistent with ``cov_3``.

    Draws 512 multivariate-normal periods with zero mean and covariance
    ``cov_3``. The sample covariance of ``.values`` is close (not equal) to
    ``cov_3``; tests that need the *exact* matrix should use the ``cov_3``
    fixture directly.
    """
    periods = 512
    draws = rng.multivariate_normal(mean=np.zeros(len(assets)), cov=cov_3, size=periods)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    return ReturnMatrix(pd.DataFrame(draws, index=dates, columns=assets))


@pytest.fixture
def price_data(assets: list[str], rng: np.random.Generator) -> PriceData:
    """A deterministic synthetic price panel (strictly positive, no gaps)."""
    periods = 256
    daily = rng.normal(loc=0.0003, scale=0.01, size=(periods, len(assets)))
    prices = 100.0 * np.exp(np.cumsum(daily, axis=0))
    dates = pd.date_range("2021-01-01", periods=periods, freq="B")
    return PriceData(pd.DataFrame(prices, index=dates, columns=assets))
