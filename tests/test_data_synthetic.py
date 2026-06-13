"""Tests for the synthetic correlated-returns data source (Agent 2).

Covers: covariance recovery for large N, determinism, factor-structure
covariance, the ``DataSource`` protocol, error handling, and a property-based
invariant (sample covariance converges to target across random seeds/specs).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import DataError
from riskbudget.core.interfaces import DataSource
from riskbudget.core.types import PriceData, ReturnMatrix
from riskbudget.data.synthetic import (
    SyntheticDataSource,
    covariance_from_factors,
    generate_correlated_returns,
    returns_to_prices,
    synthetic_data_source,
)

# A per-period (daily-scale) covariance so compounded prices stay positive.
_COV = (
    np.array(
        [
            [0.04, 0.006, 0.0],
            [0.006, 0.09, -0.012],
            [0.0, -0.012, 0.16],
        ]
    )
    / 252.0
)
_ASSETS = ["AAA", "BBB", "CCC"]


def _random_psd(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random symmetric positive-definite covariance of size n."""
    a = rng.standard_normal((n, n))
    return (a @ a.T) / n + np.eye(n) * 1e-3


# ---------------------------------------------------------------------------
# Covariance recovery
# ---------------------------------------------------------------------------


def test_recovers_known_covariance_large_sample() -> None:
    rm = generate_correlated_returns(n_periods=50_000, cov=_COV, assets=_ASSETS, seed=42)
    sample = np.cov(rm.values, rowvar=False)
    assert np.max(np.abs(sample - _COV)) < 1e-4


def test_recovers_mean() -> None:
    mu = np.array([0.001, -0.0005, 0.002])
    rm = generate_correlated_returns(n_periods=80_000, cov=_COV, assets=_ASSETS, mu=mu, seed=11)
    assert np.allclose(rm.values.mean(axis=0), mu, atol=2e-4)


def test_factor_structure_covariance_matches_formula() -> None:
    loadings = np.array([[1.0, 0.2], [0.8, -0.3], [0.5, 0.9]])
    fvols = np.array([0.1, 0.05])
    evols = np.array([0.02, 0.03, 0.015])
    sigma = covariance_from_factors(loadings, fvols, evols)
    expected = loadings @ np.diag(fvols**2) @ loadings.T + np.diag(evols**2)
    assert np.allclose(sigma, expected)
    # symmetric PSD
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_generate_from_factor_inputs_recovers_implied_cov() -> None:
    loadings = np.array([[1.0, 0.2], [0.8, -0.3], [0.5, 0.9]])
    fvols = np.array([0.1, 0.05])
    evols = np.array([0.02, 0.03, 0.015])
    target = covariance_from_factors(loadings, fvols, evols)
    rm = generate_correlated_returns(
        n_periods=60_000,
        loadings=loadings,
        factor_vols=fvols,
        idiosyncratic_vols=evols,
        seed=5,
    )
    sample = np.cov(rm.values, rowvar=False)
    assert np.max(np.abs(sample - target)) < 5e-4


def test_psd_only_factor_covariance_is_drawable() -> None:
    # K < N and zero idiosyncratic vol => rank-deficient (PSD, not PD) covariance.
    loadings = np.array([[1.0], [0.5], [0.25]])
    sigma = covariance_from_factors(loadings, [0.1], [0.0, 0.0, 0.0])
    assert np.linalg.matrix_rank(sigma) == 1
    rm = generate_correlated_returns(n_periods=1000, cov=sigma, seed=3)
    assert rm.shape == (1000, 3)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_output() -> None:
    a = generate_correlated_returns(n_periods=500, cov=_COV, assets=_ASSETS, seed=99)
    b = generate_correlated_returns(n_periods=500, cov=_COV, assets=_ASSETS, seed=99)
    assert np.array_equal(a.values, b.values)


def test_different_seed_different_output() -> None:
    a = generate_correlated_returns(n_periods=500, cov=_COV, assets=_ASSETS, seed=1)
    b = generate_correlated_returns(n_periods=500, cov=_COV, assets=_ASSETS, seed=2)
    assert not np.array_equal(a.values, b.values)


def test_generator_object_threads_through() -> None:
    rng = np.random.default_rng(7)
    a = generate_correlated_returns(n_periods=100, cov=_COV, assets=_ASSETS, seed=rng)
    rng2 = np.random.default_rng(7)
    b = generate_correlated_returns(n_periods=100, cov=_COV, assets=_ASSETS, seed=rng2)
    assert np.array_equal(a.values, b.values)


# ---------------------------------------------------------------------------
# DataSource protocol & price round-trip
# ---------------------------------------------------------------------------


def test_satisfies_datasource_protocol() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0)
    assert isinstance(src, DataSource)


def test_get_prices_returns_pricedata_and_recovers_returns() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0, mu=0.0003)
    prices = src.get_prices(["AAA", "BBB"], date(2000, 1, 3), date(2002, 1, 3))
    assert isinstance(prices, PriceData)
    assert prices.assets == ["AAA", "BBB"]
    # to_returns recovers the generated simple returns (panel has +1 leading row).
    rets = prices.to_returns("simple")
    assert rets.shape[0] == prices.shape[0] - 1


def test_get_prices_is_deterministic() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=123)
    p1 = src.get_prices(_ASSETS, date(2010, 1, 1), date(2011, 1, 1))
    p2 = src.get_prices(_ASSETS, date(2010, 1, 1), date(2011, 1, 1))
    assert np.array_equal(p1.values, p2.values)


def test_subuniverse_is_consistent_slice() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=55)
    full = src.get_prices(_ASSETS, date(2005, 1, 3), date(2006, 1, 3))
    sub = src.get_prices(["AAA"], date(2005, 1, 3), date(2006, 1, 3))
    assert np.allclose(full.frame["AAA"].to_numpy(), sub.frame["AAA"].to_numpy())


def test_returns_to_prices_roundtrip() -> None:
    rm = generate_correlated_returns(n_periods=200, cov=_COV, assets=_ASSETS, seed=8)
    prices = returns_to_prices(rm, initial_price=100.0)
    recovered = prices.to_returns("simple")
    assert np.allclose(recovered.values, rm.values, atol=1e-10)


def test_factory_callable() -> None:
    src = synthetic_data_source(cov=_COV, assets=_ASSETS, seed=0)
    assert isinstance(src, SyntheticDataSource)


def test_default_asset_names() -> None:
    rm = generate_correlated_returns(n_periods=10, cov=_COV, seed=0)
    assert rm.assets == ["ASSET_00", "ASSET_01", "ASSET_02"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_seed_none_rejected() -> None:
    with pytest.raises(DataError, match="explicit seed"):
        generate_correlated_returns(n_periods=10, cov=_COV, seed=None)  # type: ignore[arg-type]


def test_both_cov_and_factors_rejected() -> None:
    with pytest.raises(DataError, match="not both"):
        generate_correlated_returns(
            n_periods=10, cov=_COV, loadings=np.eye(3), factor_vols=[1, 1, 1], seed=0
        )


def test_neither_cov_nor_factors_rejected() -> None:
    with pytest.raises(DataError, match="factor triple"):
        generate_correlated_returns(n_periods=10, seed=0)


def test_non_psd_covariance_rejected() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite
    with pytest.raises(DataError, match="positive semi-definite"):
        generate_correlated_returns(n_periods=10, cov=bad, seed=0)


def test_non_symmetric_covariance_rejected() -> None:
    bad = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(DataError, match="symmetric"):
        generate_correlated_returns(n_periods=10, cov=bad, seed=0)


def test_n_periods_must_be_positive() -> None:
    with pytest.raises(DataError, match="n_periods"):
        generate_correlated_returns(n_periods=0, cov=_COV, seed=0)


def test_asset_count_mismatch_rejected() -> None:
    with pytest.raises(DataError, match="asset ids"):
        generate_correlated_returns(n_periods=10, cov=_COV, assets=["A", "B"], seed=0)


def test_get_prices_unknown_asset_rejected() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0)
    with pytest.raises(DataError, match="Unknown assets"):
        src.get_prices(["ZZZ"], date(2000, 1, 3), date(2001, 1, 3))


def test_get_prices_empty_assets_rejected() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0)
    with pytest.raises(DataError, match="non-empty"):
        src.get_prices([], date(2000, 1, 3), date(2001, 1, 3))


def test_get_prices_start_after_end_rejected() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0)
    with pytest.raises(DataError, match="must not be after"):
        src.get_prices(_ASSETS, date(2002, 1, 3), date(2001, 1, 3))


def test_get_prices_tiny_window_rejected() -> None:
    src = SyntheticDataSource(cov=_COV, assets=_ASSETS, seed=0)
    with pytest.raises(DataError, match="fewer than two"):
        src.get_prices(_ASSETS, date(2000, 1, 3), date(2000, 1, 3))


# ---------------------------------------------------------------------------
# Property-based invariant: sample covariance converges to the target
# (deterministic loop over random PD targets / seeds — hypothesis is unavailable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(8))
def test_property_sample_cov_converges(trial: int) -> None:
    rng = np.random.default_rng(1000 + trial)
    n = int(rng.integers(2, 6))
    target = _random_psd(n, rng) * 1e-3  # keep scale small for valid prices
    rm = generate_correlated_returns(n_periods=40_000, cov=target, seed=int(trial))
    sample = np.cov(rm.values, rowvar=False)
    # Relative Frobenius error shrinks with sample size; loose but meaningful bound.
    rel = np.linalg.norm(sample - target) / np.linalg.norm(target)
    assert rel < 0.1, f"trial {trial}: relative cov error {rel:.4f}"


@pytest.mark.parametrize("trial", range(6))
def test_property_output_shape_and_types(trial: int) -> None:
    rng = np.random.default_rng(2000 + trial)
    n = int(rng.integers(2, 5))
    n_periods = int(rng.integers(5, 50))
    target = _random_psd(n, rng) * 1e-4
    rm = generate_correlated_returns(n_periods=n_periods, cov=target, seed=trial)
    assert isinstance(rm, ReturnMatrix)
    assert rm.shape == (n_periods, n)
    assert isinstance(rm.dates, pd.DatetimeIndex)
    assert np.isfinite(rm.values).all()
