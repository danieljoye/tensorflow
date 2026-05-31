"""Tests for the covariance estimators + universal PSD fix (Agent 3).

Covers ``sample``, ``ewma``, ``semicov``, ``ledoit_wolf`` (both targets), ``oas``,
and ``pca``, plus the nearest-PSD repair (BUILD_PLAN §12.3) and a property-based
invariant that every estimator output is symmetric and PSD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import RiskModelError
from riskbudget.core.interfaces import RiskModel
from riskbudget.core.types import ReturnMatrix
from riskbudget.riskmodel import (
    RISK_MODEL_FACTORIES,
    ewma_covariance,
    is_psd,
    ledoit_wolf,
    nearest_psd,
    oas_shrinkage,
    pca_factor_model,
    sample_covariance,
    semi_covariance,
)


def _is_symmetric(m: np.ndarray) -> bool:
    return bool(np.allclose(m, m.T, atol=1e-12))


def _is_psd_eig(m: np.ndarray) -> bool:
    return bool(np.linalg.eigvalsh(0.5 * (m + m.T)).min() >= -1e-10)


def _random_returns(rng: np.random.Generator, n_assets: int, n_periods: int) -> ReturnMatrix:
    cov = rng.standard_normal((n_assets, n_assets))
    cov = cov @ cov.T / n_assets + np.eye(n_assets) * 1e-3
    draws = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_periods)
    assets = [f"a{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    return ReturnMatrix(pd.DataFrame(draws, index=dates, columns=assets))


# ---------------------------------------------------------------------------
# PSD fix (BUILD_PLAN §12.3)
# ---------------------------------------------------------------------------


def test_nearest_psd_repairs_indefinite_matrix() -> None:
    """A deliberately indefinite matrix becomes PSD after the spectral fix."""
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    assert not is_psd(indefinite)
    repaired = nearest_psd(indefinite)
    assert _is_symmetric(repaired)
    # The negative eigenvalue is clipped to (essentially) zero (§12.3 invariant).
    assert np.linalg.eigvalsh(repaired).min() >= -1e-12


def test_nearest_psd_makes_strictly_indefinite_input_psd() -> None:
    """A non-singular-after-repair indefinite matrix becomes Cholesky-PSD."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((5, 5))
    sym = 0.5 * (base + base.T)  # symmetric, generically indefinite
    assert not is_psd(sym)
    repaired = nearest_psd(sym)
    assert is_psd(repaired)
    assert np.linalg.eigvalsh(repaired).min() >= -1e-12


def test_nearest_psd_leaves_psd_matrix_essentially_unchanged() -> None:
    psd = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert is_psd(psd)
    assert np.allclose(nearest_psd(psd), psd, atol=1e-12)


def test_nearest_psd_rejects_non_square() -> None:
    with pytest.raises(RiskModelError):
        nearest_psd(np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Per-estimator behaviour
# ---------------------------------------------------------------------------


def test_sample_covariance_matches_numpy(return_matrix: ReturnMatrix) -> None:
    cov = sample_covariance(periods_per_year=1, ddof=1).estimate(return_matrix)
    ref = np.cov(return_matrix.values, rowvar=False, ddof=1)
    assert np.allclose(cov, ref, atol=1e-10)


def test_ewma_recent_weighting_differs_from_sample(return_matrix: ReturnMatrix) -> None:
    ew = ewma_covariance(span=20, periods_per_year=1).estimate(return_matrix)
    sample = sample_covariance(periods_per_year=1).estimate(return_matrix)
    # Different weighting -> different matrix, but same shape and PSD.
    assert ew.shape == sample.shape
    assert not np.allclose(ew, sample)
    assert _is_psd_eig(ew)


def test_semicov_only_uses_downside(return_matrix: ReturnMatrix) -> None:
    semi = semi_covariance(benchmark=0.0, periods_per_year=1).estimate(return_matrix)
    # Manual: drops = min(r, 0); S = dropsᵀdrops / T.
    vals = return_matrix.values
    drops = np.minimum(vals, 0.0)
    ref = (drops.T @ drops) / vals.shape[0]
    assert np.allclose(semi, ref, atol=1e-10)
    assert _is_psd_eig(semi)


def test_pca_reconstructs_low_rank_plus_noise() -> None:
    """One strong factor + idiosyncratic noise is recovered within tolerance."""
    rng = np.random.default_rng(7)
    n_assets, n_periods = 6, 4000
    loadings = rng.uniform(0.5, 1.5, size=n_assets)
    factor = rng.standard_normal(n_periods)
    idio = rng.standard_normal((n_periods, n_assets)) * 0.1
    data = np.outer(factor, loadings) + idio
    assets = [f"a{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    rmat = ReturnMatrix(pd.DataFrame(data, index=dates, columns=assets))

    pca_cov = pca_factor_model(n_factors=1, periods_per_year=1).estimate(rmat)
    sample = sample_covariance(periods_per_year=1).estimate(rmat)
    # The PCA reconstruction is close to the sample covariance (one true factor).
    assert np.allclose(np.diag(pca_cov), np.diag(sample), atol=1e-8)
    assert np.linalg.norm(pca_cov - sample) / np.linalg.norm(sample) < 0.1
    assert _is_psd_eig(pca_cov)


def test_pca_n_factors_clipped_to_universe(return_matrix: ReturnMatrix) -> None:
    # n_factors > N falls back to full rank without error.
    cov = pca_factor_model(n_factors=99, periods_per_year=1).estimate(return_matrix)
    assert cov.shape == (3, 3)
    assert _is_psd_eig(cov)


# ---------------------------------------------------------------------------
# Shrinkage: lowers condition number on ill-conditioned input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        ledoit_wolf(target="constant_correlation", periods_per_year=1),
        ledoit_wolf(target="constant_variance", periods_per_year=1),
        oas_shrinkage(periods_per_year=1),
    ],
)
def test_shrinkage_reduces_condition_number(factory: RiskModel) -> None:
    """On near-singular input (N close to T) shrinkage cuts the condition number."""
    rng = np.random.default_rng(11)
    rmat = _random_returns(rng, n_assets=25, n_periods=30)
    sample = sample_covariance(periods_per_year=1).estimate(rmat)
    shrunk = factory.estimate(rmat)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)
    assert _is_psd_eig(shrunk)


def test_shrinkage_intensity_in_unit_interval() -> None:
    rng = np.random.default_rng(13)
    rmat = _random_returns(rng, n_assets=10, n_periods=60)
    lwcc = ledoit_wolf(target="constant_correlation", periods_per_year=1)
    lwcc.estimate(rmat)
    assert lwcc.shrinkage_ is not None
    assert 0.0 <= lwcc.shrinkage_ <= 1.0


def test_unknown_ledoit_wolf_target_raises() -> None:
    with pytest.raises(RiskModelError):
        ledoit_wolf(target="nonsense")


# ---------------------------------------------------------------------------
# Property-based invariant: every estimator output is symmetric + PSD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RISK_MODEL_FACTORIES))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_every_estimator_is_symmetric_and_psd(name: str, seed: int) -> None:
    """Invariant over random fixtures: output symmetric, PSD, finite, ordered."""
    rng = np.random.default_rng(1000 + seed)
    n_assets = int(rng.integers(2, 8))
    n_periods = int(rng.integers(n_assets + 2, 80))
    rmat = _random_returns(rng, n_assets, n_periods)
    model = RISK_MODEL_FACTORIES[name]()
    assert isinstance(model, RiskModel)
    cov = model.estimate(rmat)
    assert cov.shape == (n_assets, n_assets)
    assert np.isfinite(cov).all()
    assert _is_symmetric(cov)
    assert is_psd(cov)


def test_estimators_satisfy_protocol() -> None:
    for factory in RISK_MODEL_FACTORIES.values():
        assert isinstance(factory(), RiskModel)


# ---------------------------------------------------------------------------
# Optional cross-check vs. PyPortfolioOpt (skipped when not installed)
# ---------------------------------------------------------------------------


def test_crosscheck_pypfopt_ledoit_wolf() -> None:
    """Our constant-variance LW matches PyPortfolioOpt's LW (same sklearn engine)."""
    pytest.importorskip("pypfopt")
    from pypfopt.risk_models import CovarianceShrinkage

    rng = np.random.default_rng(99)
    rmat = _random_returns(rng, n_assets=8, n_periods=200)
    frame = rmat.frame
    # PyPortfolioOpt works on prices by default; pass returns_data=True.
    ref = CovarianceShrinkage(frame, returns_data=True, frequency=1).ledoit_wolf(
        shrinkage_target="constant_variance"
    )
    ours = ledoit_wolf(target="constant_variance", periods_per_year=1).estimate(rmat)
    assert np.allclose(ours, ref.to_numpy(), atol=1e-6)


def test_too_few_periods_raises() -> None:
    rng = np.random.default_rng(0)
    one_row = ReturnMatrix(
        pd.DataFrame(
            rng.standard_normal((1, 3)),
            index=pd.date_range("2020-01-01", periods=1),
            columns=["a", "b", "c"],
        )
    )
    with pytest.raises(RiskModelError):
        sample_covariance().estimate(one_row)
