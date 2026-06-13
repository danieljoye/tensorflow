"""Tests for Hierarchical Risk Parity (BUILD_PLAN §12.6)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.clustered.hrp import HRPConstructor, hrp
from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints, PortfolioConstructor


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n + 4))
    cov = a @ a.T / (n + 4)
    return cov + np.eye(n) * 0.05


def _block_cov() -> tuple[np.ndarray, list[str]]:
    """A 6-asset two-block covariance (two correlated clusters of 3)."""
    rng = np.random.default_rng(7)
    n_obs = 600
    # Two latent factors drive two blocks.
    f1 = rng.standard_normal(n_obs)
    f2 = rng.standard_normal(n_obs)
    cols = []
    for _ in range(3):
        cols.append(0.9 * f1 + 0.2 * rng.standard_normal(n_obs))
    for _ in range(3):
        cols.append(0.9 * f2 + 0.2 * rng.standard_normal(n_obs))
    data = np.column_stack(cols)
    cov = np.cov(data, rowvar=False)
    return cov, [f"A{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# Core HRP behaviour
# ---------------------------------------------------------------------------


def test_hrp_implements_protocol() -> None:
    assert isinstance(hrp(), PortfolioConstructor)


def test_hrp_weights_positive_and_sum_to_one() -> None:
    cov, assets = _block_cov()
    p = HRPConstructor().construct(
        cov, constraints=Constraints.long_only_fully_invested(), assets=assets
    )
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


@pytest.mark.parametrize("method", ["single", "ward", "average", "complete"])
def test_hrp_linkage_methods(method: str) -> None:
    cov, assets = _block_cov()
    con = Constraints.long_only_fully_invested()
    p = hrp(linkage_method=method).construct(cov, constraints=con, assets=assets)  # type: ignore[arg-type]
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_hrp_single_asset() -> None:
    cov = np.array([[0.04]])
    p = HRPConstructor().construct(
        cov, constraints=Constraints.long_only_fully_invested(), assets=["X"]
    )
    assert p.weights == {"X": 1.0}


def test_hrp_no_matrix_inversion_on_singular_cov() -> None:
    """HRP must run on a singular covariance (no inversion, §12.6)."""
    cov, assets = _block_cov()
    # Force a rank deficiency: make the last asset a copy of the first.
    cov[:, -1] = cov[:, 0]
    cov[-1, :] = cov[0, :]
    assert np.linalg.matrix_rank(cov) < cov.shape[0]  # genuinely singular

    p = HRPConstructor().construct(
        cov, constraints=Constraints.long_only_fully_invested(), assets=assets
    )
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert np.isfinite(w).all()


def test_hrp_inverse_variance_ordering() -> None:
    """Within a homogeneous-correlation block HRP tilts toward low-variance assets."""
    # Equicorrelated 3-asset block with increasing variances.
    rho = 0.5
    sds = np.array([0.1, 0.2, 0.3])
    corr = np.full((3, 3), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(sds, sds)
    assets = ["lo", "mid", "hi"]
    p = HRPConstructor().construct(
        cov, constraints=Constraints.long_only_fully_invested(), assets=assets
    )
    w = p.weights
    assert w["lo"] > w["mid"] > w["hi"]


def test_hrp_uses_budget_ordering() -> None:
    from riskbudget.core.types import RiskBudget

    cov, assets = _block_cov()
    budget = RiskBudget.equal(assets)
    p = HRPConstructor().construct(
        cov, budget=budget, constraints=Constraints.long_only_fully_invested()
    )
    assert set(p.assets) == set(assets)
    assert sum(p.weights.values()) == pytest.approx(1.0)


def test_hrp_rejects_shorting() -> None:
    cov, assets = _block_cov()
    with pytest.raises(OptimizationError):
        HRPConstructor().construct(cov, constraints=Constraints(long_only=False), assets=assets)


def test_hrp_non_square_raises() -> None:
    with pytest.raises(OptimizationError):
        HRPConstructor().construct(
            np.ones((3, 2)), constraints=Constraints.long_only_fully_invested()
        )


# ---------------------------------------------------------------------------
# Optional cross-check against pyportfolioopt's HRPOpt
# ---------------------------------------------------------------------------


def test_hrp_matches_pyportfolioopt() -> None:
    """Optional: cross-check HRP weights against pyportfolioopt's HRPOpt."""
    pypfopt = pytest.importorskip("pypfopt")
    import pandas as pd

    cov, assets = _block_cov()
    # Build a synthetic return panel whose sample cov matches our fixture cov.
    rng = np.random.default_rng(101)
    rets = rng.multivariate_normal(np.zeros(len(assets)), cov, size=2000)
    rdf = pd.DataFrame(rets, columns=assets)

    sample_cov = np.cov(rets, rowvar=False)
    ours = HRPConstructor().construct(
        sample_cov, constraints=Constraints.long_only_fully_invested(), assets=assets
    )

    ref = pypfopt.HRPOpt(returns=rdf)
    ref_weights = ref.optimize(linkage_method="single")
    ref_arr = np.array([ref_weights[a] for a in assets])
    ours_arr = ours.as_array(assets)

    # Same allocation philosophy → close weights (tolerant: distance metrics differ).
    np.testing.assert_allclose(ours_arr, ref_arr, atol=0.05)
