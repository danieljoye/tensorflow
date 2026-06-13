"""Tests for the minimum-torsion transform and diversification constructors."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.core.errors import OptimizationError, RiskModelError
from riskbudget.core.interfaces import Constraints, PortfolioConstructor
from riskbudget.core.types import RiskBudget
from riskbudget.diversification.constructors import (
    FactorRiskBudgetConstructor,
    MaxENBConstructor,
    MostDiversifiedConstructor,
    factor_risk_budget,
    max_enb,
    mdp,
)
from riskbudget.diversification.metrics import (
    diversification_ratio,
    effective_number_of_bets,
)
from riskbudget.diversification.torsion import (
    approximate_torsion,
    minimum_torsion,
    pca_torsion,
    torsion,
)


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n + 4))
    cov = a @ a.T / (n + 4)
    return cov + np.eye(n) * 0.05


def _factor_corr(t: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Correlation matrix of the factors ``f = t·r`` (should be diagonal)."""
    fc = t @ cov @ t.T
    d = np.sqrt(np.diag(fc))
    return fc / np.outer(d, d)


# ---------------------------------------------------------------------------
# Minimum-torsion transform (§12.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 5, 8])
def test_minimum_torsion_factors_are_uncorrelated(n: int) -> None:
    """Min-torsion factors have ≈ 0 off-diagonal correlation (the core property)."""
    cov = _random_spd(n, seed=n * 11)
    t = minimum_torsion(cov)
    corr = _factor_corr(t, cov)
    off_diag = corr - np.diag(np.diag(corr))
    assert np.abs(off_diag).max() < 1e-8


@pytest.mark.parametrize("builder", [pca_torsion, approximate_torsion, minimum_torsion])
def test_all_torsions_decorrelate(builder) -> None:  # type: ignore[no-untyped-def]
    cov = _random_spd(6, seed=99)
    t = builder(cov)
    corr = _factor_corr(t, cov)
    off_diag = corr - np.diag(np.diag(corr))
    assert np.abs(off_diag).max() < 1e-8


def test_minimum_torsion_diagonal_cov_is_identity() -> None:
    """For uncorrelated assets the min-torsion transform is the identity."""
    cov = np.diag([0.04, 0.09, 0.16])
    t = minimum_torsion(cov)
    np.testing.assert_allclose(t, np.eye(3), atol=1e-8)


def test_minimum_torsion_closer_than_pca() -> None:
    """Min-torsion minimises tracking error: it is closer to identity than PCA."""
    cov = _random_spd(6, seed=42)
    s = np.sqrt(np.diag(cov))
    # Compare the volatility-normalised transforms in correlation space.
    t_mt = np.diag(1 / s) @ minimum_torsion(cov) @ np.diag(s)
    t_pca = np.diag(1 / s) @ pca_torsion(cov) @ np.diag(s)
    dist_mt = np.linalg.norm(t_mt - np.eye(6), "fro")
    dist_pca = np.linalg.norm(t_pca - np.eye(6), "fro")
    assert dist_mt < dist_pca


def test_torsion_unknown_method_raises() -> None:
    with pytest.raises(RiskModelError):
        torsion(np.eye(3), "nonsense")  # type: ignore[arg-type]


def test_torsion_non_symmetric_raises() -> None:
    bad = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(RiskModelError):
        minimum_torsion(bad)


# ---------------------------------------------------------------------------
# Most Diversified Portfolio
# ---------------------------------------------------------------------------


def test_mdp_implements_protocol() -> None:
    assert isinstance(mdp(), PortfolioConstructor)


def test_mdp_weights_valid() -> None:
    cov = _random_spd(5, seed=5)
    assets = [f"A{i}" for i in range(5)]
    p = MostDiversifiedConstructor().construct(
        cov, constraints=Constraints.long_only_fully_invested(), assets=assets
    )
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()


def test_mdp_beats_random_on_dr() -> None:
    """The MDP maximises the Diversification Ratio: it beats every random book."""
    cov = _random_spd(6, seed=17)
    assets = [f"A{i}" for i in range(6)]
    p = mdp().construct(cov, constraints=Constraints.long_only_fully_invested(), assets=assets)
    dr_mdp = diversification_ratio(p.as_array(assets), cov)

    rng = np.random.default_rng(0)
    for _ in range(500):
        w = rng.random(6)
        w /= w.sum()
        assert diversification_ratio(w, cov) <= dr_mdp + 1e-6


def test_mdp_rejects_shorting() -> None:
    cov = _random_spd(3, seed=1)
    with pytest.raises(OptimizationError):
        mdp().construct(cov, constraints=Constraints(long_only=False))


# ---------------------------------------------------------------------------
# Max-ENB
# ---------------------------------------------------------------------------


def test_max_enb_implements_protocol() -> None:
    assert isinstance(max_enb(), PortfolioConstructor)


def test_max_enb_beats_equal_weight_and_gmv() -> None:
    """Max-ENB ≥ ENB of the equal-weight and GMV books on a correlated fixture."""
    cov = _random_spd(6, seed=23)
    assets = [f"A{i}" for i in range(6)]
    con = Constraints.long_only_fully_invested()
    p = MaxENBConstructor().construct(cov, constraints=con, assets=assets)
    enb_max = effective_number_of_bets(p.as_array(assets), cov)

    eq = np.full(6, 1 / 6)
    inv = np.linalg.solve(cov, np.ones(6))
    gmv = inv / inv.sum()

    assert enb_max >= effective_number_of_bets(eq, cov) - 1e-6
    assert enb_max >= effective_number_of_bets(gmv, cov) - 1e-6


def test_max_enb_floor_feasible() -> None:
    cov = _random_spd(5, seed=8)
    assets = [f"A{i}" for i in range(5)]
    con = Constraints.long_only_fully_invested()
    p = MaxENBConstructor(enb_floor=2.0).construct(cov, constraints=con, assets=assets)
    assert effective_number_of_bets(p.as_array(assets), cov) >= 2.0 - 1e-4


def test_max_enb_floor_infeasible_raises() -> None:
    cov = _random_spd(4, seed=3)
    assets = [f"A{i}" for i in range(4)]
    con = Constraints.long_only_fully_invested()
    with pytest.raises(OptimizationError):
        MaxENBConstructor(enb_floor=10.0).construct(cov, constraints=con, assets=assets)


# ---------------------------------------------------------------------------
# Factor risk budget
# ---------------------------------------------------------------------------


def test_factor_risk_budget_implements_protocol() -> None:
    assert isinstance(factor_risk_budget(), PortfolioConstructor)


def test_factor_risk_budget_matches_target() -> None:
    """Factor risk contributions on the min-torsion basis match the budget."""
    cov = _random_spd(5, seed=31)
    assets = [f"A{i}" for i in range(5)]
    con = Constraints.long_only_fully_invested()
    p = FactorRiskBudgetConstructor().construct(cov, constraints=con, assets=assets)
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()

    # Recompute the realized factor risk contributions and check they are ≈ 1/N.
    t = minimum_torsion(cov)
    g = np.linalg.solve(t.T, w)
    v = np.clip(np.diag(t @ cov @ t.T), 0.0, None)
    frc = (g**2) * v
    p_frc = frc / frc.sum()
    np.testing.assert_allclose(p_frc, np.full(5, 0.2), atol=5e-3)


def test_factor_risk_budget_honors_custom_budget() -> None:
    cov = _random_spd(4, seed=12)
    assets = [f"A{i}" for i in range(4)]
    budget = RiskBudget.from_weights(dict(zip(assets, [0.4, 0.3, 0.2, 0.1], strict=True)))
    con = Constraints.long_only_fully_invested()
    p = factor_risk_budget().construct(cov, budget=budget, constraints=con, assets=assets)
    w = p.as_array(assets)
    assert w.sum() == pytest.approx(1.0)

    t = minimum_torsion(cov)
    g = np.linalg.solve(t.T, w)
    v = np.clip(np.diag(t @ cov @ t.T), 0.0, None)
    frc = (g**2) * v
    p_frc = frc / frc.sum()
    np.testing.assert_allclose(p_frc, np.array([0.4, 0.3, 0.2, 0.1]), atol=1e-2)


# ---------------------------------------------------------------------------
# Optional reckziegel/uncorbets cross-check (skipped if unavailable)
# ---------------------------------------------------------------------------


def test_optional_uncorbets_crosscheck() -> None:
    pytest.importorskip("uncorbets")
    import uncorbets  # type: ignore[import-not-found]  # noqa: F401

    # If the reference package is present, the min-torsion factors must still be
    # uncorrelated (the invariant the reference vectors encode).
    cov = _random_spd(5, seed=2)
    t = minimum_torsion(cov)
    corr = _factor_corr(t, cov)
    off = corr - np.diag(np.diag(corr))
    assert np.abs(off).max() < 1e-8
