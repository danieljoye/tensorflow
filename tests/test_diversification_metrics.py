"""Tests for the Diversification Ratio and Effective Number of Bets (§12.5)."""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.core.errors import RiskModelError
from riskbudget.diversification.metrics import (
    diversification_distribution,
    diversification_ratio,
    effective_number_of_bets,
)
from riskbudget.diversification.torsion import TorsionMethod

_METHODS: tuple[TorsionMethod, ...] = ("minimum-torsion", "pca", "approximate")


def _random_spd(n: int, seed: int) -> np.ndarray:
    """A well-conditioned symmetric positive-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n + 4))
    cov = a @ a.T / (n + 4)
    return cov + np.eye(n) * 0.05


# ---------------------------------------------------------------------------
# Diversification Ratio
# ---------------------------------------------------------------------------


def test_dr_is_one_for_single_asset_book() -> None:
    cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    w = np.array([1.0, 0.0])
    assert diversification_ratio(w, cov) == pytest.approx(1.0)


def test_dr_is_one_when_perfectly_correlated() -> None:
    # Rank-1 correlation: any long-only book has DR == 1.
    s = np.array([0.2, 0.3, 0.1])
    cov = np.outer(s, s)  # ρ ≡ 1
    w = np.array([0.5, 0.3, 0.2])
    assert diversification_ratio(w, cov) == pytest.approx(1.0, abs=1e-8)


def test_dr_exceeds_one_when_diversified() -> None:
    cov = np.diag([0.04, 0.09, 0.16])  # uncorrelated
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    assert diversification_ratio(w, cov) > 1.0


def test_dr_is_scale_invariant() -> None:
    cov = _random_spd(4, 1)
    w = np.array([0.4, 0.3, 0.2, 0.1])
    assert diversification_ratio(w, cov) == pytest.approx(diversification_ratio(3.0 * w, cov))


# ---------------------------------------------------------------------------
# Effective Number of Bets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
@pytest.mark.parametrize("method", _METHODS)
def test_enb_equal_weight_over_k_independent_factors(k: int, method: TorsionMethod) -> None:
    """1/N over k independent unit-variance assets ⇒ ENB ≈ k (§12.5 sanity)."""
    cov = np.eye(k) * 0.04
    w = np.full(k, 1.0 / k)
    assert effective_number_of_bets(w, cov, method=method) == pytest.approx(k, rel=1e-6)


def test_enb_fully_correlated_pca_basis_is_one() -> None:
    """A rank-1 (fully correlated) book has a single principal factor ⇒ ENB ≈ 1.

    The "fully correlated ⇒ ENB ≈ 1" sanity (§12.5) is a property of the
    *principal-component* basis: the lone non-trivial PCA factor carries all the
    risk. The min-torsion basis deliberately stays close to the (here nearly
    identical) assets, so it keeps the bets spread — a known, intended contrast.
    """
    s = np.array([0.2, 0.25, 0.3, 0.15])
    cov = np.outer(s, s) + np.eye(4) * 1e-9  # near rank-1, just PD
    w = np.full(4, 0.25)
    assert effective_number_of_bets(w, cov, method="pca") == pytest.approx(1.0, abs=1e-3)


def test_enb_single_asset_on_pca_basis_is_one() -> None:
    """A single principal factor: holding only it gives ENB ≈ 1 on the PCA basis."""
    cov = np.diag([0.04, 0.09, 0.16])  # already principal axes
    w = np.array([1.0, 0.0, 0.0])
    assert effective_number_of_bets(w, cov, method="pca") == pytest.approx(1.0, abs=1e-6)


def test_diversification_distribution_sums_to_one() -> None:
    cov = _random_spd(6, 3)
    w = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    p = diversification_distribution(w, cov)
    assert float(np.sum(p)) == pytest.approx(1.0, abs=1e-10)


def test_passing_torsion_matrix_matches_method() -> None:
    from riskbudget.diversification.torsion import torsion

    cov = _random_spd(5, 9)
    w = np.full(5, 0.2)
    t = torsion(cov, "minimum-torsion")
    assert effective_number_of_bets(w, cov, t=t) == pytest.approx(
        effective_number_of_bets(w, cov, method="minimum-torsion")
    )


# ---------------------------------------------------------------------------
# Property-based: 1 <= ENB <= N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_enb_bounds_property(n: int) -> None:
    """ENB always lies in [1, N] for any long-only book on an SPD covariance.

    A property-style invariant test: sweeps many randomized covariances and
    weight vectors (deterministic seeds) and asserts the ``1 ≤ ENB ≤ N`` bound
    on every draw. (Implemented with a seeded loop rather than ``hypothesis``,
    which is not a project dependency.)
    """
    wrng = np.random.default_rng(1234 + n)
    for seed in range(40):
        cov = _random_spd(n, seed)
        w = wrng.random(n)
        w /= w.sum()
        enb = effective_number_of_bets(w, cov)
        assert 1.0 - 1e-6 <= enb <= n + 1e-6, (n, seed, enb)


# ---------------------------------------------------------------------------
# Error handling (§3.1 taxonomy)
# ---------------------------------------------------------------------------


def test_non_square_cov_raises() -> None:
    with pytest.raises(RiskModelError):
        diversification_ratio(np.ones(3), np.ones((3, 2)))


def test_zero_weight_book_variance_raises() -> None:
    cov = np.diag([0.04, 0.09])
    with pytest.raises(RiskModelError):
        diversification_ratio(np.zeros(2), cov)
