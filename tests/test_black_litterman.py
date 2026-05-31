"""Tests for the Black-Litterman model (Agent 3, ``black_litterman.py``; §12.7).

Covers: empty views return the prior ``Π``; a confident view moves the posterior
toward ``Q``; the posterior is computed via a linear solve (no explicit inverse);
the Idzorek confidence mapping; implied weights; and the absolute/relative view
helpers. Includes a property-based invariant and a skippable PyPortfolioOpt
cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import RiskModelError
from riskbudget.core.interfaces import MeanModel
from riskbudget.core.types import ReturnMatrix
from riskbudget.riskmodel.black_litterman import (
    BlackLitterman,
    BLInputs,
    Views,
    black_litterman,
    default_omega,
    idzorek_omega,
    implied_prior_returns,
    market_implied_risk_aversion,
    posterior,
)

ASSETS = ["A", "B", "C"]
_COV_TRUE = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, -0.01], [0.0, -0.01, 0.16]], dtype=float)


@pytest.fixture
def bl_returns() -> ReturnMatrix:
    rng = np.random.default_rng(2024)
    draws = rng.multivariate_normal(np.zeros(3), _COV_TRUE, size=750)
    dates = pd.date_range("2018-01-01", periods=750, freq="B")
    return ReturnMatrix(pd.DataFrame(draws, index=dates, columns=ASSETS))


@pytest.fixture
def caps_inputs() -> BLInputs:
    return BLInputs(market_caps={"A": 3.0, "B": 2.0, "C": 1.0}, delta=2.5, tau=0.05)


# ---------------------------------------------------------------------------
# Core BL behaviour
# ---------------------------------------------------------------------------


def test_empty_views_returns_prior(bl_returns: ReturnMatrix, caps_inputs: BLInputs) -> None:
    """With no views the posterior expected returns equal the prior Π exactly."""
    bl = BlackLitterman(views=None, inputs=caps_inputs)
    post = bl.estimate_posterior(bl_returns)
    assert np.allclose(post.mu, post.prior)
    # And the MeanModel surface returns the same.
    er = bl.estimate(bl_returns)
    assert np.allclose(er.as_array(ASSETS), post.prior)


def test_confident_view_moves_posterior_toward_q(
    bl_returns: ReturnMatrix, caps_inputs: BLInputs
) -> None:
    """A highly-confident view drags the posterior toward the view return Q."""
    prior = BlackLitterman(views=None, inputs=caps_inputs).estimate_posterior(bl_returns).prior
    q_target = float(prior[0]) + 0.5  # a view well above the prior on asset A

    vague = Views.absolute({"A": q_target}, ASSETS, confidences={"A": 0.01})
    confident = Views.absolute({"A": q_target}, ASSETS, confidences={"A": 0.99})

    mu_vague = BlackLitterman(views=vague, inputs=caps_inputs).estimate_posterior(bl_returns).mu
    mu_conf = BlackLitterman(views=confident, inputs=caps_inputs).estimate_posterior(bl_returns).mu

    # Both move up toward Q; the confident one moves further and lands near Q.
    assert mu_conf[0] > mu_vague[0] > prior[0]
    assert abs(mu_conf[0] - q_target) < abs(mu_vague[0] - q_target)
    assert abs(mu_conf[0] - q_target) < 0.05


def test_posterior_uses_solve_not_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The posterior must be built with np.linalg.solve, never np.linalg.inv."""

    def _boom(*_a: object, **_k: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv must not be used in the BL posterior.")

    # Disable np.linalg.inv globally for the duration of the test; the posterior
    # must still compute (it uses solve/lstsq, never inv).
    monkeypatch.setattr(np.linalg, "inv", _boom)

    cov = _COV_TRUE.copy()
    prior = np.array([0.05, 0.07, 0.10])
    p = np.array([[1.0, 0.0, 0.0]])
    q = np.array([0.12])
    omega = default_omega(cov, p, 0.05)
    out = posterior(cov, prior, p, q, omega, 0.05)
    assert out.mu.shape == (3,)
    assert out.cov.shape == (3, 3)


def test_posterior_lstsq_fallback_on_singular_a() -> None:
    """A singular A-matrix falls back to lstsq without raising."""
    cov = _COV_TRUE.copy()
    prior = np.array([0.05, 0.07, 0.10])
    # Two identical views -> P rows collinear; with Ω=0 the A-matrix is singular.
    p = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    q = np.array([0.12, 0.12])
    omega = np.zeros((2, 2))
    out = posterior(cov, prior, p, q, omega, 0.05)
    assert np.isfinite(out.mu).all()


def test_implied_weights_sum_to_one(bl_returns: ReturnMatrix, caps_inputs: BLInputs) -> None:
    views = Views.absolute({"B": 0.15}, ASSETS)
    bl = BlackLitterman(views=views, inputs=caps_inputs)
    w = bl.implied_weights(bl_returns)
    assert set(w) == set(ASSETS)
    assert np.isclose(sum(w.values()), 1.0, atol=1e-9)


def test_posterior_covariance_is_psd(bl_returns: ReturnMatrix, caps_inputs: BLInputs) -> None:
    views = Views.absolute({"A": 0.1, "C": 0.2}, ASSETS)
    cov = BlackLitterman(views=views, inputs=caps_inputs).posterior_covariance(bl_returns)
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() >= -1e-10


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_market_implied_risk_aversion() -> None:
    delta = market_implied_risk_aversion(
        market_return=0.08, market_variance=0.04, risk_free_rate=0.0
    )
    assert np.isclose(delta, 2.0)
    with pytest.raises(RiskModelError):
        market_implied_risk_aversion(0.08, 0.0)


def test_implied_prior_reverse_optimization() -> None:
    w = np.array([0.5, 0.3, 0.2])
    pi = implied_prior_returns(_COV_TRUE, w, delta=2.5, risk_free_rate=0.01)
    assert np.allclose(pi, 2.5 * (_COV_TRUE @ w) + 0.01)


def test_default_omega_is_diagonal() -> None:
    p = np.eye(3)
    omega = default_omega(_COV_TRUE, p, 0.05)
    assert np.allclose(omega, np.diag(np.diag(omega)))
    assert np.all(np.diag(omega) > 0)


def test_idzorek_higher_confidence_means_smaller_omega() -> None:
    p = np.array([[1.0, 0.0, 0.0]])
    low = idzorek_omega(_COV_TRUE, p, 0.05, np.array([0.2]))
    high = idzorek_omega(_COV_TRUE, p, 0.05, np.array([0.9]))
    assert high[0, 0] < low[0, 0]


# ---------------------------------------------------------------------------
# View helpers + validation
# ---------------------------------------------------------------------------


def test_absolute_views_one_hot() -> None:
    views = Views.absolute({"B": 0.1, "A": 0.2}, ASSETS)
    assert views.n_views == 2
    assert views.P.shape == (2, 3)
    # Each row is one-hot on the named asset.
    assert np.array_equal(np.count_nonzero(views.P, axis=1), [1, 1])


def test_relative_view_picks_two_assets() -> None:
    views = Views.relative("A", "C", 0.05, ASSETS)
    assert views.P.shape == (1, 3)
    assert views.P[0, 0] == 1.0
    assert views.P[0, 2] == -1.0
    assert views.Q[0] == 0.05


def test_views_reject_unknown_asset() -> None:
    with pytest.raises(RiskModelError):
        Views.absolute({"ZZZ": 0.1}, ASSETS)


def test_views_reject_mismatched_universe(bl_returns: ReturnMatrix) -> None:
    bad = Views.absolute({"A": 0.1}, ["A", "B"])  # only 2 assets
    bl = BlackLitterman(views=bad, inputs=BLInputs(delta=2.5))
    with pytest.raises(RiskModelError):
        bl.estimate(bl_returns)


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(RiskModelError):
        Views.absolute({"A": 0.1}, ASSETS, confidences={"A": 1.5})


def test_bl_is_mean_model() -> None:
    assert isinstance(black_litterman(inputs=BLInputs(delta=2.5)), MeanModel)


def test_equal_weight_market_when_no_caps(bl_returns: ReturnMatrix) -> None:
    """Absent market caps, the prior uses equal market weights."""
    bl = BlackLitterman(views=None, inputs=BLInputs(delta=2.5))
    post = bl.estimate_posterior(bl_returns)
    assert np.isfinite(post.prior).all()


# ---------------------------------------------------------------------------
# Property-based invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_empty_views_invariant_over_random_data(seed: int) -> None:
    """For random fixtures, no-views posterior == prior and covariance == Σ-PSD."""
    rng = np.random.default_rng(500 + seed)
    n = int(rng.integers(2, 7))
    a = rng.standard_normal((n, n))
    cov = a @ a.T / n + np.eye(n) * 1e-3
    draws = rng.multivariate_normal(np.zeros(n), cov, size=300)
    assets = [f"a{i}" for i in range(n)]
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rmat = ReturnMatrix(pd.DataFrame(draws, index=dates, columns=assets))
    bl = BlackLitterman(views=None, inputs=BLInputs(delta=2.5))
    post = bl.estimate_posterior(rmat)
    assert np.allclose(post.mu, post.prior)
    assert np.linalg.eigvalsh(post.cov).min() >= -1e-10


# ---------------------------------------------------------------------------
# Optional cross-check vs. PyPortfolioOpt (skipped when not installed)
# ---------------------------------------------------------------------------


def test_crosscheck_pypfopt_prior() -> None:
    pytest.importorskip("pypfopt")
    from pypfopt.black_litterman import market_implied_prior_returns

    cov_df = pd.DataFrame(_COV_TRUE, index=ASSETS, columns=ASSETS)
    caps = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
    delta = 2.5
    ref = market_implied_prior_returns(caps, delta, cov_df, risk_free_rate=0.0)

    w = (caps / caps.sum()).to_numpy()
    ours = implied_prior_returns(_COV_TRUE, w, delta, 0.0)
    assert np.allclose(ours, ref.to_numpy(), atol=1e-10)
