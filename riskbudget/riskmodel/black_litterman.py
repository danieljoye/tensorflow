"""Black-Litterman views-based expected returns + covariance (BUILD_PLAN §12.7).

Blends a market-equilibrium prior with subjective views to produce posterior
expected returns far more stable than a raw sample mean — the natural ``μ`` source
for the MSR / efficient-frontier / mean-variance paths (Agent 4). v1 pipeline:

1. **Prior** by reverse optimization: ``Π = δ·Σ·w_mkt`` from market-cap weights and
   a market-implied risk aversion ``δ = (E[R_m] − r_f)/σ_m²``.
2. **Views** ``(P, Q, Ω)``: ``P`` picks the assets, ``Q`` the view returns, ``Ω``
   the view uncertainty (He–Litterman default ``Ω = diag(diag(τPΣPᵀ))``, or
   Idzorek per-view confidences ∈ [0, 1]).
3. **Posterior** returns and covariance via the He–Litterman master formula, solved
   as a **linear system** (``np.linalg.solve`` / ``lstsq`` fallback), never an
   explicit matrix inverse, for numerical stability.

The component is shaped like a :class:`~riskbudget.core.interfaces.MeanModel`
(``estimate(returns) -> ExpectedReturns``) and also exposes the posterior
covariance and implied weights. The :class:`Views` / :class:`BLInputs` structures
live **inside this module** (we do not modify ``core/``).

References: BUILD_PLAN §12.7; He & Litterman (1999/2002); Idzorek (2007);
PyPortfolioOpt ``BlackLittermanModel`` (§11).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.core.interfaces import RiskModel
from riskbudget.core.types import ExpectedReturns, ReturnMatrix
from riskbudget.riskmodel._common import (
    DEFAULT_PERIODS_PER_YEAR,
    finalize_cov,
    to_expected_returns,
    validate_periods_per_year,
)
from riskbudget.riskmodel.sample import SampleCovariance

DEFAULT_TAU = 0.05


# ---------------------------------------------------------------------------
# View / input structures (module-local; core/ is NOT modified)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Views:
    """A set of Black-Litterman views over a fixed, ordered asset universe.

    A view is a linear statement ``Pᵢ · μ = Qᵢ`` (± view uncertainty). The
    ``picking`` matrix ``P`` has shape ``(K, N)`` (``K`` views over ``N`` assets);
    ``Q`` has length ``K``. Optional per-view ``confidences`` ∈ ``(0, 1]`` enable
    the Idzorek (2007) ``Ω`` mapping.

    Use :meth:`absolute` for the common "asset X will return q" case, or
    :meth:`relative` for "asset X outperforms asset Y by q".

    Attributes
    ----------
    assets:
        Ordered universe the views are expressed over.
    P:
        Picking matrix, shape ``(K, N)``.
    Q:
        View returns, shape ``(K,)`` (annualized).
    confidences:
        Optional per-view confidence ∈ ``(0, 1]`` (Idzorek). ``None`` selects the
        He–Litterman default ``Ω``.
    """

    assets: Sequence[str]
    P: np.ndarray
    Q: np.ndarray
    confidences: np.ndarray | None = None

    def __post_init__(self) -> None:
        assets = [str(a) for a in self.assets]
        if not assets:
            raise RiskModelError("Views require a non-empty asset universe.")
        if len(set(assets)) != len(assets):
            raise RiskModelError("Views asset universe has duplicates.")
        p = np.atleast_2d(np.asarray(self.P, dtype=float))
        q = np.asarray(self.Q, dtype=float).ravel()
        if p.shape[1] != len(assets):
            raise RiskModelError(
                f"View picking matrix has {p.shape[1]} columns; expected {len(assets)} assets."
            )
        if p.shape[0] != q.shape[0]:
            raise RiskModelError(
                f"View count mismatch: P has {p.shape[0]} rows, Q has {q.shape[0]}."
            )
        if p.shape[0] == 0:
            raise RiskModelError("Views must contain at least one view.")
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            raise RiskModelError("View matrices contain non-finite values.")
        conf = self.confidences
        if conf is not None:
            conf_arr = np.asarray(conf, dtype=float).ravel()
            if conf_arr.shape[0] != q.shape[0]:
                raise RiskModelError("confidences length must match number of views.")
            if np.any(conf_arr <= 0) or np.any(conf_arr > 1):
                raise RiskModelError("confidences must lie in (0, 1].")
            object.__setattr__(self, "confidences", conf_arr)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "P", p)
        object.__setattr__(self, "Q", q)

    @property
    def n_views(self) -> int:
        """Number of views ``K``."""
        return int(self.P.shape[0])

    @classmethod
    def absolute(
        cls,
        views: Mapping[str, float],
        assets: Sequence[str],
        *,
        confidences: Mapping[str, float] | None = None,
    ) -> Views:
        """Build absolute views (``asset → return``) into a one-hot ``(P, Q)``.

        Parameters
        ----------
        views:
            ``asset_id -> expected annualized return`` (the view ``Q``).
        assets:
            The ordered universe (defines ``P``'s columns).
        confidences:
            Optional ``asset_id -> confidence`` ∈ ``(0, 1]`` (Idzorek).
        """
        asset_list = [str(a) for a in assets]
        index = {a: i for i, a in enumerate(asset_list)}
        rows = []
        q_vals = []
        conf_vals: list[float] = []
        for asset, q in views.items():
            asset = str(asset)
            if asset not in index:
                raise RiskModelError(f"View asset {asset!r} not in universe {asset_list}.")
            row = np.zeros(len(asset_list))
            row[index[asset]] = 1.0
            rows.append(row)
            q_vals.append(float(q))
            if confidences is not None:
                if asset not in confidences:
                    raise RiskModelError(f"Missing confidence for view asset {asset!r}.")
                conf_vals.append(float(confidences[asset]))
        if not rows:
            raise RiskModelError("absolute() requires at least one view.")
        conf_arr = np.array(conf_vals) if confidences is not None else None
        return cls(
            assets=asset_list,
            P=np.vstack(rows),
            Q=np.array(q_vals),
            confidences=conf_arr,
        )

    @classmethod
    def relative(
        cls,
        outperformer: str,
        underperformer: str,
        spread: float,
        assets: Sequence[str],
        *,
        confidence: float | None = None,
    ) -> Views:
        """Build a single relative view: ``outperformer − underperformer = spread``."""
        asset_list = [str(a) for a in assets]
        index = {a: i for i, a in enumerate(asset_list)}
        for a in (outperformer, underperformer):
            if str(a) not in index:
                raise RiskModelError(f"View asset {a!r} not in universe {asset_list}.")
        row = np.zeros(len(asset_list))
        row[index[str(outperformer)]] = 1.0
        row[index[str(underperformer)]] = -1.0
        conf_arr = None if confidence is None else np.array([float(confidence)])
        return cls(
            assets=asset_list,
            P=row.reshape(1, -1),
            Q=np.array([float(spread)]),
            confidences=conf_arr,
        )


@dataclass(frozen=True)
class BLInputs:
    """Configuration for the Black-Litterman model (BUILD_PLAN §12.7).

    Attributes
    ----------
    market_caps:
        Optional ``asset_id -> market cap`` (or any positive proxy). Normalized to
        market-cap weights ``w_mkt``. If ``None``, equal weights are used.
    tau:
        Scalar ``τ`` scaling the prior uncertainty ``τΣ``. Default ``0.05``.
    delta:
        Risk-aversion ``δ`` for the reverse-optimization prior. If ``None`` it is
        market-implied from ``market_return`` and ``market_variance`` (or, absent
        those, defaults to ``2.5``).
    risk_free_rate:
        Annualized ``r_f`` added to the prior ``Π``. Default ``0.0``.
    market_return, market_variance:
        Optional annualized market return / variance used to imply
        ``δ = (market_return − r_f)/market_variance`` when ``delta`` is ``None``.
    """

    market_caps: Mapping[str, float] | None = None
    tau: float = DEFAULT_TAU
    delta: float | None = None
    risk_free_rate: float = 0.0
    market_return: float | None = None
    market_variance: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau) or self.tau <= 0:
            raise RiskModelError("tau must be finite and positive.")
        if self.delta is not None and (not np.isfinite(self.delta) or self.delta <= 0):
            raise RiskModelError("delta must be finite and positive when set.")
        if not np.isfinite(self.risk_free_rate):
            raise RiskModelError("risk_free_rate must be finite.")
        if self.market_caps is not None:
            vals = list(self.market_caps.values())
            if any((not np.isfinite(v)) or v <= 0 for v in vals):
                raise RiskModelError("market_caps must be finite and strictly positive.")


# ---------------------------------------------------------------------------
# Pure-function master formula (testable in isolation)
# ---------------------------------------------------------------------------


def market_implied_risk_aversion(
    market_return: float, market_variance: float, risk_free_rate: float = 0.0
) -> float:
    """Market-implied risk aversion ``δ = (E[R_m] − r_f)/σ_m²`` (BUILD_PLAN §12.7)."""
    if not np.isfinite(market_variance) or market_variance <= 0:
        raise RiskModelError("market_variance must be finite and positive.")
    delta = (market_return - risk_free_rate) / market_variance
    if not np.isfinite(delta) or delta <= 0:
        raise RiskModelError("Market-implied risk aversion is non-positive; check inputs.")
    return float(delta)


def implied_prior_returns(
    cov: np.ndarray, w_mkt: np.ndarray, delta: float, risk_free_rate: float = 0.0
) -> np.ndarray:
    """Reverse-optimization prior ``Π = δ·Σ·w_mkt + r_f`` (BUILD_PLAN §12.7)."""
    pi: np.ndarray = delta * (cov @ w_mkt) + risk_free_rate
    return pi


def default_omega(cov: np.ndarray, p: np.ndarray, tau: float) -> np.ndarray:
    """He–Litterman default ``Ω = diag(diag(τ·P·Σ·Pᵀ))`` (BUILD_PLAN §12.7)."""
    m = tau * (p @ cov @ p.T)
    return np.diag(np.clip(np.diag(m), a_min=1e-16, a_max=None))


def idzorek_omega(
    cov: np.ndarray, p: np.ndarray, tau: float, confidences: np.ndarray
) -> np.ndarray:
    """Idzorek (2007) ``Ω`` from per-view confidences ∈ ``(0, 1]`` (BUILD_PLAN §12.7).

    For each view ``k``, ``Ω_kk = α_k · (τ · Pₖ Σ Pₖᵀ)`` with
    ``α_k = (1 − c_k)/c_k`` — so confidence ``1`` drives ``Ω → 0`` (the posterior
    is pulled fully onto the view) and low confidence inflates ``Ω``.
    """
    base = tau * np.diag(p @ cov @ p.T)
    alpha = (1.0 - confidences) / confidences
    diag = np.clip(alpha * base, a_min=1e-16, a_max=None)
    return np.diag(diag)


@dataclass
class BLPosterior:
    """Posterior outputs of the Black-Litterman master formula.

    Attributes
    ----------
    mu:
        Posterior expected returns ``E(R)``, shape ``(N,)``.
    cov:
        Posterior covariance ``Σ_post``, shape ``(N, N)`` (PSD-repaired).
    prior:
        The prior ``Π`` (returned unchanged when there are no views).
    omega:
        The view-uncertainty matrix ``Ω`` actually used.
    """

    mu: np.ndarray
    cov: np.ndarray
    prior: np.ndarray
    omega: np.ndarray


def posterior(
    cov: np.ndarray,
    prior: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float,
) -> BLPosterior:
    """He–Litterman posterior, solved as a linear system (BUILD_PLAN §12.7).

    ``τΣP = τ·Σ·Pᵀ``; ``A = P·(τΣP) + Ω``; ``b = Q − P·Π``; then
    ``E(R) = Π + (τΣP)·solve(A, b)`` and
    ``Σ_post = Σ + [τΣ − (τΣP)·solve(A, (τΣP)ᵀ)]``. No explicit inverse is formed;
    a least-squares fallback is used if ``A`` is singular.
    """
    n = cov.shape[0]
    tau_sigma = tau * cov
    tau_sigma_p = tau_sigma @ p.T  # τΣPᵀ, shape (N, K)
    a_mat = p @ tau_sigma_p + omega  # (K, K)
    b_vec = q - p @ prior  # (K,)

    sol_b = _solve(a_mat, b_vec)
    mu = prior + tau_sigma_p @ sol_b

    sol_m = _solve(a_mat, tau_sigma_p.T)  # (K, N)
    posterior_cov = cov + (tau_sigma - tau_sigma_p @ sol_m)
    posterior_cov = finalize_cov(posterior_cov, n_assets=n)

    return BLPosterior(mu=mu, cov=posterior_cov, prior=prior, omega=omega)


def _solve(a_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` as a linear system, with an ``lstsq`` fallback (no inverse)."""
    try:
        solved: np.ndarray = np.linalg.solve(a_mat, b)
        return solved
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(a_mat, b, rcond=None)
        return np.asarray(sol)


# ---------------------------------------------------------------------------
# MeanModel-shaped estimator
# ---------------------------------------------------------------------------


class BlackLitterman:
    """Black-Litterman posterior expected returns (implements ``MeanModel``).

    Estimates a covariance from the returns (via ``risk_model``, default sample
    covariance), builds the reverse-optimization prior from ``inputs``, blends in
    ``views``, and returns the posterior ``E(R)`` as :class:`ExpectedReturns`. The
    posterior covariance and implied weights are available via
    :meth:`estimate_posterior` and :meth:`implied_weights`.

    With **no views** (``views=None``) the posterior equals the prior ``Π`` exactly.

    Parameters
    ----------
    views:
        The :class:`Views` to blend in, or ``None`` for the pure prior.
    inputs:
        :class:`BLInputs` configuration (market caps, ``τ``, ``δ``, ``r_f``).
    risk_model:
        A :class:`~riskbudget.core.interfaces.RiskModel` for ``Σ``. Defaults to the
        annualized sample covariance.
    periods_per_year:
        Annualization factor used only for the equal-weight market-variance
        fallback when implying ``δ``.
    """

    def __init__(
        self,
        *,
        views: Views | None = None,
        inputs: BLInputs | None = None,
        risk_model: RiskModel | None = None,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        self.views = views
        self.inputs = inputs if inputs is not None else BLInputs()
        self.risk_model: RiskModel = risk_model if risk_model is not None else SampleCovariance()
        self.periods_per_year = validate_periods_per_year(periods_per_year)

    # -- helpers ----------------------------------------------------------------

    def _market_weights(self, assets: list[str]) -> np.ndarray:
        caps = self.inputs.market_caps
        if caps is None:
            return np.full(len(assets), 1.0 / len(assets))
        missing = [a for a in assets if a not in caps]
        if missing:
            raise RiskModelError(f"market_caps missing assets: {missing}.")
        vec = np.array([float(caps[a]) for a in assets])
        total = float(vec.sum())
        if total <= 0:
            raise RiskModelError("market_caps sum to a non-positive value.")
        weights: np.ndarray = vec / total
        return weights

    def _resolve_delta(self) -> float:
        if self.inputs.delta is not None:
            return float(self.inputs.delta)
        if self.inputs.market_return is not None and self.inputs.market_variance is not None:
            return market_implied_risk_aversion(
                self.inputs.market_return,
                self.inputs.market_variance,
                self.inputs.risk_free_rate,
            )
        # No market figures supplied: fall back to a standard δ = 2.5.
        return 2.5

    def _build(self, returns: ReturnMatrix) -> tuple[BLPosterior, list[str], np.ndarray, float]:
        if not isinstance(returns, ReturnMatrix):
            raise RiskModelError("BlackLitterman requires a ReturnMatrix input.")
        assets = returns.assets
        cov = np.asarray(self.risk_model.estimate(returns), dtype=np.float64)
        if cov.shape != (len(assets), len(assets)):
            raise RiskModelError("Risk-model covariance does not match the asset count.")

        w_mkt = self._market_weights(assets)
        delta = self._resolve_delta()
        prior = implied_prior_returns(cov, w_mkt, delta, self.inputs.risk_free_rate)
        tau = self.inputs.tau

        if self.views is None:
            # No views: posterior == prior; posterior covariance == Σ.
            post = BLPosterior(
                mu=prior,
                cov=finalize_cov(cov, n_assets=len(assets)),
                prior=prior,
                omega=np.zeros((0, 0)),
            )
            return post, assets, w_mkt, delta

        if list(self.views.assets) != assets:
            raise RiskModelError(
                "Views asset universe must match returns.assets in the same order; "
                f"views={list(self.views.assets)} returns={assets}."
            )
        p = self.views.P
        q = self.views.Q
        if self.views.confidences is not None:
            omega = idzorek_omega(cov, p, tau, self.views.confidences)
        else:
            omega = default_omega(cov, p, tau)
        post = posterior(cov, prior, p, q, omega, tau)
        return post, assets, w_mkt, delta

    # -- public API -------------------------------------------------------------

    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        """Return posterior ``E(R)`` as :class:`ExpectedReturns` (implements ``MeanModel``)."""
        post, assets, _w, _d = self._build(returns)
        return to_expected_returns(post.mu, assets)

    def estimate_posterior(self, returns: ReturnMatrix) -> BLPosterior:
        """Return the full :class:`BLPosterior` (mu, covariance, prior, omega)."""
        post, _assets, _w, _d = self._build(returns)
        return post

    def posterior_covariance(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the posterior covariance ``Σ_post`` (PSD-repaired)."""
        post, _assets, _w, _d = self._build(returns)
        return post.cov

    def implied_weights(self, returns: ReturnMatrix) -> dict[str, float]:
        """Posterior implied weights ``w = (δΣ)⁻¹·E(R)``, normalized to sum to 1.

        Solved as a linear system ``(δΣ) w_raw = E(R)`` (no explicit inverse).
        """
        post, assets, _w, delta = self._build(returns)
        cov = post.cov
        w_raw = _solve(delta * cov, post.mu)
        total = w_raw.sum()
        if abs(total) < 1e-12:
            raise RiskModelError("Implied weights sum to ~0; cannot normalize.")
        w = w_raw / total
        return {a: float(x) for a, x in zip(assets, w, strict=True)}


def black_litterman(
    *,
    views: Views | None = None,
    inputs: BLInputs | None = None,
    risk_model: RiskModel | None = None,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> BlackLitterman:
    """Factory for the ``"black_litterman"`` mean model (BUILD_PLAN §5.2)."""
    return BlackLitterman(
        views=views,
        inputs=inputs,
        risk_model=risk_model,
        periods_per_year=periods_per_year,
    )


__all__ = [
    "BLInputs",
    "BLPosterior",
    "BlackLitterman",
    "Views",
    "black_litterman",
    "default_omega",
    "idzorek_omega",
    "implied_prior_returns",
    "market_implied_risk_aversion",
    "posterior",
]
