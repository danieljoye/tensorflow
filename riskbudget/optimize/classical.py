"""Classical benchmark constructors (EDHEC course formulations).

Each implements the :class:`~riskbudget.core.interfaces.PortfolioConstructor`
contract so the backtester can drive ERC and the classical benchmarks through one
interface:

- :class:`EqualWeight` — naive 1/N baseline.
- :class:`GlobalMinimumVariance` — ``min wᵀΣw`` s.t. constraints; cov only. Keeps the
  closed-form ``σ_gmv = √(1/Σ pinv(Σ))`` as an unconstrained sanity check
  (BUILD_PLAN §12.2).
- :class:`MaxSharpe` — tangency portfolio via the **Cornuéjols–Tütüncü** variable
  substitution (BUILD_PLAN §12.2), NOT naive fractional-Sharpe maximisation.
- :class:`EfficientMSR` — MSR with a *risk-based / volatility-proxy* ``μ``
  (Martellini 2008, "Toward the Design of Better Equity Benchmarks").
- :func:`efficient_frontier` — trace min-variance weights across target returns.

Unlike the risk-budget path these support shorting/leverage when the constraints
allow it (BUILD_PLAN §3.1).

Source: PyPortfolioOpt (Cornuéjols–Tütüncü 2006); EDHEC course
``edhec_risk_kit``; Martellini (JPM 2008); BUILD_PLAN §2, §11, §12.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget

try:
    import cvxpy as cp

    _HAVE_CVXPY = True
except ImportError:  # pragma: no cover
    _HAVE_CVXPY = False

_ZERO_TOL = 1e-12


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_order(
    cov: np.ndarray,
    mu: ExpectedReturns | None,
    budget: RiskBudget | None,
) -> list[str]:
    """Determine the asset ordering for a constructor call."""
    n = np.asarray(cov, dtype=float).shape[0]
    if mu is not None:
        return list(mu.assets)
    if budget is not None:
        return list(budget.assets)
    return [str(i) for i in range(n)]


def _validate_cov(cov: np.ndarray, n: int) -> np.ndarray:
    sigma = np.asarray(cov, dtype=float)
    if sigma.shape != (n, n):
        raise OptimizationError(f"Covariance shape {sigma.shape} inconsistent with {n} assets.")
    if not np.isfinite(sigma).all():
        raise OptimizationError("Covariance contains non-finite values.")
    return sigma


def _box_bounds(constraints: Constraints, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset (lower, upper) for the classical QP, honoring long_only + bounds."""
    lo = np.zeros(n) if constraints.long_only else np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if constraints.min_weight is not None:
        lo = np.maximum(lo, float(constraints.min_weight))
    if constraints.max_weight is not None:
        hi = np.minimum(hi, float(constraints.max_weight))
    if np.any(lo > hi):
        raise OptimizationError("Classical optimizer bounds are infeasible.")
    return lo, hi


def _qp_min_variance(
    sigma: np.ndarray,
    constraints: Constraints,
    *,
    target_return: float | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    """Solve ``min wᵀΣw`` s.t. ``Σw = leverage`` (+ bounds, optional target return)."""
    if not _HAVE_CVXPY:  # pragma: no cover
        raise OptimizationError("cvxpy is required for the classical optimizers.")
    n = sigma.shape[0]
    leverage = 1.0 if constraints.leverage is None else float(constraints.leverage)
    lo, hi = _box_bounds(constraints, n)

    w = cp.Variable(n)
    cons = [cp.sum(w) == leverage]
    if constraints.long_only:
        cons.append(w >= 0)
    fin_lo = np.isfinite(lo)
    if fin_lo.any():
        cons.append(w[fin_lo] >= lo[fin_lo])
    fin_hi = np.isfinite(hi)
    if fin_hi.any():
        cons.append(w[fin_hi] <= hi[fin_hi])
    if target_return is not None and mu is not None:
        cons.append(mu @ w == target_return)

    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons)
    try:
        problem.solve(solver=cp.CLARABEL)
    except (cp.error.SolverError, cp.error.DCPError) as exc:
        raise OptimizationError(f"GMV/frontier QP failed: {exc}") from exc
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise OptimizationError(f"GMV/frontier QP is {problem.status} (infeasible constraints).")
    return np.asarray(w.value, dtype=float)


def gmv_sigma_closed_form(cov: np.ndarray) -> float:
    """Unconstrained GMV volatility ``σ_gmv = √(1/(1ᵀ Σ⁺ 1))`` (BUILD_PLAN §12.2)."""
    sigma = np.asarray(cov, dtype=float)
    ones = np.ones(sigma.shape[0])
    denom = float(ones @ np.linalg.pinv(sigma) @ ones)
    if denom <= _ZERO_TOL:
        raise OptimizationError("Degenerate covariance: closed-form σ_gmv undefined.")
    return float(np.sqrt(1.0 / denom))


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EqualWeight:
    """Naive ``1/N`` portfolio, scaled to the target leverage."""

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        order = _resolve_order(cov, mu, budget)
        n = len(order)
        leverage = 1.0 if constraints.leverage is None else float(constraints.leverage)
        w = np.full(n, leverage / n)
        return Portfolio(dict(zip(order, (float(x) for x in w), strict=True)))


@dataclass(frozen=True)
class GlobalMinimumVariance:
    """Global minimum variance: ``min wᵀΣw`` s.t. constraints. Needs only ``Σ``."""

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        order = _resolve_order(cov, mu, budget)
        sigma = _validate_cov(cov, len(order))
        w = _qp_min_variance(sigma, constraints)
        return Portfolio(dict(zip(order, (float(x) for x in w), strict=True)))


@dataclass(frozen=True)
class MaxSharpe:
    """Tangency portfolio via the Cornuéjols–Tütüncü substitution (BUILD_PLAN §12.2).

    Solves the convex QP ``min yᵀΣy`` s.t. ``(μ−r_f)ᵀy = 1``, ``Σy = k``, ``k ≥ 0``
    (all linear constraints scaled by ``k``) and recovers ``w = y/k`` — avoiding the
    non-convex fractional Sharpe. Requires ``max(μ) > r_f``.

    Attributes
    ----------
    risk_free:
        Annualized risk-free rate ``r_f`` (default ``0.0``).
    """

    risk_free: float = 0.0

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        if mu is None:
            raise OptimizationError("MaxSharpe (MSR) requires an expected-returns vector `mu`.")
        order = list(mu.assets)
        sigma = _validate_cov(cov, len(order))
        mu_vec = mu.as_array(order)
        return _solve_msr(sigma, mu_vec, order, constraints, self.risk_free)


@dataclass(frozen=True)
class EfficientMSR:
    """Efficient-MSR: MSR using a risk-based / volatility-proxy ``μ`` (Martellini 2008).

    Identical machinery to :class:`MaxSharpe`, but intended to consume Agent 3's
    risk-based proxy expected returns (total volatility / semi-deviation) rather
    than unreliable sample means — the "Efficient Indexation" benchmark.

    Source: Martellini, "Toward the Design of Better Equity Benchmarks" (JPM 2008);
    Amenc–Goltz–Martellini–Retkowsky (JOIM 2011). BUILD_PLAN §2, §11.
    """

    risk_free: float = 0.0

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        if mu is None:
            raise OptimizationError(
                "EfficientMSR requires a risk-based expected-returns proxy `mu` (Agent 3)."
            )
        order = list(mu.assets)
        sigma = _validate_cov(cov, len(order))
        mu_vec = mu.as_array(order)
        return _solve_msr(sigma, mu_vec, order, constraints, self.risk_free)


def _solve_msr(
    sigma: np.ndarray,
    mu_vec: np.ndarray,
    order: list[str],
    constraints: Constraints,
    risk_free: float,
) -> Portfolio:
    if not _HAVE_CVXPY:  # pragma: no cover
        raise OptimizationError("cvxpy is required for the MSR optimizer.")
    excess = mu_vec - float(risk_free)
    if np.max(excess) <= 0:
        raise OptimizationError(
            "MSR is undefined: no asset has expected return above the risk-free rate."
        )
    n = sigma.shape[0]
    leverage = 1.0 if constraints.leverage is None else float(constraints.leverage)
    lo, hi = _box_bounds(constraints, n)

    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    cons = [excess @ y == 1.0, cp.sum(y) == leverage * k]
    if constraints.long_only:
        cons.append(y >= 0)
    fin_lo = np.isfinite(lo)
    if fin_lo.any():
        cons.append(y[fin_lo] >= lo[fin_lo] * k)
    fin_hi = np.isfinite(hi)
    if fin_hi.any():
        cons.append(y[fin_hi] <= hi[fin_hi] * k)

    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), cons)
    try:
        problem.solve(solver=cp.CLARABEL)
    except (cp.error.SolverError, cp.error.DCPError) as exc:
        raise OptimizationError(f"MSR QP failed: {exc}") from exc
    bad = problem.status not in ("optimal", "optimal_inaccurate")
    if bad or y.value is None or k.value is None:
        raise OptimizationError(f"MSR QP is {problem.status} (infeasible constraints).")
    if float(k.value) <= _ZERO_TOL:
        raise OptimizationError("MSR substitution degenerate (k ≈ 0); check μ and r_f.")

    w = np.asarray(y.value, dtype=float) / float(k.value)
    return Portfolio(dict(zip(order, (float(x) for x in w), strict=True)))


def efficient_frontier(
    cov: np.ndarray,
    mu: ExpectedReturns,
    constraints: Constraints,
    *,
    n_points: int = 20,
) -> list[Portfolio]:
    """Trace min-variance portfolios across a grid of target returns.

    Returns ``n_points`` portfolios spanning ``[min(μ), max(μ)]`` (the
    ``minimize_vol`` / ``optimal_weights`` pattern of the EDHEC course). The
    sequence is monotone in target return and convex in (vol, return) space.

    Raises
    ------
    OptimizationError
        If ``n_points < 2`` or a target return is infeasible under ``constraints``.
    """
    if n_points < 2:
        raise OptimizationError("efficient_frontier needs at least 2 points.")
    order = list(mu.assets)
    sigma = _validate_cov(cov, len(order))
    mu_vec = mu.as_array(order)
    targets = np.linspace(float(mu_vec.min()), float(mu_vec.max()), n_points)
    portfolios: list[Portfolio] = []
    for t in targets:
        w = _qp_min_variance(sigma, constraints, target_return=float(t), mu=mu_vec)
        portfolios.append(Portfolio(dict(zip(order, (float(x) for x in w), strict=True))))
    return portfolios


# ---------------------------------------------------------------------------
# §5.2 factory callables (registry-facing)
# ---------------------------------------------------------------------------


def equal_weight() -> EqualWeight:
    """Factory for the equal-weight constructor (§5.2 name ``equal_weight``)."""
    return EqualWeight()


def gmv() -> GlobalMinimumVariance:
    """Factory for the global-minimum-variance constructor (§5.2 name ``gmv``)."""
    return GlobalMinimumVariance()


def msr(risk_free: float = 0.0) -> MaxSharpe:
    """Factory for the max-Sharpe / tangency constructor (§5.2 name ``msr``)."""
    return MaxSharpe(risk_free=risk_free)


def efficient_msr(risk_free: float = 0.0) -> EfficientMSR:
    """Factory for the Efficient-MSR constructor (§5.2 name ``efficient_msr``)."""
    return EfficientMSR(risk_free=risk_free)


__all__ = [
    "EfficientMSR",
    "EqualWeight",
    "GlobalMinimumVariance",
    "MaxSharpe",
    "efficient_frontier",
    "efficient_msr",
    "equal_weight",
    "gmv",
    "gmv_sigma_closed_form",
    "msr",
]
