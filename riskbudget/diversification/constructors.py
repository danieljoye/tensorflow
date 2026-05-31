"""Diversification-driven portfolio constructors (BUILD_PLAN §5.1, §12.4–§12.5).

Three :class:`~riskbudget.core.interfaces.PortfolioConstructor` implementations
sitting on the diversification metrics and the minimum-torsion factor basis:

- :class:`MostDiversifiedConstructor` — the **Most Diversified Portfolio** (MDP),
  the long-only book that maximises the Choueifaty–Coignard Diversification Ratio
  ``DR(w) = (wᵀσ) / √(wᵀΣw)``. Choueifaty & Coignard, "Toward Maximum
  Diversification" (*Journal of Portfolio Management*, 2008).
- :class:`MaxENBConstructor` — the **max-ENB portfolio**, maximising Meucci's
  Effective Number of Bets on an uncorrelated (min-torsion by default) factor
  basis by minimising ``−ENB`` with SLSQP under ``Σw = 1, 0 ≤ w ≤ 1``. An
  optional ``enb_floor`` adds an ``ENB(w) ≥ k`` inequality constraint. Meucci,
  "Managing Diversification" (*Risk*, 2009).
- :class:`FactorRiskBudgetConstructor` — the **factor-risk-budget portfolio**:
  weights whose *factor* risk contributions (on the min-torsion basis) match a
  target budget — the factor analogue of asset-level ERC (Agent 4). Meucci,
  Santangelo & Deguest (min-torsion, SSRN 2276632); Roncalli, *Introduction to
  Risk Parity and Budgeting* (2013) for the budgeting formulation.

All three are long-only (``constraints.long_only`` defaults to ``True``) and fully
invested; weights sum to one. Optimisation uses
:func:`scipy.optimize.minimize(method="SLSQP")`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget
from riskbudget.diversification.metrics import (
    diversification_ratio,
    effective_number_of_bets,
)
from riskbudget.diversification.torsion import TorsionMethod, torsion

# Numerical-zero / convergence tolerances (BUILD_PLAN §3.1).
_ZERO_TOL = 1e-12
_SLSQP_TOL = 1e-10
_SLSQP_MAXITER = 500


def _validate_cov(cov: np.ndarray) -> np.ndarray:
    """Coerce and check a covariance matrix; raise OptimizationError on bad input."""
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise OptimizationError(f"Covariance must be square, got shape {sigma.shape}.")
    if sigma.shape[0] == 0:
        raise OptimizationError("Covariance must have at least one asset.")
    if not np.isfinite(sigma).all():
        raise OptimizationError("Covariance contains NaN or infinite values.")
    if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-10):
        raise OptimizationError("Covariance matrix is not symmetric.")
    return np.asarray(0.5 * (sigma + sigma.T), dtype=float)


def _resolve_assets(
    n: int,
    *,
    budget: RiskBudget | None,
    mu: ExpectedReturns | None,
    assets: Sequence[str] | None,
) -> list[str]:
    """Pick a canonical asset ordering of length ``n`` from the available inputs."""
    if assets is not None:
        order = [str(a) for a in assets]
    elif budget is not None:
        order = budget.assets
    elif mu is not None:
        order = mu.assets
    else:
        order = [str(i) for i in range(n)]
    if len(order) != n:
        raise OptimizationError(
            f"Asset labels ({len(order)}) do not match covariance dimension ({n})."
        )
    return order


def _bounds(n: int, constraints: Constraints) -> list[tuple[float, float]]:
    """Per-asset bounds honouring long-only and optional min/max weight caps."""
    if not constraints.long_only:
        # Risk budgeting / diversification constructors are inherently long-only
        # (BUILD_PLAN §3.1): a maximally diversified or factor-budgeted book is
        # defined over w >= 0. Surface the unsupported request explicitly.
        raise OptimizationError(
            "Diversification constructors are long-only; "
            "set constraints.long_only=True (shorting is not supported)."
        )
    lo = 0.0 if constraints.min_weight is None else max(0.0, float(constraints.min_weight))
    hi = 1.0 if constraints.max_weight is None else float(constraints.max_weight)
    if hi < lo:
        raise OptimizationError("Constraints.max_weight cannot be below the long-only floor.")
    return [(lo, hi)] * n


def _sum_to_one_constraint() -> dict[str, object]:
    """SLSQP equality constraint ``Σw = 1``."""
    return {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}


def _clean_weights(w: np.ndarray) -> np.ndarray:
    """Zero out sub-tolerance dust and renormalise to sum exactly to one."""
    w = np.where(np.abs(w) < _ZERO_TOL, 0.0, w)
    total = float(np.sum(w))
    if abs(total) < _ZERO_TOL:
        raise OptimizationError("Solver produced an all-zero weight vector.")
    return w / total


def _to_portfolio(w: np.ndarray, order: Sequence[str]) -> Portfolio:
    """Assemble a :class:`Portfolio` from a weight vector and an asset ordering."""
    return Portfolio({a: float(wi) for a, wi in zip(order, w, strict=True)})


def _solve_slsqp(
    objective: Callable[[np.ndarray], float],
    n: int,
    constraints: Constraints,
    *,
    extra_constraints: list[dict[str, object]] | None = None,
    label: str,
) -> np.ndarray:
    """Run SLSQP from an equal-weight start under ``Σw=1`` + bounds (+ extras)."""
    bounds = _bounds(n, constraints)
    cons: list[dict[str, object]] = [_sum_to_one_constraint()]
    if extra_constraints:
        cons.extend(extra_constraints)
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": _SLSQP_MAXITER, "ftol": _SLSQP_TOL},
    )
    if not result.success:
        raise OptimizationError(f"{label} optimisation did not converge: {result.message}")
    return _clean_weights(np.asarray(result.x, dtype=float))


# ---------------------------------------------------------------------------
# Most Diversified Portfolio (maximise the Diversification Ratio)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MostDiversifiedConstructor:
    """Most Diversified Portfolio: long-only weights maximising ``DR(w)``.

    Maximises the Choueifaty–Coignard Diversification Ratio
    ``DR(w) = (wᵀσ) / √(wᵀΣw)`` subject to ``Σw = 1`` and ``0 ≤ w``. Equivalent
    to the minimum-variance portfolio on the *correlation* matrix, rescaled by
    volatilities; here it is solved directly via SLSQP on ``−DR``.
    """

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
        assets: Sequence[str] | None = None,
    ) -> Portfolio:
        """Return the Most Diversified Portfolio for ``cov`` under ``constraints``.

        ``mu`` and ``budget`` are accepted for interface uniformity but unused.
        ``cov`` must be ordered to match ``assets`` (or ``budget``/``mu`` when
        those carry the ordering).
        """
        sigma = _validate_cov(cov)
        n = sigma.shape[0]
        order = _resolve_assets(n, budget=budget, mu=mu, assets=assets)

        def neg_dr(w: np.ndarray) -> float:
            # Guard against the degenerate all-zero interior point.
            if float(w @ sigma @ w) <= _ZERO_TOL:
                return 0.0
            return -diversification_ratio(w, sigma)

        w = _solve_slsqp(neg_dr, n, constraints, label="Most Diversified Portfolio")
        return _to_portfolio(w, order)


# ---------------------------------------------------------------------------
# Max-ENB (maximise the Effective Number of Bets)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxENBConstructor:
    """Max-ENB portfolio: maximise Meucci's Effective Number of Bets.

    Minimises ``−ENB(w)`` via SLSQP under ``Σw = 1, 0 ≤ w ≤ 1`` on an
    uncorrelated factor basis (``method``, default ``"minimum-torsion"``). The
    torsion matrix is computed once from ``cov`` and reused across the objective
    evaluations.

    Parameters
    ----------
    method:
        Factor basis for the ENB distribution: ``"minimum-torsion"`` (default),
        ``"pca"``, or ``"approximate"``.
    enb_floor:
        Optional ``ENB(w) ≥ k`` inequality constraint (the ``ENB(w) ≥ k``
        wrapper). When set, the construction still maximises ENB but additionally
        requires the floor to be feasible (useful when wrapping another
        objective).
    """

    method: TorsionMethod = "minimum-torsion"
    enb_floor: float | None = None

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
        assets: Sequence[str] | None = None,
    ) -> Portfolio:
        """Return the max-ENB portfolio for ``cov`` under ``constraints``."""
        sigma = _validate_cov(cov)
        n = sigma.shape[0]
        order = _resolve_assets(n, budget=budget, mu=mu, assets=assets)

        # Precompute the torsion basis once; reuse it for every ENB evaluation.
        t = torsion(sigma, self.method)

        def neg_enb(w: np.ndarray) -> float:
            if float(w @ sigma @ w) <= _ZERO_TOL:
                return 0.0
            try:
                return -effective_number_of_bets(w, sigma, t=t)
            except Exception:
                return 0.0

        extra: list[dict[str, object]] | None = None
        if self.enb_floor is not None:
            floor = float(self.enb_floor)

            def enb_margin(w: np.ndarray, _floor: float = floor) -> float:
                if float(w @ sigma @ w) <= _ZERO_TOL:
                    return -_floor
                try:
                    return effective_number_of_bets(w, sigma, t=t) - _floor
                except Exception:
                    return -_floor

            extra = [{"type": "ineq", "fun": enb_margin}]

        w = _solve_slsqp(neg_enb, n, constraints, extra_constraints=extra, label="Max-ENB")

        if self.enb_floor is not None:
            achieved = effective_number_of_bets(w, sigma, t=t)
            if achieved < float(self.enb_floor) - 1e-6:
                raise OptimizationError(
                    f"ENB floor {self.enb_floor!r} is infeasible (achieved {achieved:.4f})."
                )
        return _to_portfolio(w, order)


# ---------------------------------------------------------------------------
# Factor risk budget (match factor risk contributions to a target budget)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorRiskBudgetConstructor:
    """Factor-risk-budget portfolio on the min-torsion (uncorrelated) basis.

    The factor analogue of asset-level ERC: find long-only weights whose *factor*
    risk contributions match a target budget ``b`` over the uncorrelated factors
    ``f = t · r``. The factors are uncorrelated, so the factor covariance
    ``Σ_f = t·Σ·tᵀ`` is diagonal with variances ``v``; the factor exposures of a
    portfolio ``w`` are ``g = (tᵀ)⁻¹ w`` (so ``wᵀr = gᵀf``), giving factor risk
    contributions ``FRCₖ = gₖ² vₖ`` and total factor variance ``Σₖ gₖ² vₖ``. We
    minimise the squared deviation of the normalised FRC vector from the target
    budget under ``Σw = 1, 0 ≤ w``. When no budget is supplied the target is the
    equal (ERC-over-factors) budget ``1/N``.

    Parameters
    ----------
    method:
        Factor basis (default ``"minimum-torsion"``).
    """

    method: TorsionMethod = "minimum-torsion"

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
        assets: Sequence[str] | None = None,
    ) -> Portfolio:
        """Return the factor-risk-budget portfolio for ``cov`` under ``constraints``."""
        sigma = _validate_cov(cov)
        n = sigma.shape[0]
        order = _resolve_assets(n, budget=budget, mu=mu, assets=assets)

        target = np.full(n, 1.0 / n) if budget is None else budget.as_array(order)
        target = target / float(np.sum(target))

        t = torsion(sigma, self.method)
        # Factor variances v_k = diag(t Σ tᵀ); exposures g = (tᵀ)⁻¹ w solved per call.
        factor_cov = t @ sigma @ t.T
        v = np.clip(np.diag(factor_cov), 0.0, None)
        t_t = t.T

        def objective(w: np.ndarray) -> float:
            try:
                g = np.linalg.solve(t_t, w)  # (tᵀ)⁻¹ w
            except np.linalg.LinAlgError:
                return 1.0
            frc = (g**2) * v  # factor risk contributions (variance units)
            total = float(np.sum(frc))
            if total <= _ZERO_TOL:
                return 1.0
            p = frc / total
            diff = p - target
            return float(diff @ diff)

        w = _solve_slsqp(objective, n, constraints, label="Factor risk budget")
        return _to_portfolio(w, order)


# ---------------------------------------------------------------------------
# §5.2 factory callables
# ---------------------------------------------------------------------------


def mdp() -> MostDiversifiedConstructor:
    """Factory for the ``"mdp"`` method: the Most Diversified Portfolio."""
    return MostDiversifiedConstructor()


def max_enb(
    method: TorsionMethod = "minimum-torsion", enb_floor: float | None = None
) -> MaxENBConstructor:
    """Factory for the ``"max_enb"`` method: the max-ENB portfolio."""
    return MaxENBConstructor(method=method, enb_floor=enb_floor)


def factor_risk_budget(method: TorsionMethod = "minimum-torsion") -> FactorRiskBudgetConstructor:
    """Factory for the ``"factor_risk_budget"`` method: factor-risk-budget portfolio."""
    return FactorRiskBudgetConstructor(method=method)


__all__ = [
    "FactorRiskBudgetConstructor",
    "MaxENBConstructor",
    "MostDiversifiedConstructor",
    "factor_risk_budget",
    "max_enb",
    "mdp",
]
