"""Log-barrier risk-budget solver via ``scipy.optimize`` — the cross-check oracle.

Minimises the same Spinu objective as :mod:`riskbudget.optimize.ccd`

    f(x) = ½ xᵀ Σ x − Σᵢ bᵢ ln xᵢ      over   x > 0

but with a general Newton-type optimiser (``scipy.optimize.minimize`` /
``trust-constr``) fed the **analytic gradient** ``∇f = Σx − b/x`` and **Hessian**
``H = Σ + diag(b/x²)``. This is the independent oracle the CCD path is validated
against, and it also serves as the *bounded separable* path: optional per-asset
box bounds ``lo ≤ x ≤ hi`` are passed straight to ``trust-constr`` (CCD cannot
honour upper bounds, only the implicit positivity).

Final weights are ``w = x / Σx`` rescaled to the target leverage, exactly as in
the CCD path.

Source: Spinu (2013); BUILD_PLAN §3.1, §11, §12.1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, minimize

from riskbudget.core.errors import OptimizationError

_DEFAULT_TOL = 1e-10
_ZERO_TOL = 1e-12


def _validate(cov: np.ndarray, budget: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(cov, dtype=float)
    b = np.asarray(budget, dtype=float).ravel()
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise OptimizationError(f"Covariance must be square, got shape {sigma.shape}.")
    if sigma.shape[0] != b.size:
        raise OptimizationError(f"Covariance shape {sigma.shape} does not match {b.size} budgets.")
    if not np.isfinite(sigma).all() or not np.isfinite(b).all():
        raise OptimizationError("Covariance/budget contains non-finite values.")
    if np.any(b <= 0):
        raise OptimizationError("Risk budgets must be strictly positive (log-barrier).")
    return sigma, b


def solve_risk_budget_scipy(
    cov: np.ndarray,
    budget: np.ndarray,
    *,
    leverage: float = 1.0,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    tol: float = _DEFAULT_TOL,
    maxiter: int = 500,
) -> np.ndarray:
    """Solve the long-only risk-budget program via scipy; return weights ``w``.

    Parameters
    ----------
    cov, budget, leverage:
        As in :func:`riskbudget.optimize.ccd.solve_risk_budget_ccd`.
    lower, upper:
        Optional per-asset bounds **in normalized weight space** (fractions of the
        leverage). The objective is solved in ``x``-space and the bounds are mapped
        to ``x`` by the leverage scale; ``lower`` defaults to a tiny positive floor
        (positivity for the log barrier), ``upper`` to ``+inf``.

    Raises
    ------
    OptimizationError
        On invalid inputs or if the optimiser fails to converge.
    """
    sigma, b = _validate(cov, budget)
    n = b.size
    b = b / float(b.sum())

    def objective(x: np.ndarray) -> float:
        return float(0.5 * x @ sigma @ x - b @ np.log(x))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad: np.ndarray = sigma @ x - b / x
        return grad

    def hessian(x: np.ndarray) -> np.ndarray:
        return sigma + np.diag(b / (x * x))

    total = float(sigma.sum())
    x0_scalar = np.sqrt(1.0 / total) if total > _ZERO_TOL else 1.0 / np.sqrt(n)
    x0 = np.full(n, x0_scalar, dtype=float)

    # Bounds: positivity floor by default; honor caller's box (mapped to x-space).
    lo = np.full(n, 1e-12)
    hi = np.full(n, np.inf)
    if lower is not None:
        lo = np.maximum(lo, np.asarray(lower, dtype=float).ravel())
    if upper is not None:
        hi = np.minimum(hi, np.asarray(upper, dtype=float).ravel())
    if np.any(lo > hi):
        raise OptimizationError("scipy solver bounds are infeasible (lower > upper).")
    x0 = np.clip(x0, lo + 1e-12, np.where(np.isfinite(hi), hi, x0_scalar) + 0.0)
    x0 = np.maximum(x0, lo)

    try:
        result = minimize(
            objective,
            x0,
            method="trust-constr",
            jac=gradient,
            hess=hessian,
            bounds=Bounds(lo, hi),
            options={"gtol": tol, "xtol": tol, "maxiter": maxiter, "verbose": 0},
        )
    except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - defensive
        raise OptimizationError(f"scipy risk-budget solver failed: {exc}") from exc

    x = np.asarray(result.x, dtype=float)
    if not result.success and result.status not in (1, 2):
        raise OptimizationError(f"scipy risk-budget solver did not converge: {result.message}")
    if not np.all(x > 0) or not np.isfinite(x).all():
        raise OptimizationError("scipy solver produced non-positive or non-finite weights.")

    w = x / float(x.sum())
    return w * float(leverage)


__all__ = ["solve_risk_budget_scipy"]
