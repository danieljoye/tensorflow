"""cvxpy log-barrier risk-budget program for non-separable constraints.

CCD (:mod:`riskbudget.optimize.ccd`) handles the *separable* case (long-only +
box bounds + leverage). Genuinely coupling constraints — group/sector caps,
turnover limits, arbitrary linear ``C w ≤ d`` — cannot be expressed as a
per-coordinate update, so we solve the disciplined-convex program directly:

    minimize   ½ wᵀ Σ w − bᵀ log(w)
    subject to  Σ w = 1,  w ≥ 0,  (optional) bounds / group caps / turnover

Because the log barrier already forces ``w > 0`` and the standard form fixes the
budget at ``Σw = 1``, the solution's risk contributions are proportional to ``b``;
the added linear constraints bend it toward feasibility. The result is rescaled to
the requested leverage.

Source: Spinu (2013); pyrb constrained risk budgeting (Richard–Roncalli 2019);
BUILD_PLAN §3.1, §11, §12.1.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import OptimizationError
from riskbudget.optimize.constraints import GroupCapSpec

try:  # cvxpy is a runtime dep, but guard so import errors surface cleanly.
    import cvxpy as cp

    _HAVE_CVXPY = True
except ImportError:  # pragma: no cover - cvxpy is declared in pyproject
    _HAVE_CVXPY = False


def solve_risk_budget_convex(
    cov: np.ndarray,
    budget: np.ndarray,
    *,
    leverage: float = 1.0,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    group_caps: list[GroupCapSpec] | None = None,
    prev_weights: np.ndarray | None = None,
    max_turnover: float | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve the constrained log-barrier risk-budget program with cvxpy.

    Parameters
    ----------
    cov, budget, leverage:
        As in :func:`riskbudget.optimize.ccd.solve_risk_budget_ccd`.
    lower, upper:
        Optional per-asset bounds in normalized weight space.
    group_caps:
        Resolved :class:`~riskbudget.optimize.constraints.GroupCapSpec` list:
        ``Σ_{i in members} wᵢ ≤ cap`` (long-only, so ``wᵢ ≥ 0``).
    prev_weights, max_turnover:
        If both supplied, add ``Σ |wᵢ − prevᵢ| ≤ max_turnover`` (one-period
        turnover cap) in normalized weight space.

    Raises
    ------
    OptimizationError
        If cvxpy is unavailable, the problem is infeasible/unbounded, or the
        solver does not return an optimal point.
    """
    if not _HAVE_CVXPY:  # pragma: no cover
        raise OptimizationError("cvxpy is required for constrained risk budgeting but is missing.")

    sigma = np.asarray(cov, dtype=float)
    b = np.asarray(budget, dtype=float).ravel()
    n = b.size
    if sigma.shape != (n, n):
        raise OptimizationError("Covariance shape does not match the budget length.")
    if np.any(b <= 0):
        raise OptimizationError("Risk budgets must be strictly positive (log-barrier).")
    b = b / float(b.sum())

    # Solve the *unnormalized* Spinu program over x > 0 (no Σx constraint): its
    # stationarity gives xᵢ(Σx)ᵢ ∝ bᵢ — equal/target risk contributions — and the
    # final weights are w = x / Σx. Fixing Σx = 1 would distort the risk
    # contributions, so we instead express every normalized-weight constraint
    # homogeneously in x: a bound ``wᵢ ≤ uᵢ`` becomes ``xᵢ ≤ uᵢ · (1ᵀx)``, etc.
    x = cp.Variable(n, pos=True)
    objective = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(sigma)) - b @ cp.log(x))
    sx = cp.sum(x)
    constraints: list = []

    if lower is not None:
        lo = np.asarray(lower, dtype=float).ravel()
        finite = np.isfinite(lo) & (lo > 0.0)
        if finite.any():
            constraints.append(x[finite] >= lo[finite] * sx)
    if upper is not None:
        hi = np.asarray(upper, dtype=float).ravel()
        finite = np.isfinite(hi)
        if finite.any():
            constraints.append(x[finite] <= hi[finite] * sx)

    for spec in group_caps or []:
        if spec.members:
            idx = list(spec.members)
            constraints.append(cp.sum(x[idx]) <= (spec.cap / float(leverage)) * sx)

    if prev_weights is not None and max_turnover is not None:
        prev = np.asarray(prev_weights, dtype=float).ravel()
        if prev.size != n:
            raise OptimizationError("prev_weights length does not match the universe.")
        prev_norm = prev / float(prev.sum()) if prev.sum() != 0 else prev
        # ‖w − prev‖₁ ≤ τ in normalized space ⇔ ‖x − prev·(1ᵀx)‖₁ ≤ τ·(1ᵀx).
        constraints.append(cp.norm1(x - prev_norm * sx) <= float(max_turnover) * sx)

    # No Σx constraint: the Spinu objective ½xᵀΣx − bᵀlog(x) has a unique
    # unconstrained minimizer over x > 0, and a Σx=1 constraint would distort the
    # risk contributions (it adds a Lagrange term to the stationarity condition).
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    except (cp.error.SolverError, cp.error.DCPError) as exc:
        raise OptimizationError(f"cvxpy risk-budget solver failed: {exc}") from exc

    if problem.status not in ("optimal", "optimal_inaccurate") or x.value is None:
        raise OptimizationError(
            f"cvxpy risk-budget problem is {problem.status} (likely infeasible constraints)."
        )

    sol = np.asarray(x.value, dtype=float)
    if not np.all(sol > -1e-9) or not np.isfinite(sol).all():
        raise OptimizationError("cvxpy solver produced invalid weights.")
    sol = np.clip(sol, 0.0, None)
    out: np.ndarray = sol / float(sol.sum()) * float(leverage)
    return out


__all__ = ["solve_risk_budget_convex"]
