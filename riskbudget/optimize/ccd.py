"""Spinu log-barrier cyclical coordinate descent — the DEFAULT risk-budget solver.

Solves the convex risk-budgeting program (Maillard–Roncalli–Teïletche / Spinu):

    minimize   f(x) = ½ xᵀ Σ x − Σᵢ bᵢ ln xᵢ      over   x > 0

whose stationarity condition ``Σx = b / x`` is exactly ``xᵢ (Σx)ᵢ = bᵢ`` — i.e. the
risk contributions of the (unnormalized) ``x`` are proportional to the budget ``b``.
The final portfolio weights are ``w = x / Σx`` (then rescaled to the target leverage).

Per-coordinate update (BUILD_PLAN §12.1). Holding the other coordinates fixed, the
optimum in ``xᵢ`` is the positive root of the scalar quadratic
``Σᵢᵢ xᵢ² + (Σx)₋ᵢ xᵢ − bᵢ = 0`` where ``(Σx)₋ᵢ = (Σx)ᵢ − Σᵢᵢ xᵢ``:

    xᵢ ← (aux + √(aux² + 4 Σᵢᵢ bᵢ)) / (2 Σᵢᵢ),   aux = xᵢ Σᵢᵢ − (Σx)ᵢ

After each coordinate moves we maintain the running product ``Σx`` with a rank-1
update (``Σx += Σ[:, i] · Δxᵢ``) so a full sweep is O(N²). We initialise
``x = √(1/Σ.sum()) · 1`` and stop when ``maxᵢ |RCᵢ/ΣRC − bᵢ| < 1e-8`` (with
``RCᵢ = xᵢ (Σx)ᵢ``), ``maxiter ≈ 200``.

An optional expected-return tilt (Roncalli) uses the std-dev-measure variant
``f = c·σ(x) − πᵀx − Σ bᵢ ln xᵢ``; its per-coordinate root adds the linear ``π``
term and divides the quadratic by ``c/σ`` — implemented via the same positive-root
formula on the rescaled coefficients.

Source: Spinu (2013) SSRN 2297383; Griveau-Billion–Richard–Roncalli (2013)
arXiv:1311.4057; `convexfi/riskparity.py`. BUILD_PLAN §2, §11, §12.1.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import OptimizationError

# Convergence on max abs risk-contribution error (BUILD_PLAN §3.1).
_DEFAULT_TOL = 1e-8
_DEFAULT_MAXITER = 200
# Treat |x| < this as zero (BUILD_PLAN §3.1).
_ZERO_TOL = 1e-12


def _validate_inputs(cov: np.ndarray, budget: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(cov, dtype=float)
    b = np.asarray(budget, dtype=float).ravel()
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise OptimizationError(f"Covariance must be square, got shape {sigma.shape}.")
    if sigma.shape[0] != b.size:
        raise OptimizationError(f"Covariance shape {sigma.shape} does not match {b.size} budgets.")
    if not np.isfinite(sigma).all():
        raise OptimizationError("Covariance contains NaN or infinite values.")
    if not np.isfinite(b).all():
        raise OptimizationError("Budget contains NaN or infinite values.")
    if np.any(b <= 0):
        raise OptimizationError("Risk budgets must be strictly positive (log-barrier).")
    diag = np.diag(sigma)
    if np.any(diag <= _ZERO_TOL):
        raise OptimizationError(
            "Covariance has a non-positive diagonal; CCD requires positive variances."
        )
    return sigma, b


def solve_risk_budget_ccd(
    cov: np.ndarray,
    budget: np.ndarray,
    *,
    leverage: float = 1.0,
    pi: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    tol: float = _DEFAULT_TOL,
    maxiter: int = _DEFAULT_MAXITER,
) -> np.ndarray:
    """Solve the long-only risk-budget program by Spinu CCD; return weights ``w``.

    Parameters
    ----------
    cov:
        Covariance matrix ``Σ`` of shape ``(N, N)``, aligned to ``budget``.
    budget:
        Target risk-contribution fractions ``bᵢ`` (strictly positive; need not be
        pre-normalized — only ratios matter to the solution direction).
    leverage:
        Target gross exposure to rescale the final weights to (default ``1.0``).
    pi:
        Optional expected-return tilt ``π`` (Roncalli std-dev-measure variant). When
        given, the objective becomes ``c·σ(x) − πᵀx − Σ bᵢ ln xᵢ``.
    risk_aversion:
        The ``c`` coefficient on volatility in the tilt variant (ignored when
        ``pi`` is ``None``). Larger ``c`` → more risk-averse (closer to plain ERC).
    tol, maxiter:
        Convergence tolerance on the max risk-contribution error and iteration cap.

    Returns
    -------
    np.ndarray
        Long-only weights summing (in absolute value) to ``leverage``.

    Raises
    ------
    OptimizationError
        On invalid inputs or non-convergence within ``maxiter`` sweeps.
    """
    sigma, b = _validate_inputs(cov, budget)
    n = b.size
    diag = np.diag(sigma).copy()

    use_tilt = pi is not None
    if use_tilt:
        pi_vec = np.asarray(pi, dtype=float).ravel()
        if pi_vec.size != n:
            raise OptimizationError("Expected-return tilt pi must match the number of assets.")
        if not np.isfinite(pi_vec).all():
            raise OptimizationError("Expected-return tilt pi contains non-finite values.")
        if not np.isfinite(risk_aversion) or risk_aversion <= 0:
            raise OptimizationError("risk_aversion must be finite and positive.")
    else:
        pi_vec = np.zeros(n)

    # Initialisation (BUILD_PLAN §12.1).
    total = float(sigma.sum())
    x0 = np.sqrt(1.0 / total) if total > _ZERO_TOL else 1.0 / np.sqrt(n)
    x = np.full(n, x0, dtype=float)
    sigma_x = sigma @ x  # running Σx, maintained by rank-1 updates

    converged = False
    err = np.inf
    b_norm = b / b.sum()
    for _ in range(maxiter):
        max_step = 0.0
        for i in range(n):
            # (Σx)₋ᵢ = (Σx)ᵢ − Σᵢᵢ xᵢ ; aux = xᵢ Σᵢᵢ − (Σx)ᵢ = −(Σx)₋ᵢ
            if use_tilt:
                # Std-dev measure: gradient_i = c (Σx)ᵢ/σ − πᵢ − bᵢ/xᵢ = 0.
                vol = float(np.sqrt(max(x @ sigma_x, _ZERO_TOL)))
                scale = risk_aversion / vol
                a_coef = scale * diag[i]
                off = scale * (sigma_x[i] - diag[i] * x[i]) - pi_vec[i]
                aux = -off
                new_xi = (aux + np.sqrt(aux * aux + 4.0 * a_coef * b[i])) / (2.0 * a_coef)
            else:
                aux = x[i] * diag[i] - sigma_x[i]
                new_xi = (aux + np.sqrt(aux * aux + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
            delta = new_xi - x[i]
            if delta != 0.0:
                max_step = max(max_step, abs(delta) / max(abs(new_xi), _ZERO_TOL))
                x[i] = new_xi
                sigma_x += sigma[:, i] * delta  # rank-1 maintenance of Σx

        if use_tilt:
            # The tilt breaks the equal-RC stationarity; stop on the relative
            # coordinate step instead (fixed-point convergence).
            if max_step < tol:
                converged = True
                break
            continue

        rc = x * sigma_x  # RCᵢ = xᵢ (Σx)ᵢ
        rc_sum = float(rc.sum())
        if rc_sum <= _ZERO_TOL:
            raise OptimizationError("CCD produced a degenerate (zero-risk) iterate.")
        err = float(np.max(np.abs(rc / rc_sum - b_norm)))
        if err < tol:
            converged = True
            break

    if not converged:
        raise OptimizationError(
            f"Spinu CCD did not converge within {maxiter} sweeps (last error {err:.2e})."
        )
    if not np.all(x > 0) or not np.isfinite(x).all():
        raise OptimizationError("CCD produced non-positive or non-finite weights.")

    w = x / float(x.sum())
    return w * float(leverage)


__all__ = ["solve_risk_budget_ccd"]
