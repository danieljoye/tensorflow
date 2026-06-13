"""Shrinkage covariance estimators (BUILD_PLAN §4, Agent 3).

Sample covariance is noisy and ill-conditioned when the number of assets ``N`` is
close to (or exceeds) the number of periods ``T``. Shrinkage pulls the sample
estimate toward a structured, low-variance *target*, trading a little bias for a
large reduction in estimation error (and condition number). Three estimators:

- :class:`LedoitWolfConstantVariance` — Ledoit–Wolf (2004) toward the
  scaled-identity target ``μ·I`` (``μ`` = mean variance). Thin wrapper over
  scikit-learn's analytic ``LedoitWolf``.
- :class:`LedoitWolfConstantCorrelation` — Ledoit–Wolf (2003) toward the
  *constant-correlation* target (common pairwise correlation, individual
  variances). The target the equity literature recommends; a self-contained
  NumPy port of the analytic shrinkage-intensity formula.
- :class:`OAS` — Chen et al. (2010) Oracle-Approximating Shrinkage toward
  ``μ·I``, a refinement of LW with lower MSE under Gaussianity. Wraps
  scikit-learn's ``oas``.

References: BUILD_PLAN §4, §11 (PyPortfolioOpt / sklearn). Ledoit & Wolf,
"Honey, I Shrunk the Sample Covariance Matrix" (2004) and "Improved Estimation of
the Covariance Matrix of Stock Returns…" (2003); Chen, Wiesel, Eldar & Hero,
"Shrinkage Algorithms for MMSE Covariance Estimation" (2010). Every output is
routed through the §12.3 PSD fix.
"""

from __future__ import annotations

import numpy as np

from riskbudget.core.errors import RiskModelError
from riskbudget.core.types import ReturnMatrix
from riskbudget.riskmodel._common import (
    DEFAULT_PERIODS_PER_YEAR,
    finalize_cov,
    returns_array,
    validate_periods_per_year,
)


def _clip_shrinkage(delta: float) -> float:
    """Clamp a shrinkage intensity into ``[0, 1]`` (washes out tiny float drift)."""
    if not np.isfinite(delta):
        return 1.0
    return float(min(1.0, max(0.0, delta)))


class LedoitWolfConstantVariance:
    """Ledoit–Wolf shrinkage toward the scaled-identity target (implements ``RiskModel``).

    The shrinkage target is ``μ·I`` with ``μ`` the average sample variance — i.e.
    every asset gets the mean variance and zero covariance. Uses scikit-learn's
    analytic :class:`sklearn.covariance.LedoitWolf` for the (unannualized)
    estimate, then annualizes by ``periods_per_year``.

    Parameters
    ----------
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(self, *, periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR) -> None:
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        self.shrinkage_: float | None = None

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized constant-variance LW covariance."""
        from sklearn.covariance import LedoitWolf

        values = returns_array(returns, min_periods=2)
        n_assets = values.shape[1]
        try:
            lw = LedoitWolf().fit(values)
        except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
            raise RiskModelError(f"Ledoit-Wolf shrinkage failed: {exc}") from exc
        self.shrinkage_ = float(lw.shrinkage_)
        cov = np.asarray(lw.covariance_, dtype=float) * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


class LedoitWolfConstantCorrelation:
    """Ledoit–Wolf shrinkage toward the constant-correlation target (``RiskModel``).

    The target preserves each asset's sample variance but replaces every pairwise
    correlation with the average sample correlation ``r̄``. The analytic optimal
    shrinkage intensity follows Ledoit & Wolf (2003); this is a self-contained
    NumPy port (no sklearn target for this variant). The intensity is stored on
    :attr:`shrinkage_` after :meth:`estimate`.

    Parameters
    ----------
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(self, *, periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR) -> None:
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        self.shrinkage_: float | None = None

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized constant-correlation LW covariance."""
        values = returns_array(returns, min_periods=2)
        t, n = values.shape

        # Sample covariance (MLE, 1/T) about the sample mean.
        x = values - values.mean(axis=0, keepdims=True)
        sample = (x.T @ x) / t

        var = np.diag(sample).copy()
        std = np.sqrt(var)
        if np.any(std <= 0):
            # A zero-variance asset has no correlation structure to shrink toward;
            # fall back to the sample covariance (PSD-repaired downstream).
            self.shrinkage_ = 0.0
            return finalize_cov(sample * self.periods_per_year, n_assets=n)

        outer_std = np.outer(std, std)
        corr = sample / outer_std
        # Average off-diagonal correlation r̄.
        r_bar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
        target = r_bar * outer_std
        np.fill_diagonal(target, var)

        # --- Ledoit–Wolf (2003) optimal shrinkage intensity ---------------------
        # pi: sum of asymptotic variances of sample covariance entries.
        x2 = x**2
        pi_mat = (x2.T @ x2) / t - sample**2
        pi_hat = float(pi_mat.sum())

        # rho: sum of asymptotic covariances between sample entries and target.
        # Diagonal contributes pi_ii; off-diagonal uses the term derived in LW(2003).
        term = (x**3).T @ x / t - var[:, None] * sample
        theta_ij = term  # E[x_i^2 x_i x_j] style cross term, asymmetric on purpose

        rho_diag = float(np.diag(pi_mat).sum())
        # Off-diagonal contribution; r_bar/2 * (sqrt(var_j/var_i) term_ij + transpose).
        ratio = np.sqrt(np.outer(var, 1.0 / var))  # sqrt(var_i / var_j) in [i,j]
        rho_off_mat = (r_bar / 2.0) * (ratio * theta_ij.T + ratio.T * theta_ij)
        np.fill_diagonal(rho_off_mat, 0.0)
        rho_hat = rho_diag + float(rho_off_mat.sum())

        # gamma: Frobenius distance between sample and target.
        gamma_hat = float(np.sum((target - sample) ** 2))

        if gamma_hat <= 0:
            delta = 0.0
        else:
            kappa = (pi_hat - rho_hat) / gamma_hat
            delta = _clip_shrinkage(kappa / t)
        self.shrinkage_ = delta

        shrunk = delta * target + (1.0 - delta) * sample
        cov = shrunk * self.periods_per_year
        return finalize_cov(cov, n_assets=n)


class OAS:
    """Oracle-Approximating Shrinkage toward ``μ·I`` (Chen et al. 2010; ``RiskModel``).

    A refinement of Ledoit–Wolf with lower MSE under Gaussianity. Wraps
    scikit-learn's analytic :func:`sklearn.covariance.oas`. The intensity is
    stored on :attr:`shrinkage_` after :meth:`estimate`.

    Parameters
    ----------
    periods_per_year:
        Annualization factor (BUILD_PLAN §3.1).
    """

    def __init__(self, *, periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR) -> None:
        self.periods_per_year = validate_periods_per_year(periods_per_year)
        self.shrinkage_: float | None = None

    def estimate(self, returns: ReturnMatrix) -> np.ndarray:
        """Return the annualized OAS-shrunk covariance."""
        from sklearn.covariance import oas

        values = returns_array(returns, min_periods=2)
        n_assets = values.shape[1]
        try:
            cov_raw, shrink = oas(values)
        except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
            raise RiskModelError(f"OAS shrinkage failed: {exc}") from exc
        self.shrinkage_ = float(shrink)
        cov = np.asarray(cov_raw, dtype=float) * self.periods_per_year
        return finalize_cov(cov, n_assets=n_assets)


def ledoit_wolf(
    *,
    target: str = "constant_correlation",
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> LedoitWolfConstantCorrelation | LedoitWolfConstantVariance:
    """Factory for the ``"ledoit_wolf"`` risk model (BUILD_PLAN §5.2).

    Parameters
    ----------
    target:
        ``"constant_correlation"`` (default, the literature-recommended target) or
        ``"constant_variance"`` (scaled identity).
    periods_per_year:
        Annualization factor.

    Raises
    ------
    RiskModelError
        If ``target`` is not a recognized shrinkage target.
    """
    if target == "constant_correlation":
        return LedoitWolfConstantCorrelation(periods_per_year=periods_per_year)
    if target == "constant_variance":
        return LedoitWolfConstantVariance(periods_per_year=periods_per_year)
    raise RiskModelError(
        f"Unknown Ledoit-Wolf target {target!r}; use 'constant_correlation' or 'constant_variance'."
    )


def oas_shrinkage(*, periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR) -> OAS:
    """Factory for the ``"oas"`` risk model (BUILD_PLAN §5.2)."""
    return OAS(periods_per_year=periods_per_year)


__all__ = [
    "OAS",
    "LedoitWolfConstantCorrelation",
    "LedoitWolfConstantVariance",
    "ledoit_wolf",
    "oas_shrinkage",
]
