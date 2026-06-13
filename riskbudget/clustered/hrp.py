"""Hierarchical Risk Parity (HRP) constructor (BUILD_PLAN §12.6).

HRP allocates capital with **no matrix inversion**, so it remains well behaved on
ill-conditioned or singular covariance matrices where Markowitz-style optimisers
break down. The recipe (López de Prado, "Building Diversified Portfolios that
Outperform Out of Sample", *Journal of Portfolio Management*, 2016):

1. **Tree clustering.** Form the correlation-distance matrix
   ``Dᵢⱼ = √(½(1 − ρᵢⱼ))`` and hierarchically cluster it with
   :func:`scipy.cluster.hierarchy.linkage`.
2. **Quasi-diagonalisation.** Reorder assets by the dendrogram leaf order
   (:func:`scipy.cluster.hierarchy.leaves_list`) so that similar assets sit
   adjacently — this places large covariances near the diagonal.
3. **Recursive bisection.** Walk down the (re-ordered) tree splitting capital
   between the two halves inversely to each half's cluster variance, using
   inverse-variance weights within a cluster.

This module is a :class:`~riskbudget.core.interfaces.PortfolioConstructor`. v1 is
plain HRP with the variance risk measure; NCO / HERC and alternative risk measures
are roadmap (BUILD_PLAN §10).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget

LinkageMethod = Literal["single", "complete", "average", "ward"]

_ZERO_TOL = 1e-12


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
    if np.any(np.diag(sigma) < -_ZERO_TOL):
        raise OptimizationError("Covariance has a negative variance on the diagonal.")
    return np.asarray(0.5 * (sigma + sigma.T), dtype=float)


def _resolve_assets(n: int, assets: Sequence[str] | None) -> list[str]:
    """Pick a canonical asset ordering of length ``n``."""
    if assets is None:
        return [str(i) for i in range(n)]
    order = [str(a) for a in assets]
    if len(order) != n:
        raise OptimizationError(
            f"Asset labels ({len(order)}) do not match covariance dimension ({n})."
        )
    return order


def _correlation_distance(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(corr, dist)`` with ``Dᵢⱼ = √(½(1 − ρᵢⱼ))``.

    Assets with zero variance are treated as fully decorrelated from everything
    (correlation 0 off-diagonal) so the distance stays well defined without any
    inversion. Negative variances are rejected upstream.
    """
    variances = np.diag(cov)
    std = np.sqrt(np.clip(variances, 0.0, None))
    nonzero = std > _ZERO_TOL
    inv = np.where(nonzero, 1.0 / np.where(nonzero, std, 1.0), 0.0)
    corr = inv[:, None] * cov * inv[None, :]
    # Pin the diagonal and clip float drift into the valid correlation range.
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return corr, dist


def _quasi_diag_order(dist: np.ndarray, method: LinkageMethod) -> list[int]:
    """Hierarchically cluster ``dist`` and return the dendrogram leaf order."""
    n = dist.shape[0]
    if n == 1:
        return [0]
    # scipy.linkage wants a condensed (upper-triangular) distance vector.
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=method)
    return [int(i) for i in leaves_list(link)]


def _inverse_variance_weights(cov_block: np.ndarray) -> np.ndarray:
    """Inverse-variance weights over the diagonal of ``cov_block`` (no inversion)."""
    ivp = 1.0 / np.clip(np.diag(cov_block), _ZERO_TOL, None)
    return np.asarray(ivp / ivp.sum(), dtype=float)


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Variance of the inverse-variance sub-portfolio over ``items``."""
    block = cov[np.ix_(items, items)]
    w = _inverse_variance_weights(block)
    return float(w @ block @ w)


def _recursive_bisection(cov: np.ndarray, sorted_items: list[int]) -> np.ndarray:
    """Split capital top-down inversely to each cluster's variance."""
    n = cov.shape[0]
    weights = np.ones(n)
    clusters: list[list[int]] = [sorted_items]

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            total = var_left + var_right
            # If both halves are riskless, split evenly.
            alpha = 0.5 if total <= _ZERO_TOL else 1.0 - var_left / total
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters

    return weights


@dataclass(frozen=True)
class HRPConstructor:
    """Hierarchical Risk Parity portfolio constructor (López de Prado 2016).

    Parameters
    ----------
    linkage_method:
        Agglomeration method passed to :func:`scipy.cluster.hierarchy.linkage`
        (``"single"`` default — the classic López de Prado choice; ``"ward"``,
        ``"average"``, ``"complete"`` also supported).
    """

    linkage_method: LinkageMethod = "single"

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
        assets: Sequence[str] | None = None,
    ) -> Portfolio:
        """Return the HRP portfolio for ``cov``.

        Weights are non-negative and sum to one. ``mu`` and ``budget`` are accepted
        for interface uniformity but unused (plain HRP is variance-driven). HRP is
        long-only by construction; a request to permit shorting
        (``constraints.long_only=False``) is rejected.
        """
        if not constraints.long_only:
            raise OptimizationError(
                "HRP is long-only by construction; set constraints.long_only=True."
            )
        sigma = _validate_cov(cov)
        n = sigma.shape[0]
        # Prefer the budget's ordering if it carries one and no explicit list given.
        if assets is None and budget is not None:
            order = budget.assets
            if len(order) != n:
                raise OptimizationError(
                    f"Budget assets ({len(order)}) do not match covariance dimension ({n})."
                )
        else:
            order = _resolve_assets(n, assets)

        if n == 1:
            return Portfolio({order[0]: 1.0})

        _corr, dist = _correlation_distance(sigma)
        sort_ix = _quasi_diag_order(dist, self.linkage_method)
        weights = _recursive_bisection(sigma, sort_ix)

        weights = np.where(np.abs(weights) < _ZERO_TOL, 0.0, weights)
        total = float(np.sum(weights))
        if total <= _ZERO_TOL:
            raise OptimizationError("HRP produced an all-zero weight vector.")
        weights = weights / total
        return Portfolio({a: float(w) for a, w in zip(order, weights, strict=True)})


def hrp(linkage_method: LinkageMethod = "single") -> HRPConstructor:
    """Factory for the ``"hrp"`` method: a Hierarchical Risk Parity constructor."""
    return HRPConstructor(linkage_method=linkage_method)


__all__ = ["HRPConstructor", "LinkageMethod", "hrp"]
