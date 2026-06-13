"""Ensemble construction — "diversifying the diversifiers" (Amenc–Goltz–Lodh–Martellini).

Any single weighting scheme can suffer severe short-term underperformance from
estimation / model risk. This constructor blends several
:class:`~riskbudget.core.interfaces.PortfolioConstructor` constituents (GMV,
Efficient-MSR, ERC, and — if injected — Agent 9's MDP / max-ENB) by **averaging
their normalized weights**. An optional **tracking-error-control overlay** shrinks
the blend toward a reference portfolio (equal- or cap-weight) when the ex-ante
tracking error against that reference exceeds a target — EDHEC's "Diversified
Multi-Strategy" / "Tracking the Tracking Error".

Degrades gracefully: a constituent that raises (e.g. MSR with no ``μ``, or an
absent Agent-9 method) is skipped, so the ensemble works with whatever
constructors are available. At least one must succeed.

Source: Amenc, Goltz, Lodh & Martellini (JPM 2012); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from riskbudget.core.errors import OptimizationError, RiskBudgetError
from riskbudget.core.interfaces import Constraints, PortfolioConstructor
from riskbudget.core.types import ExpectedReturns, Portfolio, RiskBudget

_ZERO_TOL = 1e-12


def _normalize(weights: np.ndarray) -> np.ndarray:
    """Normalize a weight vector to sum 1 (by its net sum)."""
    total = float(weights.sum())
    if abs(total) <= _ZERO_TOL:
        raise OptimizationError("Cannot normalize a zero-sum weight vector.")
    return weights / total


def _tracking_error(w: np.ndarray, ref: np.ndarray, sigma: np.ndarray) -> float:
    """Ex-ante tracking error ``√((w−ref)ᵀ Σ (w−ref))``."""
    d = w - ref
    var = float(d @ sigma @ d)
    return float(np.sqrt(max(var, 0.0)))


class EnsembleConstructor:
    """Blend several constructors' weights, with an optional TE-control overlay.

    Parameters
    ----------
    constituents:
        The :class:`PortfolioConstructor` instances to blend. Each is called with
        the same ``cov`` / ``mu`` / ``budget`` / ``constraints``; those that raise
        a :class:`RiskBudgetError` are skipped (graceful degradation).
    reference:
        Optional reference weights (keyed by asset) for the TE overlay; defaults to
        equal weight over the universe when the overlay is active and no reference
        is given.
    te_target:
        Optional ex-ante tracking-error budget. When set and the blend's TE against
        the reference exceeds it, the blend is shrunk toward the reference by the
        largest factor ``λ∈[0,1]`` that brings TE down to the target.
    """

    def __init__(
        self,
        constituents: Sequence[PortfolioConstructor],
        *,
        reference: dict[str, float] | None = None,
        te_target: float | None = None,
    ) -> None:
        if not constituents:
            raise OptimizationError("EnsembleConstructor requires at least one constituent.")
        self._constituents = list(constituents)
        self._reference = dict(reference) if reference else None
        if te_target is not None and (not np.isfinite(te_target) or te_target < 0):
            raise OptimizationError("te_target must be finite and non-negative.")
        self._te_target = te_target

    def _constituent_weights(
        self,
        cov: np.ndarray,
        order: list[str],
        mu: ExpectedReturns | None,
        budget: RiskBudget | None,
        constraints: Constraints,
    ) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for constructor in self._constituents:
            try:
                pf = constructor.construct(cov, mu=mu, budget=budget, constraints=constraints)
                vec = pf.as_array(order)
            except RiskBudgetError:
                continue  # skip constituents that can't run on these inputs
            vectors.append(_normalize(vec))
        if not vectors:
            raise OptimizationError(
                "No ensemble constituent could be constructed on the given inputs."
            )
        return vectors

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        """Return the blended (and optionally TE-controlled) portfolio.

        The asset order is taken from ``budget`` / ``mu`` when present, else a
        ``0..N-1`` default. Raises :class:`OptimizationError` if no constituent
        could be constructed.
        """
        sigma = np.asarray(cov, dtype=float)
        n = sigma.shape[0]
        if mu is not None:
            order = list(mu.assets)
        elif budget is not None:
            order = list(budget.assets)
        else:
            order = [str(i) for i in range(n)]

        vectors = self._constituent_weights(sigma, order, mu, budget, constraints)
        blend = _normalize(np.mean(vectors, axis=0))

        if self._te_target is not None:
            if self._reference is not None:
                ref = np.array([self._reference.get(a, 0.0) for a in order], dtype=float)
                ref = _normalize(ref)
            else:
                ref = np.full(n, 1.0 / n)
            te = _tracking_error(blend, ref, sigma)
            if te > self._te_target:
                # Shrink toward reference: w(λ) = λ·blend + (1−λ)·ref has TE λ·te.
                lam = self._te_target / te if te > _ZERO_TOL else 0.0
                blend = lam * blend + (1.0 - lam) * ref
                blend = _normalize(blend)

        leverage = 1.0 if constraints.leverage is None else float(constraints.leverage)
        blend = blend * leverage
        return Portfolio(dict(zip(order, (float(x) for x in blend), strict=True)))


def ensemble(
    constituents: Sequence[PortfolioConstructor],
    *,
    reference: dict[str, float] | None = None,
    te_target: float | None = None,
) -> EnsembleConstructor:
    """Factory for the ensemble constructor (§5.2 name ``ensemble``)."""
    return EnsembleConstructor(constituents, reference=reference, te_target=te_target)


__all__ = ["EnsembleConstructor", "ensemble"]
