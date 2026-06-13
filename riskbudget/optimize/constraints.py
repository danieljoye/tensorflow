"""Helpers for interpreting :class:`Constraints` in the solver layer.

The :class:`~riskbudget.core.interfaces.Constraints` dataclass is the frozen
contract (it lives in ``core``). This module adds the *interpretation* utilities
the solvers share: classifying a constraint set as separable (CCD-expressible)
vs. non-separable (needs cvxpy), resolving the target leverage, and building the
per-asset bound / group-cap / linear-constraint structures the convex and scipy
paths consume.

Source: pyrb constrained risk budgeting (Richard–Roncalli 2019); BUILD_PLAN §3.1,
§11, §12.1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from riskbudget.core.errors import OptimizationError
from riskbudget.core.interfaces import Constraints

# Default gross exposure when leverage is left unconstrained on a path that needs
# a concrete target (the risk-budget paths always normalize, so 1.0 is natural).
_DEFAULT_LEVERAGE = 1.0


def resolve_leverage(constraints: Constraints) -> float:
    """Target gross exposure to rescale weights to (defaults to 1.0)."""
    if constraints.leverage is None:
        return _DEFAULT_LEVERAGE
    return float(constraints.leverage)


def require_long_only(constraints: Constraints, *, context: str) -> None:
    """Raise :class:`OptimizationError` if ``constraints`` permits shorting.

    The log-barrier / CCD risk-budget solver requires ``w > 0`` (BUILD_PLAN
    §3.1): cross-sectional risk budgeting is *inherently long-only*. We must not
    silently long-only a shorting request — we raise instead.
    """
    if not constraints.long_only:
        raise OptimizationError(
            f"{context} is inherently long-only (log-barrier requires w > 0); "
            "Constraints.long_only=False is unsupported on this path. Use a "
            "classical constructor (GMV/MSR) for shorting."
        )
    if constraints.min_weight is not None and constraints.min_weight < 0:
        raise OptimizationError(
            f"{context} is inherently long-only; a negative min_weight "
            f"({constraints.min_weight}) would permit shorting and is unsupported."
        )


def has_group_caps(constraints: Constraints) -> bool:
    """Whether any non-trivial group cap is set (needs the cvxpy path)."""
    return bool(constraints.group_caps)


def is_separable(constraints: Constraints) -> bool:
    """Whether ``constraints`` are box-separable (CCD/scipy can express them).

    Separable = long-only + leverage target + optional uniform per-asset bounds.
    Group caps and turnover couple assets together and force the convex path.
    """
    if has_group_caps(constraints):
        return False
    return constraints.max_turnover is None


def resolve_bounds(constraints: Constraints, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset ``(lower, upper)`` weight bounds in *normalized* weight space.

    Bounds are expressed as fractions of the target leverage (i.e. on ``w`` after
    the final rescale). ``long_only`` pins the lower bound at ``0``. ``None``
    upper bounds become ``+inf``.
    """
    lower = np.zeros(n) if constraints.long_only else np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    if constraints.min_weight is not None:
        lower = np.maximum(lower, float(constraints.min_weight))
    if constraints.max_weight is not None:
        upper = np.minimum(upper, float(constraints.max_weight))
    if np.any(lower > upper):
        raise OptimizationError("Constraint bounds are infeasible (min_weight > max_weight).")
    return lower, upper


@dataclass(frozen=True)
class GroupCapSpec:
    """A resolved group cap as an index mask + cap value.

    ``Σ_{i in members} |wᵢ| <= cap`` (long-only here, so ``|wᵢ| = wᵢ``).
    """

    label: str
    members: tuple[int, ...]
    cap: float


def resolve_group_caps(constraints: Constraints, assets: Sequence[str]) -> list[GroupCapSpec]:
    """Translate ``group_caps`` + ``groups`` into index-based specs.

    The cap is scaled by the target leverage so it applies in normalized weight
    space. Assets with no group assignment are simply uncapped.
    """
    if not constraints.group_caps:
        return []
    index = {str(a): i for i, a in enumerate(assets)}
    leverage = resolve_leverage(constraints)
    specs: list[GroupCapSpec] = []
    for label, cap in constraints.group_caps.items():
        members = tuple(
            index[a] for a, g in constraints.groups.items() if g == label and a in index
        )
        specs.append(GroupCapSpec(label=str(label), members=members, cap=float(cap) * leverage))
    return specs


__all__ = [
    "GroupCapSpec",
    "has_group_caps",
    "is_separable",
    "require_long_only",
    "resolve_bounds",
    "resolve_group_caps",
    "resolve_leverage",
]
