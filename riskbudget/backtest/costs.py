"""Transaction-cost models for the walk-forward backtester (BUILD_PLAN §3.1, §7).

The v1 model is a **proportional** cost on turnover: rebalancing from a weight
vector ``w_prev`` to ``w_new`` incurs a cost of ``rate * turnover``, where
``turnover`` is the one-sided sum of absolute weight changes
``½ · Σᵢ |w_newᵢ − w_prevᵢ|`` by default (the standard "one-way" turnover: a
round-trip that sells one asset to buy another counts the traded notional once).

Costs are expressed in **basis points of traded notional** (1 bp = 1e-4) and are
deducted from the portfolio's gross return at each rebalance, so higher turnover
and higher cost rates monotonically reduce net returns (BUILD_PLAN §7 acceptance:
"turnover and costs reduce returns").

This module is intentionally light: the backtester (``engine.py``) owns the
turnover bookkeeping across rebalances and calls a :class:`CostModel` to convert a
turnover figure into a return drag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from riskbudget.core.errors import BacktestError


def compute_turnover(
    prev_weights: np.ndarray,
    new_weights: np.ndarray,
    *,
    one_way: bool = True,
) -> float:
    """Return the turnover between two weight vectors.

    Parameters
    ----------
    prev_weights, new_weights:
        Aligned weight vectors (same asset ordering, same length). The
        pre-rebalance book is assumed to have *drifted* to ``prev_weights`` (the
        engine supplies the drifted weights), so the trade is
        ``new_weights − prev_weights``.
    one_way:
        If ``True`` (default) return the one-sided traded notional
        ``max(Σ buys, Σ sells)``. For gross-preserving rebalances this equals the
        textbook ``½ · Σ |Δw|``; when gross exposure changes (leverage up/down,
        initial cash deployment) it correctly counts the larger traded leg. If
        ``False`` return the gross ``Σ |Δw|`` (two-way) figure.

    Returns
    -------
    float
        Non-negative turnover as a fraction of portfolio notional.

    Raises
    ------
    BacktestError
        If the two vectors differ in length or contain non-finite values.
    """
    prev = np.asarray(prev_weights, dtype=float)
    new = np.asarray(new_weights, dtype=float)
    if prev.shape != new.shape:
        raise BacktestError(
            f"Turnover needs aligned weight vectors; got {prev.shape} vs {new.shape}."
        )
    if not (np.isfinite(prev).all() and np.isfinite(new).all()):
        raise BacktestError("Turnover inputs contain non-finite weights.")
    delta = new - prev
    gross = float(np.abs(delta).sum())
    if not one_way:
        return gross
    # One-way turnover = max(buys, sells): the one-side traded notional. When the
    # rebalance preserves gross exposure (buys == sells) this equals the textbook
    # 0.5 * sum(|dw|); when leverage changes (e.g. a vol-target re-lever from 1x to
    # 2x buys 1.0 NAV against zero sells) the 0.5 convention would undercount the
    # traded notional by half, so we charge the larger leg instead.
    buys = float(np.clip(delta, 0.0, None).sum())
    sells = float(np.clip(-delta, 0.0, None).sum())
    return max(buys, sells)


@runtime_checkable
class CostModel(Protocol):
    """Converts a rebalance (prev → new weights) into a fractional return drag."""

    def cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Return the cost of the trade as a fraction of portfolio value."""
        ...


@dataclass(frozen=True)
class ProportionalCost:
    """Proportional transaction cost: ``rate * turnover`` per rebalance.

    Attributes
    ----------
    bps:
        Cost rate in **basis points** of one-way traded notional (1 bp = 1e-4).
        ``10.0`` means 10 bps (0.1%) charged on the one-sided turnover. Must be
        finite and non-negative.
    one_way:
        Whether turnover is measured one-way (``½ Σ|Δw|``, default) or two-way.

    Notes
    -----
    The drag is ``(bps · 1e-4) · turnover`` and is subtracted from the portfolio
    return in the rebalance period. With ``bps == 0`` the model is a no-op, which
    the backtester uses as its costless baseline.
    """

    bps: float = 0.0
    one_way: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.bps) or self.bps < 0:
            raise BacktestError("ProportionalCost.bps must be finite and non-negative.")

    @property
    def rate(self) -> float:
        """The cost rate as a plain fraction (``bps · 1e-4``)."""
        return float(self.bps) * 1e-4

    def cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Return ``rate · turnover`` for the trade ``prev → new``."""
        turnover = compute_turnover(prev_weights, new_weights, one_way=self.one_way)
        return self.rate * turnover


def proportional_cost(bps: float = 0.0, *, one_way: bool = True) -> ProportionalCost:
    """Registry-friendly factory for :class:`ProportionalCost` (BUILD_PLAN §5.2).

    Parameters
    ----------
    bps:
        Proportional cost in basis points of one-way traded notional.
    one_way:
        Turnover convention (see :class:`ProportionalCost`).
    """
    return ProportionalCost(bps=bps, one_way=one_way)


def no_cost() -> ProportionalCost:
    """Factory for the costless baseline model (0 bps)."""
    return ProportionalCost(bps=0.0)


__all__ = [
    "CostModel",
    "ProportionalCost",
    "compute_turnover",
    "no_cost",
    "proportional_cost",
]
