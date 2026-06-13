"""Constant Proportion Portfolio Insurance (CPPI) — temporal risk budgeting.

The *temporal* sense of risk budgeting (BUILD_PLAN §2): dynamically split a risk
budget between a risky and a safe asset over time, subject to a floor or a
maximum-drawdown constraint. This follows the EDHEC course ``run_cppi``
formulation (BUILD_PLAN §11, *Introduction to Portfolio Construction and Analysis
with Python*, Martellini & Vaidyanathan; public ``edhec_risk_kit`` reference).

At each step, with account value ``A`` and protected ``floor``:

    cushion = (A − floor) / A
    risky_weight = clip(m × cushion, 0, leverage_cap)

The portfolio earns ``risky_weight`` of the risky return and ``1 − risky_weight``
of the safe return; the account compounds, the floor (optionally) ratchets, and
the loop repeats. Because the risky exposure is bounded by the cushion times a
finite multiplier, in continuous time the account never breaches the floor; in
discrete time a single-step jump larger than ``1 / m`` can pierce it, which is the
documented "gap risk" of CPPI.

Two floor variants are supported:

- **Fixed floor:** ``floor = floor_fraction × start_value`` (constant).
- **Drawdown floor:** ``floor = (1 − max_drawdown) × running_peak`` — the floor
  trails the running peak of the account, capping the peak-to-trough drawdown.

Return convention (BUILD_PLAN §3.1): inputs are *simple* periodic returns; the
account compounds multiplicatively. The safe asset may be an explicit per-period
return series or a constant *annualized* ``safe_rate`` (de-annualized by
``periods_per_year``). Determinism is inherited from the (already-realized) input
return paths — CPPI itself draws no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix

DEFAULT_PERIODS_PER_YEAR = 252

# Numerical guard: treat tiny magnitudes as zero (BUILD_PLAN §3.1).
_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class CPPIHistory:
    """Per-step CPPI bookkeeping for a single risky path.

    Every array has length ``n_steps`` (one row per realized return period).

    Attributes
    ----------
    account:
        Account value after each step (the wealth / equity curve).
    floor:
        Protected floor in effect *during* each step (ratchets in the drawdown
        variant).
    cushion:
        ``(account − floor) / account`` measured at the *start* of each step
        (the quantity the risky weight is computed from), clipped at 0.
    risky_weight:
        ``clip(m × cushion, 0, leverage_cap)`` applied during each step.
    peak:
        Running peak of the account (used by the drawdown floor; equals the
        running max of ``account`` for the fixed-floor variant too).
    """

    account: np.ndarray
    floor: np.ndarray
    cushion: np.ndarray
    risky_weight: np.ndarray
    peak: np.ndarray


def run_cppi_path(
    risky_returns: np.ndarray,
    safe_returns: np.ndarray,
    *,
    multiplier: float,
    floor_fraction: float,
    start_value: float = 1.0,
    max_drawdown: float | None = None,
    leverage_cap: float = 1.0,
) -> CPPIHistory:
    """Run CPPI over one realized risky path and return its step-by-step history.

    Parameters
    ----------
    risky_returns, safe_returns:
        Length-``T`` arrays of *simple* per-period returns for the risky and safe
        assets. Must be the same length and finite.
    multiplier:
        The CPPI multiplier ``m`` (``> 0``). Risky exposure is ``m × cushion``.
    floor_fraction:
        Protected wealth fraction of ``start_value`` for the fixed-floor variant
        (``0 ≤ floor_fraction < 1``). For the drawdown variant this is the
        *initial* floor; the trailing ``(1 − max_drawdown)`` floor takes over once
        it rises above it.
    start_value:
        Initial account value (``> 0``).
    max_drawdown:
        If set (``0 < max_drawdown ≤ 1``), use the drawdown floor
        ``(1 − max_drawdown) × running_peak``; otherwise the fixed floor.
    leverage_cap:
        Upper bound on the risky weight (``≥ 0``). ``1.0`` (default) forbids
        leverage; ``> 1`` permits a capped levered risky tilt.

    Returns
    -------
    CPPIHistory
        Account / floor / cushion / risky-weight / peak arrays of length ``T``.

    Raises
    ------
    ValidationError
        On mismatched lengths, non-finite inputs, or out-of-range parameters.
    """
    risky = np.asarray(risky_returns, dtype=float)
    safe = np.asarray(safe_returns, dtype=float)
    if risky.ndim != 1 or safe.ndim != 1:
        raise ValidationError("run_cppi_path expects 1-D return arrays.")
    if risky.shape != safe.shape:
        raise ValidationError(
            f"risky and safe returns must have equal length, got {risky.shape} vs {safe.shape}."
        )
    if risky.size == 0:
        raise ValidationError("run_cppi_path needs at least one return period.")
    if not np.isfinite(risky).all() or not np.isfinite(safe).all():
        raise ValidationError("CPPI return inputs contain NaN or infinite values.")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValidationError("multiplier must be finite and positive.")
    if not np.isfinite(floor_fraction) or not (0.0 <= floor_fraction < 1.0):
        raise ValidationError("floor_fraction must be finite and in [0, 1).")
    if not np.isfinite(start_value) or start_value <= 0:
        raise ValidationError("start_value must be finite and positive.")
    if max_drawdown is not None and (
        not np.isfinite(max_drawdown) or not (0.0 < max_drawdown <= 1.0)
    ):
        raise ValidationError("max_drawdown must be finite and in (0, 1] when set.")
    if not np.isfinite(leverage_cap) or leverage_cap < 0:
        raise ValidationError("leverage_cap must be finite and non-negative.")

    n = risky.size
    account = np.empty(n, dtype=float)
    floor_hist = np.empty(n, dtype=float)
    cushion_hist = np.empty(n, dtype=float)
    weight_hist = np.empty(n, dtype=float)
    peak_hist = np.empty(n, dtype=float)

    value = float(start_value)
    peak = float(start_value)
    base_floor = floor_fraction * start_value

    for t in range(n):
        peak = max(peak, value)
        floor_value = (1.0 - max_drawdown) * peak if max_drawdown is not None else base_floor
        cushion = 0.0 if value <= _ZERO_TOL else max(0.0, (value - floor_value) / value)

        weight = float(np.clip(multiplier * cushion, 0.0, leverage_cap))
        step_return = weight * risky[t] + (1.0 - weight) * safe[t]
        value = value * (1.0 + step_return)

        account[t] = value
        floor_hist[t] = floor_value
        cushion_hist[t] = cushion
        weight_hist[t] = weight
        peak_hist[t] = max(peak, value)

    return CPPIHistory(
        account=account,
        floor=floor_hist,
        cushion=cushion_hist,
        risky_weight=weight_hist,
        peak=peak_hist,
    )


def _deannualize(rate: float, periods_per_year: int | float) -> float:
    """Convert an annualized simple rate to its per-period equivalent."""
    return float((1.0 + rate) ** (1.0 / periods_per_year) - 1.0)


def _resolve_safe_series(
    safe: ReturnMatrix | float,
    *,
    n_steps: int,
    index: pd.Index,
    safe_rate: float,
    periods_per_year: int | float,
) -> np.ndarray:
    """Resolve ``safe`` into a length-``n_steps`` array of per-period returns."""
    if isinstance(safe, ReturnMatrix):
        frame = safe.frame
        if frame.shape[1] != 1:
            raise ValidationError(
                f"CPPI safe ReturnMatrix must have exactly one column (got {frame.shape[1]})."
            )
        if frame.shape[0] != n_steps:
            raise ValidationError(
                f"safe series length {frame.shape[0]} does not match risky length {n_steps}."
            )
        return frame.to_numpy(dtype=float).reshape(-1)

    rate = float(safe)
    if not np.isfinite(rate):
        raise ValidationError("scalar safe rate must be finite.")
    # A scalar safe input is treated as an annualized rate (BUILD_PLAN §3.1).
    per_period = _deannualize(rate, periods_per_year)
    return np.full(n_steps, per_period, dtype=float)


class CPPI:
    """Constant Proportion Portfolio Insurance allocator (implements ``Allocator``).

    Splits wealth between one risky asset and a safe asset so the risky exposure
    is ``m × cushion`` each period (EDHEC ``run_cppi``; BUILD_PLAN §2, §11). The
    ``risky`` :class:`ReturnMatrix` must carry a single asset column (the risky
    asset); for running CPPI across many scenario columns use
    :func:`riskbudget.dynamic.allocators.bt_mix`.

    Parameters
    ----------
    leverage_cap:
        Upper bound on the risky weight (``≥ 0``); ``1.0`` forbids leverage.
    periods_per_year:
        Annualization factor used to de-annualize a scalar ``safe`` rate and to
        annualize reported metrics (BUILD_PLAN §3.1).
    """

    def __init__(
        self,
        *,
        leverage_cap: float = 1.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if not np.isfinite(leverage_cap) or leverage_cap < 0:
            raise ValidationError("leverage_cap must be finite and non-negative.")
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValidationError("periods_per_year must be finite and positive.")
        self.leverage_cap = float(leverage_cap)
        self.periods_per_year = periods_per_year

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Run CPPI on the single risky asset in ``risky``.

        ``risky`` returns are *simple* per-period returns. ``safe`` is either a
        single-column :class:`ReturnMatrix` aligned to ``risky`` or a scalar
        *annualized* rate (defaults to ``params.safe_rate`` semantics; pass the
        rate explicitly). Returns a :class:`BacktestResult` whose ``equity_curve``
        is the CPPI account, ``weights`` is the risky/safe split through time, and
        ``diagnostics`` carries the floor / cushion / peak series.

        Raises
        ------
        BacktestError
            If the CPPI run cannot be produced.
        """
        if not isinstance(risky, ReturnMatrix):
            raise BacktestError("CPPI.allocate expects a ReturnMatrix for `risky`.")
        if risky.shape[1] != 1:
            raise BacktestError(
                "CPPI.allocate operates on a single risky asset; "
                f"`risky` has {risky.shape[1]} columns. Use bt_mix for scenario sets."
            )

        try:
            risky_returns = risky.values.reshape(-1)
            index = risky.dates
            n_steps = risky_returns.size
            safe_returns = _resolve_safe_series(
                safe,
                n_steps=n_steps,
                index=index,
                safe_rate=params.safe_rate,
                periods_per_year=self.periods_per_year,
            )
            history = run_cppi_path(
                risky_returns,
                safe_returns,
                multiplier=params.multiplier,
                floor_fraction=params.floor,
                start_value=params.start_value,
                max_drawdown=params.max_drawdown,
                leverage_cap=self.leverage_cap,
            )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"CPPI allocation failed: {exc}") from exc

        return _history_to_result(
            history,
            index=index,
            risky_asset=risky.assets[0],
            params=params,
            periods_per_year=self.periods_per_year,
            leverage_cap=self.leverage_cap,
            strategy="cppi",
        )


def _history_to_result(
    history: CPPIHistory,
    *,
    index: pd.Index,
    risky_asset: str,
    params: AllocatorParams,
    periods_per_year: int | float,
    leverage_cap: float,
    strategy: str,
) -> BacktestResult:
    """Assemble a :class:`BacktestResult` from a CPPI history."""
    equity = pd.Series(history.account, index=index, name="account")

    safe_col = "SAFE"
    risky_col = str(risky_asset)
    if safe_col == risky_col:
        safe_col = "SAFE_ASSET"
    weights = pd.DataFrame(
        {
            risky_col: history.risky_weight,
            safe_col: 1.0 - history.risky_weight,
        },
        index=index,
    )

    # Per-period portfolio simple returns from the account path.
    start_value = float(params.start_value)
    prev = np.concatenate(([start_value], history.account[:-1]))
    returns = pd.Series(history.account / prev - 1.0, index=index, name="returns")

    breached = bool(np.any(history.account < history.floor - 1e-9))
    metrics = {
        "terminal_wealth": float(history.account[-1]),
        "min_account": float(np.min(history.account)),
        "floor_breached": float(breached),
        "max_risky_weight": float(np.max(history.risky_weight)),
        "mean_risky_weight": float(np.mean(history.risky_weight)),
    }

    diagnostics = {
        "floor": pd.Series(history.floor, index=index, name="floor"),
        "cushion": pd.Series(history.cushion, index=index, name="cushion"),
        "peak": pd.Series(history.peak, index=index, name="peak"),
        "risky_weight": pd.Series(history.risky_weight, index=index, name="risky_weight"),
    }

    metadata = {
        "strategy": strategy,
        "multiplier": float(params.multiplier),
        "floor_fraction": float(params.floor),
        "max_drawdown": params.max_drawdown,
        "start_value": start_value,
        "leverage_cap": float(leverage_cap),
        "periods_per_year": float(periods_per_year),
        "risky_asset": risky_col,
        "safe_asset": safe_col,
    }

    return BacktestResult(
        equity_curve=equity,
        weights=weights,
        returns=returns,
        metrics=metrics,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def cppi(
    *,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> CPPI:
    """Factory for the ``"cppi"`` allocator (BUILD_PLAN §5.2)."""
    return CPPI(leverage_cap=leverage_cap, periods_per_year=periods_per_year)


__all__ = [
    "CPPI",
    "CPPIHistory",
    "cppi",
    "run_cppi_path",
]
