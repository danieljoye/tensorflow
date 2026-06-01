"""Pluggable dynamic allocators and a scenario driver (BUILD_PLAN §2, §11).

The *temporal* sense of risk budgeting: each allocator decides, period by period,
how to split wealth between a risky asset and a safe asset, then compounds the
realized return. Every allocator here implements the
:class:`~riskbudget.core.interfaces.Allocator` protocol
(``allocate(risky, safe, params) -> BacktestResult``) and follows the EDHEC course
allocator formulations (BUILD_PLAN §11, *Introduction to Portfolio Construction and
Analysis with Python*, Martellini & Vaidyanathan; public ``edhec_risk_kit``).

Allocators provided
-------------------
- :class:`FixedMixAllocator` — constant risky/safe weights every period.
- :class:`GlidepathAllocator` — linear shift from a start weight to an end weight.
  Supports a **state-dependent** variant (Martellini–Milhau life-cycle): the risky
  weight is driven by the *funding cushion* rather than the calendar.
- :class:`FloorAllocator` — CPPI-style fixed/drawdown floor (delegates to
  :func:`riskbudget.dynamic.cppi.run_cppi_path`).
- :class:`DrawdownAllocator` — a maximum-drawdown floor (CPPI with a trailing-peak
  floor), exposed as its own named allocator.

Plus :func:`bt_mix`, a driver that runs any allocator over a *set* of scenarios
(e.g. GBM price paths from :mod:`riskbudget.simulate.gbm`) and returns the per-
scenario terminal wealth and an aggregate :class:`BacktestResult`.

Return convention (BUILD_PLAN §3.1): inputs are *simple* per-period returns; the
account compounds multiplicatively; a scalar ``safe`` is an *annualized* rate that
is de-annualized by ``periods_per_year``. Allocators draw no randomness — any
determinism is inherited from the (already-realized) return paths handed in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix
from riskbudget.dynamic.cppi import (
    DEFAULT_PERIODS_PER_YEAR,
    CPPIHistory,
    _history_to_result,
    _resolve_safe_series,
    run_cppi_path,
)

__all__ = [
    "DrawdownAllocator",
    "FixedMixAllocator",
    "FloorAllocator",
    "GlidepathAllocator",
    "bt_mix",
    "drawdown",
    "fixed_mix",
    "floor",
    "glidepath",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _single_risky_returns(risky: ReturnMatrix) -> tuple[np.ndarray, pd.Index, str]:
    """Extract the single risky-asset return vector, index, and asset id."""
    if not isinstance(risky, ReturnMatrix):
        raise BacktestError("Allocator.allocate expects a ReturnMatrix for `risky`.")
    if risky.shape[1] != 1:
        raise BacktestError(
            "This allocator operates on a single risky asset; "
            f"`risky` has {risky.shape[1]} columns. Use bt_mix for scenario sets."
        )
    return risky.values.reshape(-1), risky.dates, risky.assets[0]


def _run_weighted_path(
    risky_returns: np.ndarray,
    safe_returns: np.ndarray,
    weights: np.ndarray,
    *,
    start_value: float,
    floor_fraction: float,
    max_drawdown: float | None,
) -> CPPIHistory:
    """Compound a path under an exogenous risky-weight schedule.

    Used by the fixed-mix and glidepath allocators (whose weights do not depend on
    the cushion). The floor / cushion bookkeeping is filled in for reporting parity
    with CPPI but does not feed back into the weights.
    """
    n = risky_returns.size
    account = np.empty(n, dtype=float)
    floor_hist = np.empty(n, dtype=float)
    cushion_hist = np.empty(n, dtype=float)
    peak_hist = np.empty(n, dtype=float)

    value = float(start_value)
    peak = float(start_value)
    base_floor = floor_fraction * start_value

    for t in range(n):
        peak = max(peak, value)
        floor_value = (1.0 - max_drawdown) * peak if max_drawdown is not None else base_floor
        cushion = max(0.0, (value - floor_value) / value) if value > 0 else 0.0

        w = float(weights[t])
        step_return = w * risky_returns[t] + (1.0 - w) * safe_returns[t]
        value = value * (1.0 + step_return)

        account[t] = value
        floor_hist[t] = floor_value
        cushion_hist[t] = cushion
        peak_hist[t] = max(peak, value)

    return CPPIHistory(
        account=account,
        floor=floor_hist,
        cushion=cushion_hist,
        risky_weight=np.asarray(weights, dtype=float),
        peak=peak_hist,
    )


# ---------------------------------------------------------------------------
# Fixed-mix allocator
# ---------------------------------------------------------------------------


class FixedMixAllocator:
    """Constant-weight (``risky_weight`` every period) allocator.

    The simplest temporal allocation: hold a fixed fraction in the risky asset and
    the remainder in the safe asset, rebalancing back to the target each period.
    A 60/40 book is ``FixedMixAllocator(risky_weight=0.6)``.

    Parameters
    ----------
    risky_weight:
        Constant risky-asset weight in ``[0, leverage_cap]``.
    leverage_cap:
        Upper bound enforced on ``risky_weight`` (``>= 0``); ``1.0`` forbids
        leverage.
    periods_per_year:
        Annualization factor used to de-annualize a scalar ``safe`` rate.
    """

    def __init__(
        self,
        *,
        risky_weight: float = 0.6,
        leverage_cap: float = 1.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if not np.isfinite(leverage_cap) or leverage_cap < 0:
            raise ValidationError("leverage_cap must be finite and non-negative.")
        if not np.isfinite(risky_weight) or not (0.0 <= risky_weight <= leverage_cap):
            raise ValidationError(
                f"risky_weight must be finite and in [0, leverage_cap={leverage_cap}]."
            )
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValidationError("periods_per_year must be finite and positive.")
        self.risky_weight = float(risky_weight)
        self.leverage_cap = float(leverage_cap)
        self.periods_per_year = periods_per_year

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Hold a constant risky weight; see module docstring for conventions."""
        risky_returns, index, asset = _single_risky_returns(risky)
        try:
            safe_returns = _resolve_safe_series(
                safe,
                n_steps=risky_returns.size,
                index=index,
                safe_rate=params.safe_rate,
                periods_per_year=self.periods_per_year,
            )
            weights = np.full(risky_returns.size, self.risky_weight, dtype=float)
            history = _run_weighted_path(
                risky_returns,
                safe_returns,
                weights,
                start_value=params.start_value,
                floor_fraction=params.floor,
                max_drawdown=params.max_drawdown,
            )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"Fixed-mix allocation failed: {exc}") from exc

        return _history_to_result(
            history,
            index=index,
            risky_asset=asset,
            params=params,
            periods_per_year=self.periods_per_year,
            leverage_cap=self.leverage_cap,
            strategy="fixed_mix",
        )


# ---------------------------------------------------------------------------
# Glidepath allocator (calendar + state-dependent)
# ---------------------------------------------------------------------------


class GlidepathAllocator:
    """Linear glidepath from a start risky weight to an end risky weight.

    Two modes:

    - **Calendar** (default): the risky weight slides linearly from
      ``start_weight`` at the first period to ``end_weight`` at the last,
      independent of realized returns. This is the classic target-date-fund
      "de-risking" glidepath.
    - **State-dependent** (``state_dependent=True``): the risky weight is set by
      the *funding cushion* rather than the calendar — ``w = start_weight +
      (end_weight − start_weight) × clip(cushion, 0, 1)`` where ``cushion =
      (account − floor) / account``. This is the Martellini–Milhau life-cycle
      idea: take more risk when the funding cushion is comfortable, de-risk as it
      thins (BUILD_PLAN §2, §11).

    Parameters
    ----------
    start_weight, end_weight:
        Risky weights at the path endpoints (calendar) or cushion endpoints
        (state-dependent), each in ``[0, leverage_cap]``.
    state_dependent:
        Select the cushion-driven variant.
    leverage_cap:
        Upper bound on the risky weight (``>= 0``).
    periods_per_year:
        Annualization factor for a scalar ``safe`` rate.
    """

    def __init__(
        self,
        *,
        start_weight: float = 0.8,
        end_weight: float = 0.2,
        state_dependent: bool = False,
        leverage_cap: float = 1.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if not np.isfinite(leverage_cap) or leverage_cap < 0:
            raise ValidationError("leverage_cap must be finite and non-negative.")
        for name, w in (("start_weight", start_weight), ("end_weight", end_weight)):
            if not np.isfinite(w) or not (0.0 <= w <= leverage_cap):
                raise ValidationError(
                    f"{name} must be finite and in [0, leverage_cap={leverage_cap}]."
                )
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValidationError("periods_per_year must be finite and positive.")
        self.start_weight = float(start_weight)
        self.end_weight = float(end_weight)
        self.state_dependent = bool(state_dependent)
        self.leverage_cap = float(leverage_cap)
        self.periods_per_year = periods_per_year

    def _calendar_weights(self, n: int) -> np.ndarray:
        """Linearly interpolated weights ``start -> end`` over ``n`` periods."""
        if n == 1:
            return np.array([self.start_weight], dtype=float)
        frac = np.linspace(0.0, 1.0, n)
        return self.start_weight + (self.end_weight - self.start_weight) * frac

    def _run_state_dependent(
        self,
        risky_returns: np.ndarray,
        safe_returns: np.ndarray,
        *,
        start_value: float,
        floor_fraction: float,
        max_drawdown: float | None,
    ) -> CPPIHistory:
        """Cushion-driven glidepath: weight scales with the funding cushion."""
        n = risky_returns.size
        account = np.empty(n, dtype=float)
        floor_hist = np.empty(n, dtype=float)
        cushion_hist = np.empty(n, dtype=float)
        weight_hist = np.empty(n, dtype=float)
        peak_hist = np.empty(n, dtype=float)

        value = float(start_value)
        peak = float(start_value)
        base_floor = floor_fraction * start_value
        span = self.end_weight - self.start_weight

        for t in range(n):
            peak = max(peak, value)
            floor_value = (1.0 - max_drawdown) * peak if max_drawdown is not None else base_floor
            cushion = max(0.0, (value - floor_value) / value) if value > 0 else 0.0

            w = self.start_weight + span * min(1.0, cushion)
            w = float(np.clip(w, 0.0, self.leverage_cap))
            step_return = w * risky_returns[t] + (1.0 - w) * safe_returns[t]
            value = value * (1.0 + step_return)

            account[t] = value
            floor_hist[t] = floor_value
            cushion_hist[t] = cushion
            weight_hist[t] = w
            peak_hist[t] = max(peak, value)

        return CPPIHistory(
            account=account,
            floor=floor_hist,
            cushion=cushion_hist,
            risky_weight=weight_hist,
            peak=peak_hist,
        )

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Run the glidepath (calendar or cushion-driven)."""
        risky_returns, index, asset = _single_risky_returns(risky)
        try:
            safe_returns = _resolve_safe_series(
                safe,
                n_steps=risky_returns.size,
                index=index,
                safe_rate=params.safe_rate,
                periods_per_year=self.periods_per_year,
            )
            if self.state_dependent:
                history = self._run_state_dependent(
                    risky_returns,
                    safe_returns,
                    start_value=params.start_value,
                    floor_fraction=params.floor,
                    max_drawdown=params.max_drawdown,
                )
            else:
                weights = np.clip(
                    self._calendar_weights(risky_returns.size), 0.0, self.leverage_cap
                )
                history = _run_weighted_path(
                    risky_returns,
                    safe_returns,
                    weights,
                    start_value=params.start_value,
                    floor_fraction=params.floor,
                    max_drawdown=params.max_drawdown,
                )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"Glidepath allocation failed: {exc}") from exc

        result = _history_to_result(
            history,
            index=index,
            risky_asset=asset,
            params=params,
            periods_per_year=self.periods_per_year,
            leverage_cap=self.leverage_cap,
            strategy="glidepath",
        )
        result.metadata["start_weight"] = self.start_weight
        result.metadata["end_weight"] = self.end_weight
        result.metadata["state_dependent"] = self.state_dependent
        return result


# ---------------------------------------------------------------------------
# Floor allocator (CPPI fixed floor) and drawdown allocator
# ---------------------------------------------------------------------------


class FloorAllocator:
    """CPPI-style floor allocator: risky weight ``= clip(m × cushion, 0, cap)``.

    Identical mechanics to :class:`riskbudget.dynamic.cppi.CPPI` with a *fixed*
    floor; exposed as its own named allocator (``"floor"``) so the registry can
    select a floor strategy distinct from the drawdown one. The
    cushion/multiplier come from :class:`AllocatorParams`.

    Parameters
    ----------
    leverage_cap:
        Upper bound on the risky weight (``>= 0``).
    periods_per_year:
        Annualization factor for a scalar ``safe`` rate.
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
        """Run CPPI with the fixed floor from ``params`` (ignores ``max_drawdown``)."""
        risky_returns, index, asset = _single_risky_returns(risky)
        try:
            safe_returns = _resolve_safe_series(
                safe,
                n_steps=risky_returns.size,
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
                max_drawdown=None,  # fixed-floor variant
                leverage_cap=self.leverage_cap,
            )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"Floor allocation failed: {exc}") from exc

        return _history_to_result(
            history,
            index=index,
            risky_asset=asset,
            params=params,
            periods_per_year=self.periods_per_year,
            leverage_cap=self.leverage_cap,
            strategy="floor",
        )


class DrawdownAllocator:
    """Maximum-drawdown allocator: trailing-peak floor ``(1 − maxDD) × peak``.

    A CPPI whose floor ratchets up with the running peak of the account so the
    peak-to-trough drawdown is bounded by ``max_drawdown`` (gap risk aside). The
    limit comes from ``params.max_drawdown`` (or the ``max_drawdown`` passed to the
    factory, which overrides it). Risky weight ``= clip(m × cushion, 0, cap)`` with
    ``cushion = (account − (1 − maxDD)·peak) / account``.

    Parameters
    ----------
    max_drawdown:
        Optional explicit drawdown limit in ``(0, 1]`` overriding
        ``params.max_drawdown``. When ``None`` the allocator uses
        ``params.max_drawdown`` and raises if that is also unset.
    leverage_cap:
        Upper bound on the risky weight (``>= 0``).
    periods_per_year:
        Annualization factor for a scalar ``safe`` rate.
    """

    def __init__(
        self,
        *,
        max_drawdown: float | None = None,
        leverage_cap: float = 1.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if max_drawdown is not None and (
            not np.isfinite(max_drawdown) or not (0.0 < max_drawdown <= 1.0)
        ):
            raise ValidationError("max_drawdown must be finite and in (0, 1] when set.")
        if not np.isfinite(leverage_cap) or leverage_cap < 0:
            raise ValidationError("leverage_cap must be finite and non-negative.")
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValidationError("periods_per_year must be finite and positive.")
        self.max_drawdown = max_drawdown
        self.leverage_cap = float(leverage_cap)
        self.periods_per_year = periods_per_year

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Run the drawdown-floored CPPI; see class docstring for the floor rule."""
        risky_returns, index, asset = _single_risky_returns(risky)
        max_dd = self.max_drawdown if self.max_drawdown is not None else params.max_drawdown
        if max_dd is None:
            raise BacktestError(
                "DrawdownAllocator requires max_drawdown via the factory or AllocatorParams."
            )
        try:
            safe_returns = _resolve_safe_series(
                safe,
                n_steps=risky_returns.size,
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
                max_drawdown=max_dd,
                leverage_cap=self.leverage_cap,
            )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"Drawdown allocation failed: {exc}") from exc

        result = _history_to_result(
            history,
            index=index,
            risky_asset=asset,
            params=params,
            periods_per_year=self.periods_per_year,
            leverage_cap=self.leverage_cap,
            strategy="drawdown",
        )
        result.metadata["max_drawdown"] = float(max_dd)
        return result


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------


def bt_mix(
    allocator: object,
    scenarios: pd.DataFrame,
    safe: ReturnMatrix | float,
    params: AllocatorParams,
    *,
    is_prices: bool = True,
    return_method: str = "simple",
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, object]:
    """Run an allocator over a *set* of scenario paths (e.g. GBM paths).

    Each column of ``scenarios`` is an independent path of the risky asset. The
    allocator is run once per column and the results are aggregated into a
    terminal-wealth distribution plus a mean equity curve.

    Parameters
    ----------
    allocator:
        Any object implementing the :class:`~riskbudget.core.interfaces.Allocator`
        protocol (``allocate(risky, safe, params) -> BacktestResult``).
    scenarios:
        A ``(n_steps[+1], n_scenarios)`` DataFrame. If ``is_prices`` (default) the
        columns are *price* paths (as from
        :func:`riskbudget.simulate.gbm.gbm`) and are converted to per-period simple
        returns; otherwise they are already per-period *simple* returns.
    safe:
        Safe-asset return series (single-column :class:`ReturnMatrix` aligned to the
        per-period return index) or a scalar annualized rate.
    params:
        :class:`AllocatorParams` driving the allocator.
    is_prices:
        Whether ``scenarios`` holds prices (convert to returns) or returns.
    return_method:
        ``"simple"`` (default) or ``"log"`` when converting prices to returns.
        Simple is the compounding convention (BUILD_PLAN §3.1).
    periods_per_year:
        Annualization factor stored in the summary.

    Returns
    -------
    dict
        ``{"terminal_wealth": np.ndarray, "results": list[BacktestResult],
        "mean_equity_curve": pd.Series, "summary": dict}``. ``summary`` reports
        mean/median/std/min/max terminal wealth, the floor-breach probability, and
        the expected shortfall below ``floor × start_value``.

    Raises
    ------
    BacktestError
        On an empty/degenerate scenario frame or an allocator failure.
    """
    if not isinstance(scenarios, pd.DataFrame):
        raise BacktestError("bt_mix expects a pandas DataFrame of scenario paths.")
    if scenarios.shape[0] == 0 or scenarios.shape[1] == 0:
        raise BacktestError("bt_mix received an empty scenario frame.")
    if not hasattr(allocator, "allocate"):
        raise BacktestError("bt_mix `allocator` must implement allocate(risky, safe, params).")

    if return_method not in ("simple", "log"):
        raise BacktestError("return_method must be 'simple' or 'log'.")

    frame = scenarios.copy()
    if is_prices:
        if frame.shape[0] < 2:
            raise BacktestError("Price scenarios need at least two rows to form returns.")
        if return_method == "log":
            ret_values = np.log(frame.to_numpy()[1:] / frame.to_numpy()[:-1])
        else:
            ret_values = frame.to_numpy()[1:] / frame.to_numpy()[:-1] - 1.0
        ret_index = frame.index[1:]
    else:
        ret_values = frame.to_numpy()
        ret_index = frame.index

    if not np.isfinite(ret_values).all():
        raise BacktestError("Scenario returns contain NaN or infinite values.")

    n_steps, n_scen = ret_values.shape
    results: list[BacktestResult] = []
    terminal = np.empty(n_scen, dtype=float)
    equity_matrix = np.empty((n_steps, n_scen), dtype=float)

    for j in range(n_scen):
        col = pd.DataFrame({"RISKY": ret_values[:, j]}, index=ret_index)
        risky = ReturnMatrix(col)
        try:
            res = allocator.allocate(risky, safe, params)
        except Exception as exc:
            raise BacktestError(f"bt_mix scenario {j} failed: {exc}") from exc
        results.append(res)
        equity_matrix[:, j] = res.equity_curve.to_numpy()
        terminal[j] = float(res.equity_curve.iloc[-1])

    mean_equity = pd.Series(equity_matrix.mean(axis=1), index=ret_index, name="mean_equity")

    floor_level = float(params.floor) * float(params.start_value)
    breach = terminal < floor_level - 1e-9
    shortfall = np.clip(floor_level - terminal, 0.0, None)
    summary: dict[str, float] = {
        "n_scenarios": float(n_scen),
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "std": float(np.std(terminal, ddof=1)) if n_scen > 1 else 0.0,
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
        "floor_level": floor_level,
        "p_breach_floor": float(np.mean(breach)),
        "e_shortfall": float(np.mean(shortfall)),
        "periods_per_year": float(periods_per_year),
    }

    return {
        "terminal_wealth": terminal,
        "results": results,
        "mean_equity_curve": mean_equity,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Factory callables (BUILD_PLAN 5.2 names)
# ---------------------------------------------------------------------------


def fixed_mix(
    *,
    risky_weight: float = 0.6,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> FixedMixAllocator:
    """Factory for the ``"fixed_mix"`` allocator (BUILD_PLAN §5.2)."""
    return FixedMixAllocator(
        risky_weight=risky_weight,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )


def glidepath(
    *,
    start_weight: float = 0.8,
    end_weight: float = 0.2,
    state_dependent: bool = False,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> GlidepathAllocator:
    """Factory for the ``"glidepath"`` allocator (BUILD_PLAN §5.2)."""
    return GlidepathAllocator(
        start_weight=start_weight,
        end_weight=end_weight,
        state_dependent=state_dependent,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )


def floor(
    *,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> FloorAllocator:
    """Factory for the ``"floor"`` allocator (BUILD_PLAN §5.2)."""
    return FloorAllocator(leverage_cap=leverage_cap, periods_per_year=periods_per_year)


def drawdown(
    *,
    max_drawdown: float | None = None,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> DrawdownAllocator:
    """Factory for the ``"drawdown"`` allocator (BUILD_PLAN §5.2)."""
    return DrawdownAllocator(
        max_drawdown=max_drawdown,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )
