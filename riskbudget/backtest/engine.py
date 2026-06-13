"""Walk-forward backtest engine for risk-budgeted (and benchmark) strategies.

:class:`WalkForwardBacktester` implements the
:class:`~riskbudget.core.interfaces.Backtester` protocol (BUILD_PLAN §5). It steps
through a :class:`~riskbudget.core.interfaces.RebalanceSchedule`; at each rebalance
it estimates the covariance from a **trailing lookback window only** (strictly past
data — no look-ahead), solves portfolio weights, holds them until the next
rebalance while the book drifts with realized returns, and accrues net-of-cost
returns into an equity curve.

Design notes (BUILD_PLAN §3.1, §7; Agent 5 brief)
-------------------------------------------------
- **Composes anything.** The engine drives any
  :class:`~riskbudget.core.interfaces.RiskModel` and any
  :class:`~riskbudget.core.interfaces.PortfolioConstructor` /
  :class:`~riskbudget.core.interfaces.Optimizer` passed in — ERC, GMV, MSR,
  equal-weight, etc. — through one interface so strategies are comparable
  head-to-head on the same data/schedule. It never hard-codes an estimator or a
  solver.
- **No look-ahead.** The covariance (and, for MSR/EF, the expected-returns vector)
  used to solve the weights *held over* a period are estimated solely from returns
  **strictly before** the rebalance date. A dedicated test asserts this.
- **MSR / efficient-frontier** runs additionally take a
  :class:`~riskbudget.core.interfaces.MeanModel` (supplied at construction) for the
  expected-returns input the classical tangency path needs.
- **Costs.** Turnover-based proportional transaction costs
  (:mod:`riskbudget.backtest.costs`) are charged at each rebalance and reduce net
  returns. Turnover is measured against the *drifted* pre-rebalance book.
- **Determinism.** Given identical inputs (and the models' own seeds), the equity
  curve is identical run to run; the engine introduces no RNG of its own.

Return convention
-----------------
Simple (arithmetic) returns are used for compounding the equity curve (BUILD_PLAN
§3.1): the per-period portfolio return is ``wᵀ r`` and growth-of-$1 compounds as
``∏(1 + r_p)``. Prices are converted with ``method="simple"`` by default.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from riskbudget.backtest.costs import CostModel, ProportionalCost, compute_turnover
from riskbudget.core.errors import BacktestError, RiskBudgetError
from riskbudget.core.interfaces import (
    Constraints,
    MeanModel,
    RebalanceFrequency,
    RebalanceSchedule,
    RiskModel,
)
from riskbudget.core.types import (
    BacktestResult,
    ExpectedReturns,
    Portfolio,
    PriceData,
    ReturnMatrix,
    RiskBudget,
)

# Pandas resample offset alias per rebalance frequency (label on the period end).
_FREQ_ALIAS: dict[RebalanceFrequency, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "annual": "YE",
}


@runtime_checkable
class _Constructor(Protocol):
    """Structural type for a :class:`~riskbudget.core.interfaces.PortfolioConstructor`."""

    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = ...,
        budget: RiskBudget | None = ...,
        constraints: Constraints,
    ) -> Portfolio: ...


@runtime_checkable
class _Solver(Protocol):
    """Structural type for the risk-budget :class:`~riskbudget.core.interfaces.Optimizer`."""

    def solve(
        self,
        cov: np.ndarray,
        budget: RiskBudget,
        constraints: Constraints,
    ) -> Portfolio: ...


class WalkForwardBacktester:
    """Walk-forward backtester (BUILD_PLAN §5, §7; Agent 5).

    Parameters
    ----------
    constraints:
        Feasibility limits handed to the constructor/optimizer at every solve.
        Defaults to long-only, fully invested (leverage 1.0).
    cost_model:
        Transaction-cost model charged on rebalance turnover. Defaults to a
        costless :class:`~riskbudget.backtest.costs.ProportionalCost` (0 bps).
    mean_model:
        Optional :class:`~riskbudget.core.interfaces.MeanModel`. Required only for
        constructors that need expected returns (MSR / efficient-frontier): when
        supplied, ``μ`` is re-estimated from the same trailing window as the
        covariance and passed to ``construct(..., mu=...)``.
    return_method:
        ``"simple"`` (default) or ``"log"`` — how prices are converted to the
        return series that both *drives* compounding and *feeds* the risk model.
        Simple returns are the compounding convention (BUILD_PLAN §3.1).

    Notes
    -----
    The protocol :meth:`run` signature is frozen (BUILD_PLAN §5); per-run policy
    that the signature does not carry (constraints, costs, the mean model) is
    configured here at construction so the engine stays a drop-in ``Backtester``.
    """

    def __init__(
        self,
        *,
        constraints: Constraints | None = None,
        cost_model: CostModel | None = None,
        mean_model: MeanModel | None = None,
        return_method: str = "simple",
    ) -> None:
        if return_method not in ("simple", "log"):
            raise BacktestError(f"Unknown return_method {return_method!r}; use 'simple' or 'log'.")
        self.constraints = (
            constraints if constraints is not None else Constraints.long_only_fully_invested()
        )
        self.cost_model: CostModel = cost_model if cost_model is not None else ProportionalCost()
        self.mean_model = mean_model
        self.return_method = return_method

    # -- public API ---------------------------------------------------------

    def run(
        self,
        data: PriceData,
        model: RiskModel,
        optimizer: _Solver | _Constructor,
        budget: RiskBudget,
        schedule: RebalanceSchedule,
    ) -> BacktestResult:
        """Walk ``data`` forward, re-solving on ``schedule``; return a result.

        ``model`` estimates the covariance from each trailing lookback window;
        ``optimizer`` solves the weights for that window (any ``Optimizer`` *or*
        ``PortfolioConstructor`` — ERC, GMV, MSR, equal-weight, …). The weights
        solved at a rebalance are held until the next, drifting with realized
        returns, and proportional costs are charged on rebalance turnover.

        Raises
        ------
        BacktestError
            If there is not enough history for a single rebalance, the schedule
            yields no rebalance dates, or any underlying model/solver fails.
        """
        if not isinstance(data, PriceData):
            raise BacktestError("WalkForwardBacktester.run expects a PriceData panel.")

        returns = self._prices_to_returns(data)
        assets = returns.assets
        ret_values = returns.values  # (T, N) simple/log returns
        dates = returns.dates
        n_periods = ret_values.shape[0]

        rebalance_idx = self._rebalance_indices(returns, schedule)

        port_returns = np.zeros(n_periods, dtype=float)
        weights_records: dict[pd.Timestamp, np.ndarray] = {}
        diagnostics: list[dict[str, object]] = []

        # The *drifted* book between rebalances (target weights at each rebalance,
        # then drifted by realized returns until the next).
        held_weights = np.zeros(len(assets), dtype=float)

        next_rebalances = set(rebalance_idx)

        for t in range(n_periods):
            if t in next_rebalances:
                solved, diag = self._solve_at(
                    t=t,
                    returns=returns,
                    ret_values=ret_values,
                    assets=assets,
                    model=model,
                    optimizer=optimizer,
                    budget=budget,
                    schedule=schedule,
                    drifted=held_weights,
                )
                turnover = compute_turnover(held_weights, solved, one_way=self._cost_is_one_way())
                cost = float(self.cost_model.cost(held_weights, solved))
                held_weights = solved.copy()
                weights_records[dates[t]] = solved.copy()
                diag["turnover"] = turnover
                diag["cost"] = cost
                diagnostics.append(diag)
            else:
                cost = 0.0

            r_t = ret_values[t]
            gross = float(held_weights @ r_t)
            net = gross - cost
            port_returns[t] = net

            # Drift the book by this period's realized returns (simple-return drift).
            held_weights = self._drift(held_weights, r_t)

        equity_curve = self._build_equity_curve(port_returns, dates)
        weights_frame = self._build_weights_frame(weights_records, assets)
        returns_series = pd.Series(port_returns, index=dates, name="return")

        result = BacktestResult(
            equity_curve=equity_curve,
            weights=weights_frame,
            returns=returns_series,
            diagnostics={
                "per_rebalance": diagnostics,
                "n_rebalances": len(diagnostics),
                "total_cost": float(sum(d["cost"] for d in diagnostics)),  # type: ignore[misc]
            },
            metadata={
                "engine": "walkforward",
                "return_method": self.return_method,
                "frequency": schedule.frequency,
                "lookback": schedule.lookback,
                "n_assets": len(assets),
                "n_periods": int(n_periods),
                "cost_bps": getattr(self.cost_model, "bps", None),
            },
        )
        return result

    # -- internals ----------------------------------------------------------

    def _prices_to_returns(self, data: PriceData) -> ReturnMatrix:
        try:
            return data.to_returns(method=self.return_method)  # type: ignore[arg-type]
        except RiskBudgetError as exc:
            raise BacktestError(f"Could not compute returns from prices: {exc}") from exc

    def _cost_is_one_way(self) -> bool:
        return bool(getattr(self.cost_model, "one_way", True))

    def _rebalance_indices(self, returns: ReturnMatrix, schedule: RebalanceSchedule) -> list[int]:
        """Resolve integer row indices into ``returns`` where a rebalance fires.

        A rebalance at row ``t`` solves weights from data strictly before ``t``
        (rows ``[t-lookback, t)``), so the first feasible index is the first row
        with at least ``effective_min_lookback`` periods of history behind it. The
        schedule's optional ``start``/``end`` bounds clip the window.
        """
        dates = returns.dates
        n = len(dates)
        min_lb = schedule.effective_min_lookback
        if n <= min_lb:
            raise BacktestError(
                f"Not enough history: {n} return periods, need > {min_lb} for one rebalance."
            )

        # Apply optional [start, end] window on the rebalance dates.
        ts_index = pd.DatetimeIndex(dates)
        in_window = np.ones(n, dtype=bool)
        if schedule.start is not None:
            in_window &= ts_index >= pd.Timestamp(schedule.start)
        if schedule.end is not None:
            in_window &= ts_index <= pd.Timestamp(schedule.end)

        if schedule.frequency == "none":
            candidate_rows = [min_lb]
        elif schedule.frequency == "daily":
            candidate_rows = list(range(min_lb, n))
        else:
            candidate_rows = self._period_boundary_rows(ts_index, schedule.frequency, min_lb)

        rows = [t for t in candidate_rows if min_lb <= t < n and in_window[t]]
        # De-dupe + sort defensively.
        rows = sorted(set(rows))
        if not rows:
            raise BacktestError(
                "Rebalance schedule produced no valid rebalance dates "
                f"(frequency={schedule.frequency!r}, min_lookback={min_lb}, "
                f"start={schedule.start}, end={schedule.end})."
            )
        return rows

    def _period_boundary_rows(
        self, ts_index: pd.DatetimeIndex, frequency: RebalanceFrequency, min_lb: int
    ) -> list[int]:
        """Rows that are the *first* trading row of each calendar period."""
        alias = _FREQ_ALIAS[frequency]
        # Group rows by their period bucket; the first row index in each bucket
        # is a rebalance row. ``to_period`` collapses each timestamp to its period.
        periods = ts_index.to_period(_period_code(alias))
        rows: list[int] = []
        seen: set[object] = set()
        for i, p in enumerate(periods):
            if p not in seen:
                seen.add(p)
                rows.append(i)
        return rows

    def _solve_at(
        self,
        *,
        t: int,
        returns: ReturnMatrix,
        ret_values: np.ndarray,
        assets: list[str],
        model: RiskModel,
        optimizer: _Solver | _Constructor,
        budget: RiskBudget,
        schedule: RebalanceSchedule,
        drifted: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Estimate the risk model on the trailing window and solve weights.

        The window is rows ``[lo, t)`` — **strictly before** ``t`` (no look-ahead).
        """
        lookback = schedule.lookback
        lo = 0 if lookback is None else max(0, t - lookback)
        window = ret_values[lo:t]
        if window.shape[0] < schedule.effective_min_lookback:
            raise BacktestError(
                f"Trailing window at row {t} has {window.shape[0]} periods, "
                f"need >= {schedule.effective_min_lookback}."
            )

        window_rm = returns.select(assets)  # ensure ordering
        window_frame = window_rm.frame.iloc[lo:t]
        window_matrix = ReturnMatrix(window_frame)

        try:
            cov = model.estimate(window_matrix)
        except RiskBudgetError as exc:
            raise BacktestError(f"Risk model failed at row {t}: {exc}") from exc
        cov = np.asarray(cov, dtype=float)

        mu = self._estimate_mu(window_matrix) if self.mean_model is not None else None

        self._sync_prev_weights(optimizer, assets, drifted)
        portfolio = self._construct(optimizer, cov, budget, mu)
        w = portfolio.as_array(assets)

        diag: dict[str, object] = {
            "date": returns.dates[t],
            "row": int(t),
            "window_start_row": int(lo),
            "window_end_row": int(t),  # exclusive — strictly past data
            "window_len": int(window.shape[0]),
            "leverage": float(np.abs(w).sum()),
            "net_exposure": float(w.sum()),
        }
        return w, diag

    @staticmethod
    def _sync_prev_weights(
        optimizer: _Solver | _Constructor,
        assets: list[str],
        drifted: np.ndarray,
    ) -> None:
        """Thread the *drifted* pre-rebalance book into turnover-aware optimizers.

        Optimizers that enforce ``Constraints.max_turnover`` (the convex
        risk-budget path) need the previous-period weights to constrain against.
        The seam is duck-typed and **optional**: any optimizer exposing a
        ``set_prev_weights(mapping | None)`` method receives the drifted book
        (keyed by asset id, in this run's asset order) before every solve;
        optimizers without the method are untouched.

        The **first** rebalance deploys from cash (``drifted`` is all zeros). A
        ``max_turnover`` small enough to be useful would make that initial
        deployment infeasible (it requires turnover ~= 1), so the first
        rebalance is deliberately treated as *unconstrained-by-turnover*:
        ``set_prev_weights(None)`` is passed, which also clears any state left
        over from a previous run of the same optimizer object.
        """
        setter = getattr(optimizer, "set_prev_weights", None)
        if not callable(setter):
            return
        if not np.any(drifted):
            # First rebalance (deployment from cash): exempt from max_turnover.
            setter(None)
            return
        setter({a: float(w) for a, w in zip(assets, drifted, strict=True)})

    def _estimate_mu(self, window: ReturnMatrix) -> ExpectedReturns:
        assert self.mean_model is not None
        try:
            return self.mean_model.estimate(window)
        except RiskBudgetError as exc:
            raise BacktestError(f"Mean model failed: {exc}") from exc

    def _construct(
        self,
        optimizer: _Solver | _Constructor,
        cov: np.ndarray,
        budget: RiskBudget,
        mu: ExpectedReturns | None,
    ) -> Portfolio:
        """Dispatch to ``construct(...)`` (PortfolioConstructor) or ``solve(...)``.

        Prefers the umbrella :class:`PortfolioConstructor` contract so any method
        (ERC/GMV/MSR/equal-weight) runs through one path; falls back to the
        risk-budget :class:`Optimizer.solve` when only that is available.
        """
        try:
            if hasattr(optimizer, "construct"):
                return optimizer.construct(cov, mu=mu, budget=budget, constraints=self.constraints)
            if hasattr(optimizer, "solve"):
                return optimizer.solve(cov, budget, self.constraints)
        except RiskBudgetError as exc:
            raise BacktestError(f"Portfolio construction failed: {exc}") from exc
        raise BacktestError(
            "optimizer must implement `construct(...)` (PortfolioConstructor) "
            "or `solve(...)` (Optimizer)."
        )

    @staticmethod
    def _drift(weights: np.ndarray, period_return: np.ndarray) -> np.ndarray:
        """Drift weights by one period of (simple) realized returns.

        After a period each asset position grows by ``(1 + r)`` while the cash
        residual ``1 - Σw`` is flat (rf = 0), so the portfolio grows by
        ``1 + w·r`` and the drifted weights are ``w(1+r) / (1 + w·r)``.

        Dividing by the *portfolio growth* — not the weight sum — preserves the
        book's gross exposure: a levered (``Σw > 1``) or partly-cash (``Σw < 1``)
        book stays levered / partly-cash between rebalances rather than being
        silently renormalized to fully-invested. For a fully-invested book
        (``Σw == 1``) this reduces exactly to ``grown / grown.sum()``.
        """
        grown = weights * (1.0 + period_return)
        portfolio_growth = 1.0 + float(weights @ period_return)
        if abs(portfolio_growth) < 1e-15:
            # the book (incl. cash) was wiped out; weights are no longer
            # meaningful — keep the previous ones rather than divide by ~0.
            return weights.copy()
        return np.asarray(grown / portfolio_growth, dtype=float)

    @staticmethod
    def _build_equity_curve(port_returns: np.ndarray, dates: pd.Index) -> pd.Series:
        equity = np.cumprod(1.0 + port_returns)
        return pd.Series(equity, index=dates, name="equity")

    @staticmethod
    def _build_weights_frame(
        records: dict[pd.Timestamp, np.ndarray], assets: list[str]
    ) -> pd.DataFrame:
        if not records:  # pragma: no cover - guarded earlier
            return pd.DataFrame(columns=assets)
        index = pd.DatetimeIndex(list(records.keys()))
        rows = np.vstack([records[k] for k in records])
        return pd.DataFrame(rows, index=index, columns=assets)


def _period_code(alias: str) -> str:
    """Map a resample offset alias to a ``to_period`` frequency code."""
    return {"D": "D", "W": "W", "ME": "M", "QE": "Q", "YE": "Y"}.get(alias, alias)


def walkforward(
    *,
    constraints: Constraints | None = None,
    cost_model: CostModel | None = None,
    mean_model: MeanModel | None = None,
    return_method: str = "simple",
) -> WalkForwardBacktester:
    """Registry-friendly factory for :class:`WalkForwardBacktester` (BUILD_PLAN §5.2).

    Exposed under the name ``"walkforward"`` so Agent 11's registry can resolve it.
    All arguments are keyword-only and optional; defaults give a long-only,
    fully-invested, costless walk-forward backtester using simple returns.
    """
    return WalkForwardBacktester(
        constraints=constraints,
        cost_model=cost_model,
        mean_model=mean_model,
        return_method=return_method,
    )


__all__ = ["WalkForwardBacktester", "walkforward"]
