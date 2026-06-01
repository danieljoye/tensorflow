"""Risk-contribution attribution through time (BUILD_PLAN §2, agent-6 §4).

Given a weights-through-time frame (one row per rebalance, one column per asset)
and a covariance matrix, this module reconstructs the additive risk decomposition
``TRCᵢ = wᵢ · (Σ w)ᵢ / σ(w)`` (with ``Σᵢ TRCᵢ = σ(w)``) at each point in time, plus

- **realized vs. target budget drift** — how far each rebalance's percentage risk
  contributions drift from a target :class:`~riskbudget.core.types.RiskBudget`, and
- **ENB and Diversification Ratio through time** — imported from Agent 9's
  :mod:`riskbudget.diversification.metrics` (not reimplemented here).

The risk math reuses Agent 4's
:mod:`riskbudget.budgeting.contributions` (``total_risk_contributions`` /
``portfolio_volatility``) so attribution always reconciles to the same numbers the
optimizer and the core :class:`~riskbudget.core.types.Portfolio` produce.

Covariance convention
---------------------
A single static covariance can be passed (applied at every rebalance), or a
mapping ``rebalance_date -> Σ`` for a time-varying risk model. Every ``Σ`` must be
ordered to match the *weights-frame column order*; the caller owns alignment, as
elsewhere in the system.

Source: Maillard–Roncalli–Teïletche (JPM 2010); Meucci (2009); Choueifaty–Coignard
(2008); BUILD_PLAN §2, §11, §12.5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from riskbudget.budgeting.contributions import (
    portfolio_volatility,
    total_risk_contributions,
)
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import RiskBudget
from riskbudget.diversification.metrics import (
    diversification_ratio,
    effective_number_of_bets,
)

# Reconciliation tolerance for the Euler identity Σᵢ TRCᵢ == σ(w).
_RECONCILE_ATOL = 1e-8


def _resolve_cov(
    cov: np.ndarray | Mapping[Any, np.ndarray],
    date: Any,
    n_assets: int,
) -> np.ndarray:
    """Select and validate the covariance for a given rebalance ``date``."""
    if isinstance(cov, Mapping):
        if date not in cov:
            raise ValidationError(f"No covariance supplied for rebalance date {date!r}.")
        sigma = np.asarray(cov[date], dtype=float)
    else:
        sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValidationError(f"Covariance must be square, got shape {sigma.shape}.")
    if sigma.shape[0] != n_assets:
        raise ValidationError(
            f"Covariance shape {sigma.shape} does not match {n_assets} weight columns."
        )
    if not np.isfinite(sigma).all():
        raise ValidationError("Covariance contains NaN or infinite values.")
    return sigma


def _validate_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Coerce/validate the weights-through-time frame (NaN treated as zero)."""
    if not isinstance(weights, pd.DataFrame):
        raise ValidationError("weights must be a pandas DataFrame.")
    if weights.shape[0] == 0 or weights.shape[1] == 0:
        raise ValidationError("weights frame cannot be empty.")
    w = weights.fillna(0.0)
    if not np.isfinite(w.to_numpy(dtype=float)).all():
        raise ValidationError("weights contains infinite values.")
    return w


def risk_contribution_history(
    weights: pd.DataFrame,
    cov: np.ndarray | Mapping[Any, np.ndarray],
    *,
    percentage: bool = False,
    verify: bool = True,
) -> pd.DataFrame:
    """Total (or percentage) risk contributions at each rebalance.

    Parameters
    ----------
    weights:
        DataFrame indexed by rebalance date, one column per asset (matching
        :attr:`riskbudget.core.types.BacktestResult.weights`). NaNs are treated
        as zero weight.
    cov:
        Either a single ``(N, N)`` covariance applied at every date, or a mapping
        ``rebalance_date -> Σ`` for a time-varying risk model. ``Σ`` columns must
        be ordered to match ``weights`` columns.
    percentage:
        If True, return percentage contributions ``PCRᵢ = TRCᵢ / σ(w)`` (each row
        sums to 1) instead of total contributions (each row sums to ``σ(w)``).
    verify:
        If True (default), assert the Euler identity ``Σᵢ TRCᵢ == σ(w)`` at every
        rebalance within ``1e-8`` and raise :class:`ValidationError` on violation.

    Returns
    -------
    pandas.DataFrame
        Same index/columns as ``weights``; values are TRC (or PCR) per asset.
    """
    w = _validate_weights(weights)
    assets = list(w.columns)
    rows: list[np.ndarray] = []
    for date, row in w.iterrows():
        vec = row.to_numpy(dtype=float)
        sigma = _resolve_cov(cov, date, len(assets))
        trc = total_risk_contributions(vec, sigma)
        vol = portfolio_volatility(vec, sigma)
        if verify and abs(float(trc.sum()) - vol) > _RECONCILE_ATOL:
            raise ValidationError(
                f"Risk contributions at {date!r} do not reconcile to total risk "
                f"({float(trc.sum())!r} vs σ={vol!r})."
            )
        if percentage:
            trc = trc / vol if vol > 0.0 else np.zeros_like(trc)
        rows.append(trc)
    return pd.DataFrame(rows, index=w.index, columns=assets)


def budget_drift(
    weights: pd.DataFrame,
    cov: np.ndarray | Mapping[Any, np.ndarray],
    budget: RiskBudget,
) -> pd.DataFrame:
    """Realized-minus-target percentage risk-contribution drift per rebalance.

    Computes percentage risk contributions ``PCRᵢ`` at each rebalance (which sum
    to 1) and subtracts the target budget ``bᵢ``. A row of all-zeros means the
    realized risk split exactly matched the target at that date; positive entries
    are assets carrying *more* risk than budgeted.

    Parameters
    ----------
    weights:
        Weights-through-time frame (see :func:`risk_contribution_history`).
    cov:
        Static covariance or ``date -> Σ`` mapping.
    budget:
        Target :class:`~riskbudget.core.types.RiskBudget`; must cover exactly the
        weights-frame columns.

    Returns
    -------
    pandas.DataFrame
        ``PCRᵢ − bᵢ`` per asset and rebalance; same index/columns as ``weights``.
    """
    w = _validate_weights(weights)
    assets = list(w.columns)
    if set(assets) != set(budget.assets):
        raise ValidationError(
            "RiskBudget assets must match the weights-frame columns exactly "
            f"(budget={sorted(budget.assets)}, weights={sorted(assets)})."
        )
    target = budget.as_array(assets)
    pcr = risk_contribution_history(w, cov, percentage=True, verify=False)
    return pcr.sub(pd.Series(target, index=assets), axis=1)


def diversification_history(
    weights: pd.DataFrame,
    cov: np.ndarray | Mapping[Any, np.ndarray],
    *,
    method: str = "minimum-torsion",
) -> pd.DataFrame:
    """Effective Number of Bets and Diversification Ratio through time.

    Imports both metrics from Agent 9's
    :mod:`riskbudget.diversification.metrics` (BUILD_PLAN §12.5) and evaluates
    them at each rebalance. All-cash / zero-variance rebalances yield ``nan`` for
    both metrics rather than raising.

    Parameters
    ----------
    weights:
        Weights-through-time frame.
    cov:
        Static covariance or ``date -> Σ`` mapping.
    method:
        Torsion basis for ENB (``"minimum-torsion"`` default, ``"pca"`` or
        ``"approximate"``); passed straight to
        :func:`riskbudget.diversification.metrics.effective_number_of_bets`.

    Returns
    -------
    pandas.DataFrame
        Columns ``["enb", "diversification_ratio"]`` indexed by rebalance date.
    """
    w = _validate_weights(weights)
    n_assets = w.shape[1]
    enb_vals: list[float] = []
    dr_vals: list[float] = []
    for date, row in w.iterrows():
        vec = row.to_numpy(dtype=float)
        sigma = _resolve_cov(cov, date, n_assets)
        var = float(vec @ sigma @ vec)
        if var <= 0.0:
            enb_vals.append(float("nan"))
            dr_vals.append(float("nan"))
            continue
        enb_vals.append(effective_number_of_bets(vec, sigma, method=method))  # type: ignore[arg-type]
        dr_vals.append(diversification_ratio(vec, sigma))
    return pd.DataFrame(
        {"enb": enb_vals, "diversification_ratio": dr_vals},
        index=w.index,
    )


__all__ = [
    "budget_drift",
    "diversification_history",
    "risk_contribution_history",
]
