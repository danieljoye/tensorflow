"""Three-fund PSP / liability-hedging / safe-asset allocator (BUILD_PLAN §2, §11).

Generalizes CPPI from a risky/safe split to a three-building-block split — a
**performance-seeking portfolio (PSP)**, a **liability-hedging portfolio (LHP)**,
and a **safe asset** — driven by the *funding ratio* and its distance above a
floor, following

    Martellini, L. & Milhau, V. (2012), "Dynamic Allocation Decisions in the
    Presence of Funding Ratio Constraints," *Journal of Pension Economics &
    Finance*, 11(4):549–580.

Mechanics (the funding-ratio analogue of CPPI). Let the funding ratio be
``F = assets / liabilities``. With a funding-ratio floor ``F_floor`` the cushion
is the *relative distance to the floor*

    cushion = (F − F_floor) / F = 1 − F_floor / F,

and the multiplier ``m`` sets the allocation to the performance-seeking portfolio

    w_psp = clip(m × cushion, 0, leverage_cap).

The remaining ``1 − w_psp`` is *liability-hedging*: when the funding ratio is
comfortable the strategy seeks performance (PSP); as the funding ratio approaches
its floor it de-risks into the LHP, whose returns track the liabilities so the
funding ratio stops falling — exactly the CPPI insurance logic transposed onto the
surplus. A safe asset is held inside the LHP sleeve when ``safe`` is supplied (a
convex blend ``lhp_safe_weight`` of safe and LHP returns), so the "hedging" sleeve
can carry the riskless building block of the three-fund theorem.

This is the *allocator* only. Full ALM (liability present-value from a term
structure, duration matching, surplus-risk optimization) is roadmap (BUILD_PLAN
§10); here liabilities are an exogenous return/level stream the caller supplies.

Return convention (BUILD_PLAN §3.1): inputs are *simple* per-period returns;
assets and liabilities compound multiplicatively. The allocator draws no
randomness — determinism is inherited from the realized return paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskbudget.core.errors import BacktestError, ValidationError
from riskbudget.core.types import AllocatorParams, BacktestResult, ReturnMatrix
from riskbudget.dynamic.cppi import DEFAULT_PERIODS_PER_YEAR, _resolve_safe_series

__all__ = [
    "FundSeparationAllocator",
    "FundSeparationHistory",
    "fund_separation",
    "run_fund_separation_path",
]


class FundSeparationHistory:
    """Per-step bookkeeping for a three-fund / funding-ratio run.

    Attributes
    ----------
    assets:
        Asset (account) value after each step — the wealth / equity curve.
    liabilities:
        Liability level after each step.
    funding_ratio:
        ``assets / liabilities`` after each step.
    floor:
        Funding-ratio floor in effect during each step.
    cushion:
        ``1 − floor / funding_ratio`` measured at the start of each step (clipped
        at 0), the quantity the PSP weight is computed from.
    psp_weight:
        ``clip(m × cushion, 0, leverage_cap)`` applied during each step.
    """

    __slots__ = (
        "assets",
        "cushion",
        "floor",
        "funding_ratio",
        "liabilities",
        "psp_weight",
    )

    def __init__(
        self,
        *,
        assets: np.ndarray,
        liabilities: np.ndarray,
        funding_ratio: np.ndarray,
        floor: np.ndarray,
        cushion: np.ndarray,
        psp_weight: np.ndarray,
    ) -> None:
        self.assets = assets
        self.liabilities = liabilities
        self.funding_ratio = funding_ratio
        self.floor = floor
        self.cushion = cushion
        self.psp_weight = psp_weight


def run_fund_separation_path(
    psp_returns: np.ndarray,
    lhp_returns: np.ndarray,
    liability_returns: np.ndarray,
    safe_returns: np.ndarray,
    *,
    multiplier: float,
    funding_floor: float,
    start_assets: float = 1.0,
    start_liabilities: float = 1.0,
    leverage_cap: float = 1.0,
    lhp_safe_weight: float = 0.0,
) -> FundSeparationHistory:
    """Run the three-fund funding-ratio strategy over one realized path.

    Parameters
    ----------
    psp_returns, lhp_returns, liability_returns, safe_returns:
        Length-``T`` arrays of *simple* per-period returns: the performance-seeking
        portfolio, the liability-hedging portfolio, the liability stream, and the
        safe asset. All four must share length ``T`` and be finite.
    multiplier:
        The multiplier ``m`` (``> 0``) on the funding-ratio cushion.
    funding_floor:
        Funding-ratio floor ``F_floor`` (``> 0``) the strategy protects; the
        cushion is ``1 − F_floor / F``.
    start_assets, start_liabilities:
        Initial asset and liability levels (``> 0``); the initial funding ratio is
        ``start_assets / start_liabilities`` and must exceed ``funding_floor``.
    leverage_cap:
        Upper bound on the PSP weight (``>= 0``); ``1.0`` forbids leverage.
    lhp_safe_weight:
        Fraction (``[0, 1]``) of the hedging sleeve held in the safe asset rather
        than the LHP (the riskless building block of the three-fund split). ``0``
        (default) makes the hedging sleeve pure LHP.

    Returns
    -------
    FundSeparationHistory
        Assets / liabilities / funding-ratio / floor / cushion / PSP-weight arrays.

    Raises
    ------
    ValidationError
        On mismatched lengths, non-finite inputs, out-of-range parameters, or an
        initial funding ratio at/below the floor.
    """
    psp = np.asarray(psp_returns, dtype=float)
    lhp = np.asarray(lhp_returns, dtype=float)
    liab = np.asarray(liability_returns, dtype=float)
    safe = np.asarray(safe_returns, dtype=float)
    arrays = {"psp": psp, "lhp": lhp, "liability": liab, "safe": safe}
    for name, arr in arrays.items():
        if arr.ndim != 1:
            raise ValidationError(f"{name}_returns must be a 1-D array.")
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1:
        raise ValidationError(
            "psp/lhp/liability/safe return arrays must share one length, "
            f"got {[arr.size for arr in arrays.values()]}."
        )
    if psp.size == 0:
        raise ValidationError("run_fund_separation_path needs at least one return period.")
    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise ValidationError(f"{name}_returns contains NaN or infinite values.")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValidationError("multiplier must be finite and positive.")
    if not np.isfinite(funding_floor) or funding_floor <= 0:
        raise ValidationError("funding_floor must be finite and positive.")
    if not np.isfinite(start_assets) or start_assets <= 0:
        raise ValidationError("start_assets must be finite and positive.")
    if not np.isfinite(start_liabilities) or start_liabilities <= 0:
        raise ValidationError("start_liabilities must be finite and positive.")
    if not np.isfinite(leverage_cap) or leverage_cap < 0:
        raise ValidationError("leverage_cap must be finite and non-negative.")
    if not np.isfinite(lhp_safe_weight) or not (0.0 <= lhp_safe_weight <= 1.0):
        raise ValidationError("lhp_safe_weight must be finite and in [0, 1].")
    if start_assets / start_liabilities <= funding_floor:
        raise ValidationError(
            "Initial funding ratio must exceed funding_floor "
            f"({start_assets / start_liabilities} <= {funding_floor})."
        )

    n = psp.size
    a_hist = np.empty(n, dtype=float)
    l_hist = np.empty(n, dtype=float)
    f_hist = np.empty(n, dtype=float)
    floor_hist = np.empty(n, dtype=float)
    cushion_hist = np.empty(n, dtype=float)
    weight_hist = np.empty(n, dtype=float)

    a_val = float(start_assets)
    l_val = float(start_liabilities)

    for t in range(n):
        funding = a_val / l_val if l_val > 0 else np.inf
        cushion = max(0.0, 1.0 - funding_floor / funding) if np.isfinite(funding) else 1.0
        w_psp = float(np.clip(multiplier * cushion, 0.0, leverage_cap))

        # Hedging sleeve = blend of LHP and safe; PSP sleeve = w_psp.
        hedge_return = (1.0 - lhp_safe_weight) * lhp[t] + lhp_safe_weight * safe[t]
        asset_return = w_psp * psp[t] + (1.0 - w_psp) * hedge_return
        a_val = a_val * (1.0 + asset_return)
        l_val = l_val * (1.0 + liab[t])

        a_hist[t] = a_val
        l_hist[t] = l_val
        f_hist[t] = a_val / l_val if l_val > 0 else np.inf
        floor_hist[t] = funding_floor
        cushion_hist[t] = cushion
        weight_hist[t] = w_psp

    return FundSeparationHistory(
        assets=a_hist,
        liabilities=l_hist,
        funding_ratio=f_hist,
        floor=floor_hist,
        cushion=cushion_hist,
        psp_weight=weight_hist,
    )


def _resolve_return_series(
    source: ReturnMatrix | float,
    *,
    n_steps: int,
    index: pd.Index,
    periods_per_year: int | float,
    name: str,
) -> np.ndarray:
    """Resolve a per-period return series from a ReturnMatrix or a scalar rate."""
    if isinstance(source, ReturnMatrix):
        frame = source.frame
        if frame.shape[1] != 1:
            raise ValidationError(f"{name} ReturnMatrix must have exactly one column.")
        if frame.shape[0] != n_steps:
            raise ValidationError(
                f"{name} series length {frame.shape[0]} does not match risky length {n_steps}."
            )
        return frame.to_numpy(dtype=float).reshape(-1)
    return _resolve_safe_series(
        float(source),
        n_steps=n_steps,
        index=index,
        safe_rate=float(source),
        periods_per_year=periods_per_year,
    )


class FundSeparationAllocator:
    """Three-fund PSP / LHP / safe allocator (Martellini–Milhau 2012).

    Implements the :class:`~riskbudget.core.interfaces.Allocator` protocol: the
    ``risky`` :class:`ReturnMatrix` carries the **performance-seeking portfolio**
    (single column) and ``safe`` the safe asset; the **liability-hedging
    portfolio** and the **liability** stream are supplied to the constructor (as a
    single-column :class:`ReturnMatrix` or a scalar annualized rate).

    The PSP weight each period is ``clip(m × cushion, 0, leverage_cap)`` with
    ``cushion = 1 − F_floor / F`` (distance of the funding ratio above its floor).
    See the module docstring for the full mechanics and citation.

    Parameters
    ----------
    lhp:
        Liability-hedging-portfolio returns (single-column :class:`ReturnMatrix`
        aligned to ``risky``) or a scalar annualized rate.
    liabilities:
        Liability returns (single-column :class:`ReturnMatrix` aligned to
        ``risky``) or a scalar annualized growth rate.
    funding_floor:
        Funding-ratio floor (``> 0``). The initial funding ratio
        ``start_assets / start_liabilities`` must exceed it.
    start_liabilities:
        Initial liability level (``> 0``). The initial asset level comes from
        ``params.start_value``.
    lhp_safe_weight:
        Fraction of the hedging sleeve held in the safe asset (``[0, 1]``).
    leverage_cap:
        Upper bound on the PSP weight (``>= 0``).
    periods_per_year:
        Annualization factor for scalar rate inputs.
    """

    def __init__(
        self,
        *,
        lhp: ReturnMatrix | float = 0.0,
        liabilities: ReturnMatrix | float = 0.0,
        funding_floor: float = 1.0,
        start_liabilities: float = 1.0,
        lhp_safe_weight: float = 0.0,
        leverage_cap: float = 1.0,
        periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
    ) -> None:
        if not np.isfinite(funding_floor) or funding_floor <= 0:
            raise ValidationError("funding_floor must be finite and positive.")
        if not np.isfinite(start_liabilities) or start_liabilities <= 0:
            raise ValidationError("start_liabilities must be finite and positive.")
        if not np.isfinite(lhp_safe_weight) or not (0.0 <= lhp_safe_weight <= 1.0):
            raise ValidationError("lhp_safe_weight must be finite and in [0, 1].")
        if not np.isfinite(leverage_cap) or leverage_cap < 0:
            raise ValidationError("leverage_cap must be finite and non-negative.")
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValidationError("periods_per_year must be finite and positive.")
        self.lhp = lhp
        self.liabilities = liabilities
        self.funding_floor = float(funding_floor)
        self.start_liabilities = float(start_liabilities)
        self.lhp_safe_weight = float(lhp_safe_weight)
        self.leverage_cap = float(leverage_cap)
        self.periods_per_year = periods_per_year

    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Run the three-fund funding-ratio strategy.

        ``risky`` is the single-column PSP return matrix; ``safe`` is the safe
        asset. The multiplier comes from ``params.multiplier``; the funding floor
        from the allocator's ``funding_floor``. Returns a :class:`BacktestResult`
        whose ``equity_curve`` is the asset (account) value, ``weights`` is the
        PSP / hedging split, and ``diagnostics`` carries the funding-ratio,
        liabilities, floor, and cushion series.

        Raises
        ------
        BacktestError
            If the run cannot be produced.
        """
        if not isinstance(risky, ReturnMatrix):
            raise BacktestError("FundSeparationAllocator.allocate expects a ReturnMatrix `risky`.")
        if risky.shape[1] != 1:
            raise BacktestError(
                "FundSeparationAllocator operates on a single PSP asset; "
                f"`risky` has {risky.shape[1]} columns."
            )

        try:
            psp_returns = risky.values.reshape(-1)
            index = risky.dates
            n_steps = psp_returns.size

            safe_returns = _resolve_safe_series(
                safe,
                n_steps=n_steps,
                index=index,
                safe_rate=params.safe_rate,
                periods_per_year=self.periods_per_year,
            )
            lhp_returns = _resolve_return_series(
                self.lhp,
                n_steps=n_steps,
                index=index,
                periods_per_year=self.periods_per_year,
                name="lhp",
            )
            liability_returns = _resolve_return_series(
                self.liabilities,
                n_steps=n_steps,
                index=index,
                periods_per_year=self.periods_per_year,
                name="liabilities",
            )
            history = run_fund_separation_path(
                psp_returns,
                lhp_returns,
                liability_returns,
                safe_returns,
                multiplier=params.multiplier,
                funding_floor=self.funding_floor,
                start_assets=params.start_value,
                start_liabilities=self.start_liabilities,
                leverage_cap=self.leverage_cap,
                lhp_safe_weight=self.lhp_safe_weight,
            )
        except (ValidationError, BacktestError) as exc:
            raise BacktestError(f"Fund-separation allocation failed: {exc}") from exc

        return self._to_result(history, index=index, psp_asset=risky.assets[0], params=params)

    def _to_result(
        self,
        history: FundSeparationHistory,
        *,
        index: pd.Index,
        psp_asset: str,
        params: AllocatorParams,
    ) -> BacktestResult:
        """Assemble a :class:`BacktestResult` from a fund-separation history."""
        equity = pd.Series(history.assets, index=index, name="assets")

        psp_col = str(psp_asset)
        hedge_col = "HEDGING"
        if hedge_col == psp_col:
            hedge_col = "HEDGING_SLEEVE"
        weights = pd.DataFrame(
            {
                psp_col: history.psp_weight,
                hedge_col: 1.0 - history.psp_weight,
            },
            index=index,
        )

        start_assets = float(params.start_value)
        prev = np.concatenate(([start_assets], history.assets[:-1]))
        returns = pd.Series(history.assets / prev - 1.0, index=index, name="returns")

        breached = bool(np.any(history.funding_ratio < self.funding_floor - 1e-9))
        metrics = {
            "terminal_wealth": float(history.assets[-1]),
            "terminal_funding_ratio": float(history.funding_ratio[-1]),
            "min_funding_ratio": float(np.min(history.funding_ratio)),
            "funding_floor_breached": float(breached),
            "max_psp_weight": float(np.max(history.psp_weight)),
            "mean_psp_weight": float(np.mean(history.psp_weight)),
        }

        diagnostics = {
            "funding_ratio": pd.Series(history.funding_ratio, index=index, name="funding_ratio"),
            "liabilities": pd.Series(history.liabilities, index=index, name="liabilities"),
            "floor": pd.Series(history.floor, index=index, name="floor"),
            "cushion": pd.Series(history.cushion, index=index, name="cushion"),
            "psp_weight": pd.Series(history.psp_weight, index=index, name="psp_weight"),
        }

        metadata = {
            "strategy": "fund_separation",
            "multiplier": float(params.multiplier),
            "funding_floor": self.funding_floor,
            "start_assets": start_assets,
            "start_liabilities": self.start_liabilities,
            "lhp_safe_weight": self.lhp_safe_weight,
            "leverage_cap": self.leverage_cap,
            "periods_per_year": float(self.periods_per_year),
            "psp_asset": psp_col,
            "hedging_sleeve": hedge_col,
        }

        return BacktestResult(
            equity_curve=equity,
            weights=weights,
            returns=returns,
            metrics=metrics,
            diagnostics=diagnostics,
            metadata=metadata,
        )


def fund_separation(
    *,
    lhp: ReturnMatrix | float = 0.0,
    liabilities: ReturnMatrix | float = 0.0,
    funding_floor: float = 1.0,
    start_liabilities: float = 1.0,
    lhp_safe_weight: float = 0.0,
    leverage_cap: float = 1.0,
    periods_per_year: int | float = DEFAULT_PERIODS_PER_YEAR,
) -> FundSeparationAllocator:
    """Factory for the ``"fund_separation"`` allocator (BUILD_PLAN §5.2)."""
    return FundSeparationAllocator(
        lhp=lhp,
        liabilities=liabilities,
        funding_floor=funding_floor,
        start_liabilities=start_liabilities,
        lhp_safe_weight=lhp_safe_weight,
        leverage_cap=leverage_cap,
        periods_per_year=periods_per_year,
    )
