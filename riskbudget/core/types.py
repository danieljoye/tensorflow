"""Shared, validated data types for the risk-budgeting system.

These are the *frozen contract* (BUILD_PLAN §5) that every downstream agent codes
against. The math-bearing methods are real implementations, not stubs:

- :meth:`PriceData.to_returns` — log or simple returns from a price panel.
- :meth:`RiskBudget.equal` — the equal-risk-contribution (ERC) budget.
- :meth:`Portfolio.risk_contributions` — total risk contributions ``TRCᵢ`` such
  that ``Σᵢ TRCᵢ == σ(w)`` (portfolio volatility).

Conventions
-----------
- A "return matrix" is a ``pandas.DataFrame`` whose index is dates and whose
  columns are asset ids. ``ReturnMatrix`` wraps it with validation + accessors.
- Covariance matrices are plain ``numpy.ndarray`` (shape ``(N, N)``), ordered to
  match a given list of asset ids — the caller is responsible for alignment, and
  helpers here make that explicit.
- Weights and budgets are keyed by asset id (``dict[str, float]``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from riskbudget.core.errors import ValidationError

ReturnMethod = Literal["log", "simple"]

# Tolerance for "sums to one" / symmetry style checks.
_SUM_TOL = 1e-8
_SYMMETRY_RTOL = 1e-8
_SYMMETRY_ATOL = 1e-10


def _as_float_array(values: Any, *, name: str) -> np.ndarray:
    """Coerce ``values`` to a float ndarray, raising ValidationError on failure."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValidationError(f"{name} could not be interpreted as floats: {exc}") from exc
    return arr


# ---------------------------------------------------------------------------
# Return / price panels
# ---------------------------------------------------------------------------


class ReturnMatrix:
    """A validated panel of asset returns (index=dates, columns=asset ids).

    Wraps a :class:`pandas.DataFrame`. The frame is copied on construction and
    sorted by date so downstream code can rely on a stable, ascending index.

    Parameters
    ----------
    frame:
        DataFrame with a date-like index and one column per asset.

    Raises
    ------
    ValidationError
        If the frame is empty, has duplicate dates/assets, non-numeric values, or
        contains NaN/inf.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("ReturnMatrix expects a pandas DataFrame.")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValidationError("ReturnMatrix cannot be empty.")
        if frame.columns.has_duplicates:
            raise ValidationError("ReturnMatrix has duplicate asset columns.")
        if frame.index.has_duplicates:
            raise ValidationError("ReturnMatrix has duplicate dates in its index.")

        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        try:
            frame = frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"ReturnMatrix values must be numeric: {exc}") from exc

        values = frame.to_numpy()
        if not np.isfinite(values).all():
            raise ValidationError("ReturnMatrix contains NaN or infinite values.")

        self._frame = frame.sort_index()

    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame (a defensive copy)."""
        return self._frame.copy()

    @property
    def values(self) -> np.ndarray:
        """Returns as a float ndarray of shape ``(n_dates, n_assets)``."""
        return self._frame.to_numpy()

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset ids (column order)."""
        return list(self._frame.columns)

    @property
    def dates(self) -> pd.DatetimeIndex | pd.Index:
        """The (ascending) date index."""
        return self._frame.index

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_dates, n_assets)``."""
        return (self._frame.shape[0], self._frame.shape[1])

    def __len__(self) -> int:
        return int(self._frame.shape[0])

    def select(self, assets: Sequence[str]) -> ReturnMatrix:
        """Return a new ReturnMatrix restricted to ``assets`` (in that order)."""
        missing = [a for a in assets if a not in self._frame.columns]
        if missing:
            raise ValidationError(f"Unknown assets requested: {missing}")
        return ReturnMatrix(self._frame.loc[:, list(assets)])

    def __repr__(self) -> str:
        n_dates, n_assets = self.shape
        return f"ReturnMatrix(n_dates={n_dates}, n_assets={n_assets}, assets={self.assets!r})"


class PriceData:
    """A validated panel of asset prices (index=dates, columns=asset ids).

    Use :meth:`to_returns` to convert to a :class:`ReturnMatrix`.

    Raises
    ------
    ValidationError
        If the frame is empty, has duplicate dates/assets, non-numeric values,
        contains NaN/inf, or contains non-positive prices.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("PriceData expects a pandas DataFrame.")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValidationError("PriceData cannot be empty.")
        if frame.columns.has_duplicates:
            raise ValidationError("PriceData has duplicate asset columns.")
        if frame.index.has_duplicates:
            raise ValidationError("PriceData has duplicate dates in its index.")

        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        try:
            frame = frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"PriceData values must be numeric: {exc}") from exc

        values = frame.to_numpy()
        if not np.isfinite(values).all():
            raise ValidationError("PriceData contains NaN or infinite values.")
        if (values <= 0).any():
            raise ValidationError("PriceData contains non-positive prices.")

        self._frame = frame.sort_index()

    @property
    def frame(self) -> pd.DataFrame:
        """The underlying DataFrame (a defensive copy)."""
        return self._frame.copy()

    @property
    def values(self) -> np.ndarray:
        """Prices as a float ndarray of shape ``(n_dates, n_assets)``."""
        return self._frame.to_numpy()

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset ids (column order)."""
        return list(self._frame.columns)

    @property
    def dates(self) -> pd.DatetimeIndex | pd.Index:
        """The (ascending) date index."""
        return self._frame.index

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_dates, n_assets)``."""
        return (self._frame.shape[0], self._frame.shape[1])

    def __len__(self) -> int:
        return int(self._frame.shape[0])

    def to_returns(self, method: ReturnMethod = "log") -> ReturnMatrix:
        """Convert prices to periodic returns.

        Parameters
        ----------
        method:
            ``"log"`` (default) for ``ln(P_t / P_{t-1})`` or ``"simple"`` for
            ``P_t / P_{t-1} - 1``.

        Returns
        -------
        ReturnMatrix
            One fewer row than the price panel (the first period is dropped).

        Raises
        ------
        ValidationError
            If ``method`` is unknown or fewer than two periods are present.
        """
        if method not in ("log", "simple"):
            raise ValidationError(f"Unknown return method: {method!r}. Use 'log' or 'simple'.")
        if self._frame.shape[0] < 2:
            raise ValidationError("Need at least two price rows to compute returns.")

        rets: pd.DataFrame
        if method == "log":
            ratio = self._frame / self._frame.shift(1)
            rets = pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)
        else:
            rets = self._frame.pct_change()

        return ReturnMatrix(rets.iloc[1:])

    def __repr__(self) -> str:
        n_dates, n_assets = self.shape
        return f"PriceData(n_dates={n_dates}, n_assets={n_assets}, assets={self.assets!r})"


# ---------------------------------------------------------------------------
# Risk budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBudget:
    """Target fraction of total portfolio risk allocated to each asset.

    The budgets are strictly positive floats that sum to one. The log-barrier
    optimizer requires strict positivity, so a zero budget is rejected. Passing
    values that do not already sum to one is rejected by the constructor (to
    catch malformed inputs); use :meth:`from_weights` to normalize an arbitrary
    vector of positive relative weights.

    Attributes
    ----------
    budgets:
        Mapping ``asset_id -> target risk fraction``. Insertion order defines the
        canonical asset ordering for this budget.
    """

    budgets: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValidationError("RiskBudget cannot be empty.")
        values = list(self.budgets.values())
        if any((not np.isfinite(v)) or v <= 0 for v in values):
            raise ValidationError("RiskBudget values must be finite and strictly positive.")
        total = float(sum(values))
        if abs(total - 1.0) > _SUM_TOL:
            raise ValidationError(f"RiskBudget values must sum to 1.0 (got {total!r}).")
        # Normalize away tiny float drift and freeze as a plain dict.
        normalized = {str(k): float(v) / total for k, v in self.budgets.items()}
        object.__setattr__(self, "budgets", normalized)

    @classmethod
    def equal(cls, assets: Iterable[str]) -> RiskBudget:
        """Construct an equal-risk-contribution (ERC) budget: ``bᵢ = 1/N``.

        Raises
        ------
        ValidationError
            If ``assets`` is empty or contains duplicates.
        """
        asset_list = [str(a) for a in assets]
        if not asset_list:
            raise ValidationError("Cannot build an equal RiskBudget over zero assets.")
        if len(set(asset_list)) != len(asset_list):
            raise ValidationError("Duplicate asset ids in equal RiskBudget.")
        n = len(asset_list)
        return cls(dict.fromkeys(asset_list, 1.0 / n))

    @classmethod
    def from_weights(cls, weights: Mapping[str, float]) -> RiskBudget:
        """Build a budget by normalizing arbitrary strictly-positive weights.

        Unlike the constructor (which requires the values to already sum to one),
        this rescales ``weights`` to sum to one. Useful when a caller thinks in
        relative terms, e.g. ``from_weights({"stocks": 3, "bonds": 1})``.

        Raises
        ------
        ValidationError
            If ``weights`` is empty or contains a non-finite/non-positive value.
        """
        if not weights:
            raise ValidationError("Cannot build a RiskBudget from empty weights.")
        values = list(weights.values())
        if any((not np.isfinite(v)) or v <= 0 for v in values):
            raise ValidationError("RiskBudget weights must be finite and strictly positive.")
        total = float(sum(values))
        return cls({str(k): float(v) / total for k, v in weights.items()})

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset ids."""
        return list(self.budgets.keys())

    def as_array(self, assets: Sequence[str] | None = None) -> np.ndarray:
        """Return budgets as a float vector ordered by ``assets``.

        Parameters
        ----------
        assets:
            Desired ordering. Defaults to this budget's own asset order. Must be a
            permutation of (subset is not allowed) the budget's assets.
        """
        order = list(self.budgets.keys()) if assets is None else [str(a) for a in assets]
        missing = [a for a in order if a not in self.budgets]
        if missing:
            raise ValidationError(f"Budget has no entry for assets: {missing}")
        if len(order) != len(self.budgets):
            raise ValidationError(
                "Requested ordering must cover exactly the budget's assets "
                f"({len(self.budgets)}), got {len(order)}."
            )
        return np.array([self.budgets[a] for a in order], dtype=float)

    def __len__(self) -> int:
        return len(self.budgets)


# ---------------------------------------------------------------------------
# Expected returns (μ)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedReturns:
    """A vector of expected (annualized) returns ``μ``, keyed by asset id.

    The classical optimizers that the core risk-budget path does not need —
    max-Sharpe / tangency (MSR), the efficient frontier, and the Efficient-MSR
    benchmark (BUILD_PLAN §5.1) — require an expected-returns input. This type
    is the additive contract carrying that ``μ`` vector, aligned to an explicit
    asset ordering exactly like :class:`RiskBudget` and :class:`Portfolio`.

    Values are signed annualized returns: unlike a :class:`RiskBudget` they may
    be negative or zero (an asset can be expected to lose money), and they do
    *not* sum to one. The only requirement is that every value is finite.

    Attributes
    ----------
    mu:
        Mapping ``asset_id -> expected annualized return``. Insertion order
        defines the canonical asset ordering for this vector.
    """

    mu: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.mu:
            raise ValidationError("ExpectedReturns cannot be empty.")
        if any(not np.isfinite(v) for v in self.mu.values()):
            raise ValidationError("ExpectedReturns values must be finite.")
        frozen = {str(k): float(v) for k, v in self.mu.items()}
        if len(frozen) != len(self.mu):
            raise ValidationError("ExpectedReturns has duplicate asset ids.")
        object.__setattr__(self, "mu", frozen)

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset ids."""
        return list(self.mu.keys())

    def as_array(self, assets: Sequence[str] | None = None) -> np.ndarray:
        """Return expected returns as a float vector ordered by ``assets``.

        Parameters
        ----------
        assets:
            Desired ordering. Defaults to this vector's own asset order. Must
            cover exactly this vector's assets (no subset, no extras) so a
            covariance matrix aligned to the same ``assets`` lines up element by
            element.
        """
        order = list(self.mu.keys()) if assets is None else [str(a) for a in assets]
        missing = [a for a in order if a not in self.mu]
        if missing:
            raise ValidationError(f"ExpectedReturns has no entry for assets: {missing}")
        if len(order) != len(self.mu):
            raise ValidationError(
                "Requested ordering must cover exactly the expected-returns assets "
                f"({len(self.mu)}), got {len(order)}."
            )
        return np.array([self.mu[a] for a in order], dtype=float)

    def __len__(self) -> int:
        return len(self.mu)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Portfolio:
    """A set of asset weights, keyed by asset id.

    Weights may be negative (short) unless a downstream constraint forbids it.
    ``leverage`` is the sum of absolute weights (gross exposure).

    Attributes
    ----------
    weights:
        Mapping ``asset_id -> weight``. Insertion order is the canonical ordering.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValidationError("Portfolio cannot be empty.")
        if any(not np.isfinite(v) for v in self.weights.values()):
            raise ValidationError("Portfolio weights must be finite.")
        frozen = {str(k): float(v) for k, v in self.weights.items()}
        object.__setattr__(self, "weights", frozen)

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset ids."""
        return list(self.weights.keys())

    @property
    def leverage(self) -> float:
        """Gross exposure: sum of absolute weights."""
        return float(sum(abs(w) for w in self.weights.values()))

    @property
    def net_exposure(self) -> float:
        """Net exposure: sum of (signed) weights."""
        return float(sum(self.weights.values()))

    def as_array(self, assets: Sequence[str] | None = None) -> np.ndarray:
        """Return weights as a float vector ordered by ``assets``.

        Parameters
        ----------
        assets:
            Desired ordering. Defaults to this portfolio's own asset order. Must
            cover exactly the portfolio's assets.
        """
        order = list(self.weights.keys()) if assets is None else [str(a) for a in assets]
        missing = [a for a in order if a not in self.weights]
        if missing:
            raise ValidationError(f"Portfolio has no weight for assets: {missing}")
        if len(order) != len(self.weights):
            raise ValidationError(
                "Requested ordering must cover exactly the portfolio's assets "
                f"({len(self.weights)}), got {len(order)}."
            )
        return np.array([self.weights[a] for a in order], dtype=float)

    def volatility(self, cov: np.ndarray, assets: Sequence[str] | None = None) -> float:
        """Portfolio volatility ``σ(w) = sqrt(wᵀ Σ w)``.

        ``cov`` must be ordered consistently with ``assets`` (or with the
        portfolio's own asset order when ``assets`` is None).
        """
        w = self.as_array(assets)
        sigma = _validate_cov(cov, n=w.size)
        var = float(w @ sigma @ w)
        if var < 0:
            # Can only happen with a non-PSD covariance; surface it clearly.
            raise ValidationError("Negative portfolio variance; covariance is not PSD.")
        return float(np.sqrt(var))

    def risk_contributions(
        self, cov: np.ndarray, assets: Sequence[str] | None = None
    ) -> dict[str, float]:
        """Total risk contributions ``TRCᵢ = wᵢ · (Σ w)ᵢ / σ(w)``.

        By construction ``Σᵢ TRCᵢ == σ(w)`` (the portfolio volatility). When the
        portfolio is all-cash (``σ(w) == 0``) every contribution is ``0``.

        Parameters
        ----------
        cov:
            Covariance matrix, ordered to match ``assets`` (or the portfolio's own
            order when ``assets`` is None).
        assets:
            Optional explicit ordering for both the weights and ``cov``.

        Returns
        -------
        dict[str, float]
            ``asset_id -> total risk contribution`` in the resolved asset order.
        """
        order = list(self.weights.keys()) if assets is None else [str(a) for a in assets]
        w = self.as_array(order)
        sigma = _validate_cov(cov, n=w.size)

        var = float(w @ sigma @ w)
        if var < 0:
            raise ValidationError("Negative portfolio variance; covariance is not PSD.")
        vol = float(np.sqrt(var))
        if vol == 0.0:
            return dict.fromkeys(order, 0.0)

        marginal = sigma @ w  # (Σ w)
        trc = w * marginal / vol  # TRCᵢ; sums to σ(w)
        return {asset: float(c) for asset, c in zip(order, trc, strict=True)}

    def __len__(self) -> int:
        return len(self.weights)


def _validate_cov(cov: np.ndarray, *, n: int) -> np.ndarray:
    """Validate a covariance matrix and return it as a float ndarray.

    Checks: square, shape ``(n, n)``, finite, symmetric. Symmetry is enforced
    within tolerance; PSD-ness is *not* enforced here (estimators own that) but
    negative-variance results downstream are reported as ValidationError.
    """
    sigma = _as_float_array(cov, name="covariance")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValidationError(f"Covariance must be a square matrix, got shape {sigma.shape}.")
    if sigma.shape[0] != n:
        raise ValidationError(
            f"Covariance shape {sigma.shape} does not match number of assets ({n})."
        )
    if not np.isfinite(sigma).all():
        raise ValidationError("Covariance contains NaN or infinite values.")
    if not np.allclose(sigma, sigma.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
        raise ValidationError("Covariance matrix is not symmetric.")
    return sigma


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest.

    This is a data container; the backtest engine (Agent 5) and analytics layer
    (Agent 6) own the logic that fills and consumes it.

    Attributes
    ----------
    equity_curve:
        Series indexed by date of cumulative portfolio value (or growth-of-$1).
    weights:
        DataFrame indexed by rebalance date, one column per asset, of the weights
        held *from* that date until the next rebalance.
    returns:
        Optional per-period portfolio return series (net of costs if applied).
    metrics:
        Scalar performance/risk metrics (Sharpe, vol, max drawdown, turnover, ...).
    diagnostics:
        Per-rebalance diagnostics (e.g. solver status, realized turnover, costs);
        free-form, keyed by date or by metric name.
    metadata:
        Run configuration / provenance (model name, schedule, budget, seeds, ...).
    """

    equity_curve: pd.Series
    weights: pd.DataFrame
    returns: pd.Series | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.equity_curve, pd.Series):
            raise ValidationError("BacktestResult.equity_curve must be a pandas Series.")
        if not isinstance(self.weights, pd.DataFrame):
            raise ValidationError("BacktestResult.weights must be a pandas DataFrame.")
        if self.returns is not None and not isinstance(self.returns, pd.Series):
            raise ValidationError("BacktestResult.returns must be a pandas Series or None.")

    @property
    def assets(self) -> list[str]:
        """Asset ids tracked through the backtest (weights columns)."""
        return [str(c) for c in self.weights.columns]

    @property
    def start(self) -> date | datetime | Any:
        """First date of the equity curve."""
        return self.equity_curve.index[0]

    @property
    def end(self) -> date | datetime | Any:
        """Last date of the equity curve."""
        return self.equity_curve.index[-1]


# ---------------------------------------------------------------------------
# Dynamic-allocation parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocatorParams:
    """Configuration for the dynamic (temporal) allocation layer (BUILD_PLAN §2).

    These parameters drive CPPI and the drawdown/floor allocators (``dynamic/``,
    Agent 8). The risky-asset allocation each period is ``m × cushion`` where
    ``cushion = (asset − floor) / asset`` and ``m`` is the multiplier; a
    drawdown variant ratchets the floor up to ``(1 − max_drawdown)`` of the
    running peak. The remainder sits in the safe asset, which (absent a safe
    return matrix) accrues at ``safe_rate``.

    Attributes
    ----------
    multiplier:
        The CPPI multiplier ``m`` (``> 0``). Allocation to the risky asset is
        ``m × cushion``, typically capped/floored downstream. Larger ``m`` means
        a more aggressive response to the cushion.
    floor:
        The protected wealth fraction (``0 <= floor < 1``) of the initial
        account value below which the strategy aims not to fall. ``0.8`` means
        protect 80% of starting wealth.
    max_drawdown:
        Optional maximum-drawdown limit (``0 < max_drawdown <= 1``) for the
        drawdown variant, where the floor trails the running peak at
        ``(1 − max_drawdown)``. ``None`` selects the plain fixed-floor CPPI.
    safe_rate:
        Annualized return of the safe asset used when no explicit safe-return
        series is supplied to the allocator. Must be finite; defaults to ``0.0``.
    start_value:
        Initial account value the floor is measured against (``> 0``); defaults
        to ``1.0`` (growth-of-$1 convention shared with
        :class:`BacktestResult`).

    Raises
    ------
    ValidationError
        If any field is non-finite or outside its documented range.
    """

    multiplier: float = 3.0
    floor: float = 0.8
    max_drawdown: float | None = None
    safe_rate: float = 0.0
    start_value: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValidationError("AllocatorParams.multiplier must be finite and positive.")
        if not np.isfinite(self.floor) or not (0.0 <= self.floor < 1.0):
            raise ValidationError("AllocatorParams.floor must be finite and in [0, 1).")
        if self.max_drawdown is not None and (
            not np.isfinite(self.max_drawdown) or not (0.0 < self.max_drawdown <= 1.0)
        ):
            raise ValidationError("AllocatorParams.max_drawdown must be finite and in (0, 1].")
        if not np.isfinite(self.safe_rate):
            raise ValidationError("AllocatorParams.safe_rate must be finite.")
        if not np.isfinite(self.start_value) or self.start_value <= 0:
            raise ValidationError("AllocatorParams.start_value must be finite and positive.")


__all__ = [
    "AllocatorParams",
    "BacktestResult",
    "ExpectedReturns",
    "Portfolio",
    "PriceData",
    "ReturnMatrix",
    "ReturnMethod",
    "RiskBudget",
]
