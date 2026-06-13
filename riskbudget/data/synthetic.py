"""Correlated-returns synthetic data source — the always-offline backbone.

This module is the offline data backbone the whole test suite relies on
(BUILD_PLAN §4, Agent 2). It draws multivariate return panels whose *sample*
covariance converges to a known target covariance, so downstream agents can
estimate risk models, solve risk budgets, and backtest without a network.

Two ways to specify the cross-asset risk structure are supported:

1. **Explicit target covariance** — pass an ``(N, N)`` symmetric PSD matrix and
   the generator draws zero-mean (or ``mu``-mean) returns whose sample
   covariance tends to it as the number of periods grows.
2. **Factor structure** — pass a ``(N, K)`` factor-loading matrix ``B``, the
   ``K`` factor volatilities, and the ``N`` idiosyncratic vols; the implied
   covariance is ``Σ = B diag(f²) Bᵀ + diag(ε²)``.

The draw uses the symmetric (eigendecomposition) square root of the target
covariance applied to i.i.d. standard-normal innovations. The eigendecomposition
route (rather than a Cholesky factor) tolerates a covariance that is only
positive *semi*-definite — e.g. a factor model with ``K < N`` and zero
idiosyncratic vol — which Cholesky would reject.

Determinism (BUILD_PLAN §3.1): every draw takes an explicit ``seed`` or
``numpy.random.Generator``; identical inputs produce identical output. No
implicit global RNG is ever touched.

Return convention: the generated *returns* are simple (arithmetic) returns with
the requested per-period mean and covariance. Prices are reconstructed by
compounding ``Pₜ = P₀ · Π(1 + rₛ)`` so that ``PriceData.to_returns("simple")``
recovers the generated returns exactly (up to floating point).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData, ReturnMatrix

__all__ = [
    "SyntheticDataSource",
    "covariance_from_factors",
    "generate_correlated_returns",
    "synthetic_data_source",
]

# Treat tiny negative eigenvalues (floating-point noise on a PSD matrix) as zero
# rather than rejecting the covariance.
_EIG_TOL = 1e-10
_SYMMETRY_ATOL = 1e-8


def _coerce_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a NumPy generator from ``seed`` (int, Generator, or ``None``).

    Passing an existing :class:`numpy.random.Generator` threads it through
    unchanged; an int seeds a fresh generator; ``None`` is rejected because
    BUILD_PLAN §3.1 forbids implicit/global RNG state.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        raise DataError("A synthetic draw requires an explicit seed or Generator (no global RNG).")
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Invalid seed for synthetic draw: {seed!r} ({exc}).") from exc


def _validate_cov(cov: Any, *, n_assets: int | None) -> np.ndarray:
    """Validate and return a symmetric PSD-ish covariance as a float ndarray."""
    try:
        sigma = np.asarray(cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Covariance could not be read as a float array: {exc}.") from exc
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise DataError(f"Covariance must be a square matrix, got shape {sigma.shape}.")
    if n_assets is not None and sigma.shape[0] != n_assets:
        raise DataError(
            f"Covariance shape {sigma.shape} does not match the number of assets ({n_assets})."
        )
    if not np.isfinite(sigma).all():
        raise DataError("Covariance contains NaN or infinite values.")
    if not np.allclose(sigma, sigma.T, atol=_SYMMETRY_ATOL):
        raise DataError("Covariance matrix is not symmetric.")
    return sigma


def _symmetric_sqrt(sigma: np.ndarray) -> np.ndarray:
    """Symmetric square root ``L`` with ``L Lᵀ = Σ`` via eigendecomposition.

    Negative eigenvalues below ``-_EIG_TOL`` mean the matrix is materially
    non-PSD and a real factorization does not exist; we reject those. Tiny
    negatives (floating-point noise) are clipped to zero.
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    if float(eigvals.min()) < -_EIG_TOL:
        raise DataError(
            "Target covariance is not positive semi-definite "
            f"(min eigenvalue {float(eigvals.min()):.3e}); cannot factorize."
        )
    clipped = np.clip(eigvals, 0.0, None)
    root: np.ndarray = eigvecs @ np.diag(np.sqrt(clipped)) @ eigvecs.T
    return root


def covariance_from_factors(
    loadings: Any,
    factor_vols: Any,
    idiosyncratic_vols: Any,
) -> np.ndarray:
    """Build a covariance from a linear factor structure.

    ``Σ = B diag(f²) Bᵀ + diag(ε²)`` where ``B`` is the ``(N, K)`` loading
    matrix, ``f`` the ``K`` factor volatilities, and ``ε`` the ``N``
    idiosyncratic (asset-specific) volatilities.

    Parameters
    ----------
    loadings:
        ``(N, K)`` factor loadings ``B``.
    factor_vols:
        Length-``K`` factor volatilities (standard deviations, ``>= 0``).
    idiosyncratic_vols:
        Length-``N`` idiosyncratic volatilities (``>= 0``).

    Returns
    -------
    numpy.ndarray
        The implied ``(N, N)`` symmetric PSD covariance.

    Raises
    ------
    DataError
        On shape mismatch, non-finite, or negative volatility input.
    """
    try:
        b = np.asarray(loadings, dtype=float)
        f = np.asarray(factor_vols, dtype=float)
        eps = np.asarray(idiosyncratic_vols, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Factor inputs could not be read as float arrays: {exc}.") from exc

    if b.ndim != 2:
        raise DataError(f"Loadings must be a 2-D (N, K) matrix, got shape {b.shape}.")
    n, k = b.shape
    if f.shape != (k,):
        raise DataError(f"factor_vols must have shape ({k},), got {f.shape}.")
    if eps.shape != (n,):
        raise DataError(f"idiosyncratic_vols must have shape ({n},), got {eps.shape}.")
    if not (np.isfinite(b).all() and np.isfinite(f).all() and np.isfinite(eps).all()):
        raise DataError("Factor inputs contain NaN or infinite values.")
    if (f < 0).any() or (eps < 0).any():
        raise DataError("Volatilities (factor and idiosyncratic) must be non-negative.")

    factor_cov = b @ np.diag(f**2) @ b.T
    sigma = factor_cov + np.diag(eps**2)
    # Symmetrize away floating-point asymmetry from the matrix products.
    symmetric: np.ndarray = 0.5 * (sigma + sigma.T)
    return symmetric


def _build_dates(n_periods: int, start: date | str | pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    """Construct an ascending business/calendar date index of ``n_periods``."""
    try:
        return pd.date_range(start=start, periods=n_periods, freq=freq)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"Could not build a date index (start={start!r}, freq={freq!r}): {exc}."
        ) from exc


def _resolve_assets(
    assets: Sequence[str] | None,
    n_assets: int | None,
    cov_n: int,
) -> list[str]:
    """Resolve the asset id list, defaulting to ``ASSET_0..`` names."""
    if assets is not None:
        asset_list = [str(a) for a in assets]
        if len(asset_list) != cov_n:
            raise DataError(
                f"Got {len(asset_list)} asset ids but the covariance is {cov_n}x{cov_n}."
            )
        if len(set(asset_list)) != len(asset_list):
            raise DataError("Duplicate asset ids supplied to the synthetic generator.")
        return asset_list
    count = cov_n if n_assets is None else n_assets
    width = max(2, len(str(count - 1)))
    return [f"ASSET_{i:0{width}d}" for i in range(count)]


def generate_correlated_returns(
    *,
    n_periods: int,
    cov: Any | None = None,
    n_assets: int | None = None,
    assets: Sequence[str] | None = None,
    mu: Any | float = 0.0,
    loadings: Any | None = None,
    factor_vols: Any | None = None,
    idiosyncratic_vols: Any | None = None,
    seed: int | np.random.Generator,
    start: date | str | pd.Timestamp = "2000-01-03",
    freq: str = "B",
) -> ReturnMatrix:
    """Draw a correlated *simple-return* panel matching a target covariance.

    Exactly one risk structure must be supplied: either an explicit ``cov`` or a
    full factor triple (``loadings`` + ``factor_vols`` + ``idiosyncratic_vols``).

    Parameters
    ----------
    n_periods:
        Number of return periods (rows). Must be ``>= 1``.
    cov:
        ``(N, N)`` target covariance. Mutually exclusive with the factor inputs.
    n_assets:
        Used only to size default asset names when neither ``assets`` nor a
        covariance fixes ``N`` (it is otherwise inferred).
    assets:
        Explicit asset ids (length ``N``). Defaults to ``ASSET_00, ASSET_01,...``.
    mu:
        Per-period mean return: a scalar broadcast to all assets or a length-``N``
        vector. Defaults to ``0.0``.
    loadings, factor_vols, idiosyncratic_vols:
        Factor structure ``Σ = B diag(f²) Bᵀ + diag(ε²)`` (see
        :func:`covariance_from_factors`). All three are required together.
    seed:
        Explicit seed (int) or :class:`numpy.random.Generator`. Required —
        BUILD_PLAN §3.1 forbids implicit RNG.
    start, freq:
        Start date and pandas frequency for the generated date index. ``"B"``
        (business days) by default.

    Returns
    -------
    ReturnMatrix
        ``(n_periods, N)`` of simple returns.

    Raises
    ------
    DataError
        On conflicting/missing risk-structure inputs, bad shapes, a non-PSD
        covariance, or ``n_periods < 1``.
    """
    if n_periods < 1:
        raise DataError(f"n_periods must be >= 1, got {n_periods}.")

    has_factor = any(x is not None for x in (loadings, factor_vols, idiosyncratic_vols))
    if cov is not None and has_factor:
        raise DataError("Provide either 'cov' or the factor inputs, not both.")
    if cov is None and not has_factor:
        raise DataError("Provide a target 'cov' or the full factor triple.")

    if has_factor:
        if loadings is None or factor_vols is None or idiosyncratic_vols is None:
            raise DataError(
                "Factor structure requires loadings, factor_vols, and idiosyncratic_vols together."
            )
        sigma = covariance_from_factors(loadings, factor_vols, idiosyncratic_vols)
    else:
        sigma = _validate_cov(cov, n_assets=None)

    cov_n = sigma.shape[0]
    asset_list = _resolve_assets(assets, n_assets, cov_n)

    mu_vec = np.asarray(mu, dtype=float)
    if mu_vec.ndim == 0:
        mu_vec = np.full(cov_n, float(mu_vec))
    elif mu_vec.shape != (cov_n,):
        raise DataError(f"mu must be a scalar or length-{cov_n} vector, got shape {mu_vec.shape}.")
    if not np.isfinite(mu_vec).all():
        raise DataError("mu contains NaN or infinite values.")

    rng = _coerce_rng(seed)
    sqrt_sigma = _symmetric_sqrt(sigma)
    innovations = rng.standard_normal(size=(n_periods, cov_n))
    returns = innovations @ sqrt_sigma + mu_vec

    dates = _build_dates(n_periods, start, freq)
    frame = pd.DataFrame(returns, index=dates, columns=asset_list)
    return ReturnMatrix(frame)


def returns_to_prices(
    returns: ReturnMatrix,
    *,
    initial_price: float | Any = 100.0,
) -> PriceData:
    """Compound simple returns into a strictly-positive price panel.

    A leading price row (at one step before the first return date) is prepended
    so the panel has one more row than ``returns`` and
    ``PriceData.to_returns("simple")`` exactly recovers ``returns``.

    Raises
    ------
    DataError
        If any compounded price is non-positive (a simple return ``<= -1``),
        which :class:`PriceData` would otherwise reject opaquely.
    """
    rmat = returns.values
    n_assets = rmat.shape[1]

    p0 = np.asarray(initial_price, dtype=float)
    if p0.ndim == 0:
        p0 = np.full(n_assets, float(p0))
    elif p0.shape != (n_assets,):
        raise DataError(
            f"initial_price must be a scalar or length-{n_assets} vector, got {p0.shape}."
        )
    if (p0 <= 0).any() or not np.isfinite(p0).all():
        raise DataError("initial_price must be finite and strictly positive.")

    growth = np.cumprod(1.0 + rmat, axis=0)
    if (growth <= 0).any():
        raise DataError(
            "Compounded prices went non-positive (a simple return <= -1); "
            "reduce volatility or mean for a valid price path."
        )
    prices = p0 * growth  # rows align with the return dates

    return_dates = returns.dates
    # Prepend an initial date one step before the first return.
    if len(return_dates) >= 2:
        step = return_dates[1] - return_dates[0]
        first_date = return_dates[0] - step
    else:
        first_date = return_dates[0] - pd.Timedelta(days=1)

    full_index = pd.DatetimeIndex([first_date, *list(return_dates)])
    full_values = np.vstack([p0, prices])
    frame = pd.DataFrame(full_values, index=full_index, columns=returns.assets)
    return PriceData(frame)


class SyntheticDataSource:
    """A seeded, deterministic :class:`DataSource` backed by a known covariance.

    Implements the :class:`~riskbudget.core.interfaces.DataSource` protocol
    (``get_prices(assets, start, end) -> PriceData``). Construct it with a target
    covariance (or a factor structure) and a seed; :meth:`get_prices` returns a
    compounded price panel whose simple returns have that covariance and the
    requested per-period mean.

    The price panel is generated lazily for the union of requested assets and the
    requested date range. Because the seed is fixed, two calls with the same
    arguments return identical prices (BUILD_PLAN §3.1 determinism).

    Parameters
    ----------
    cov:
        ``(N, N)`` target covariance. Mutually exclusive with the factor inputs.
    assets:
        Asset ids for the universe (length ``N``). Defaults to generated names.
    mu:
        Per-period mean return (scalar or length-``N`` vector). Default ``0.0``.
    loadings, factor_vols, idiosyncratic_vols:
        Factor structure, as an alternative to ``cov`` (all three required).
    seed:
        Explicit seed (required).
    freq:
        pandas frequency for the generated dates (default ``"B"``, business days).
    initial_price:
        Starting price level for every asset (default ``100.0``).

    Raises
    ------
    DataError
        On the same conditions as :func:`generate_correlated_returns`.
    """

    def __init__(
        self,
        *,
        cov: Any | None = None,
        assets: Sequence[str] | None = None,
        mu: Any | float = 0.0,
        loadings: Any | None = None,
        factor_vols: Any | None = None,
        idiosyncratic_vols: Any | None = None,
        seed: int = 0,
        freq: str = "B",
        initial_price: float = 100.0,
    ) -> None:
        has_factor = any(x is not None for x in (loadings, factor_vols, idiosyncratic_vols))
        if cov is not None and has_factor:
            raise DataError("Provide either 'cov' or the factor inputs, not both.")
        if cov is None and not has_factor:
            raise DataError("Provide a target 'cov' or the full factor triple.")

        if has_factor:
            if loadings is None or factor_vols is None or idiosyncratic_vols is None:
                raise DataError(
                    "Factor structure requires loadings, factor_vols, and idiosyncratic_vols."
                )
            self._cov = covariance_from_factors(loadings, factor_vols, idiosyncratic_vols)
        else:
            self._cov = _validate_cov(cov, n_assets=None)

        self._assets = _resolve_assets(assets, None, self._cov.shape[0])
        self._asset_index = {a: i for i, a in enumerate(self._assets)}

        mu_vec = np.asarray(mu, dtype=float)
        if mu_vec.ndim == 0:
            mu_vec = np.full(self._cov.shape[0], float(mu_vec))
        elif mu_vec.shape != (self._cov.shape[0],):
            raise DataError(
                f"mu must be a scalar or length-{self._cov.shape[0]} vector, got {mu_vec.shape}."
            )
        if not np.isfinite(mu_vec).all():
            raise DataError("mu contains NaN or infinite values.")
        self._mu = mu_vec

        if not np.isfinite(initial_price) or initial_price <= 0:
            raise DataError("initial_price must be finite and strictly positive.")
        self._initial_price = float(initial_price)
        self._seed = seed
        self._freq = freq

    @property
    def assets(self) -> list[str]:
        """The full asset universe this source can serve."""
        return list(self._assets)

    @property
    def covariance(self) -> np.ndarray:
        """A copy of the per-period target covariance."""
        return self._cov.copy()

    def get_prices(self, assets: list[str], start: date, end: date) -> PriceData:
        """Return a deterministic :class:`PriceData` panel for ``assets``.

        The full universe's returns are drawn once with the fixed seed, then the
        requested ``assets`` columns and ``[start, end]`` rows are sliced out — so
        a sub-universe request is a consistent slice of the same joint draw
        (cross-asset correlations are preserved).

        Raises
        ------
        DataError
            If ``assets`` is empty, contains unknown ids, or ``start > end``, or
            if the requested window contains no generated dates.
        """
        if not assets:
            raise DataError("get_prices requires a non-empty list of assets.")
        requested = [str(a) for a in assets]
        unknown = [a for a in requested if a not in self._asset_index]
        if unknown:
            raise DataError(
                f"Unknown assets for this synthetic source: {unknown}. Available: {self._assets}."
            )
        if len(set(requested)) != len(requested):
            raise DataError("Duplicate assets requested from the synthetic source.")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts > end_ts:
            raise DataError(f"start ({start_ts.date()}) must not be after end ({end_ts.date()}).")

        # Build the price date index spanning [start, end], then draw one fewer
        # return so the recovered returns line up with the price dates after the
        # leading price row.
        price_dates = pd.date_range(start=start_ts, end=end_ts, freq=self._freq)
        if len(price_dates) < 2:
            raise DataError(
                f"The window [{start_ts.date()}, {end_ts.date()}] yields fewer than two "
                f"'{self._freq}' periods; widen the range."
            )

        n_returns = len(price_dates) - 1
        returns = generate_correlated_returns(
            n_periods=n_returns,
            cov=self._cov,
            assets=self._assets,
            mu=self._mu,
            seed=self._seed,
            start=price_dates[1],
            freq=self._freq,
        )

        # Compound to prices for the full universe, then slice the columns.
        prices = returns_to_prices(returns, initial_price=self._initial_price)
        full_frame = prices.frame
        # Align the price index to the requested calendar dates exactly.
        full_frame.index = price_dates
        sliced = full_frame.loc[:, requested]
        return PriceData(sliced)


def synthetic_data_source(
    *,
    cov: Any | None = None,
    assets: Sequence[str] | None = None,
    mu: Any | float = 0.0,
    loadings: Any | None = None,
    factor_vols: Any | None = None,
    idiosyncratic_vols: Any | None = None,
    seed: int = 0,
    freq: str = "B",
    initial_price: float = 100.0,
) -> SyntheticDataSource:
    """Registry-friendly factory for :class:`SyntheticDataSource` (BUILD_PLAN §5.2)."""
    return SyntheticDataSource(
        cov=cov,
        assets=assets,
        mu=mu,
        loadings=loadings,
        factor_vols=factor_vols,
        idiosyncratic_vols=idiosyncratic_vols,
        seed=seed,
        freq=freq,
        initial_price=initial_price,
    )
