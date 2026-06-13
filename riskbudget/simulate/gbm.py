"""Geometric Brownian Motion scenario generation (BUILD_PLAN §4, Agent 2).

Monte-Carlo price-path simulation following the EDHEC course ``gbm()`` (BUILD_PLAN
§11, *Introduction to Portfolio Construction and Analysis with Python*). Under
GBM the per-step *gross* return is

    1 + rₜ = exp((μ − ½σ²)·dt + σ·√dt·Z),   Z ~ N(0, 1) i.i.d.

with ``dt = 1 / steps_per_year``. Equivalently each per-step *log* return is
normal with mean ``(μ − ½σ²)·dt`` and standard deviation ``σ·√dt``, so over a
year the log-returns have drift ``μ − ½σ²`` and volatility ``σ`` — the
parameterization the tests recover within sampling error.

This module returns simulated *price* paths as a ``pandas.DataFrame`` (one column
per scenario, index = time step), with helpers to convert to per-step simple
returns and to a :class:`~riskbudget.core.types.PriceData` panel, plus
terminal-wealth summaries (:func:`terminal_values`, :func:`terminal_stats`) for
the dynamic-allocation layer (Agent 8) and stress testing.

Determinism (BUILD_PLAN §3.1): every draw takes an explicit ``seed`` or
:class:`numpy.random.Generator`; identical inputs give identical paths.

Return convention: ``mu`` and ``sigma`` are *annualized* drift and volatility of
the continuously-compounded (log) process. The course's discrete prices use the
gross-return exponential form above, so a sanity check on log-returns recovers
``(μ − ½σ², σ)``; a check on per-step *simple* returns recovers an arithmetic
mean of ``μ·dt`` (Itô correction), matching the EDHEC convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData

__all__ = [
    "gbm",
    "gbm_price_data",
    "terminal_stats",
    "terminal_values",
]


def _coerce_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a generator from ``seed`` (int / Generator); reject ``None``."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        raise DataError("GBM simulation requires an explicit seed or Generator (no global RNG).")
    try:
        return np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise DataError(f"Invalid seed for GBM simulation: {seed!r} ({exc}).") from exc


def gbm(
    *,
    n_years: float = 10.0,
    steps_per_year: int = 252,
    mu: float = 0.07,
    sigma: float = 0.15,
    n_scenarios: int = 1000,
    s_0: float = 100.0,
    seed: int | np.random.Generator,
    prices: bool = True,
) -> pd.DataFrame:
    """Simulate GBM paths and return a ``(n_steps + 1, n_scenarios)`` DataFrame.

    Parameters
    ----------
    n_years:
        Horizon in years (``> 0``). Need not be integer.
    steps_per_year:
        Discretization steps per year (``>= 1``); ``252`` ≈ trading days.
    mu, sigma:
        Annualized drift and volatility of the log process. ``sigma >= 0``.
    n_scenarios:
        Number of independent paths (``>= 1``).
    s_0:
        Initial price level (``> 0``).
    seed:
        Explicit seed or :class:`numpy.random.Generator` (required).
    prices:
        If ``True`` (default) return price levels starting at ``s_0`` (row 0 is
        ``s_0`` for every scenario). If ``False`` return per-step *gross* returns
        ``1 + rₜ`` (row 0 is ``1.0``), matching the EDHEC ``gbm(prices=False)``.

    Returns
    -------
    pandas.DataFrame
        Shape ``(n_steps + 1, n_scenarios)`` with integer step index ``0..n_steps``
        and columns ``0..n_scenarios-1``. ``n_steps = round(n_years *
        steps_per_year)``.

    Raises
    ------
    DataError
        On non-positive horizon/steps/scenarios/``s_0`` or negative ``sigma`` or
        non-finite ``mu``/``sigma``.
    """
    if not np.isfinite(n_years) or n_years <= 0:
        raise DataError(f"n_years must be finite and > 0, got {n_years}.")
    if steps_per_year < 1:
        raise DataError(f"steps_per_year must be >= 1, got {steps_per_year}.")
    if n_scenarios < 1:
        raise DataError(f"n_scenarios must be >= 1, got {n_scenarios}.")
    if not np.isfinite(s_0) or s_0 <= 0:
        raise DataError(f"s_0 must be finite and > 0, got {s_0}.")
    if not np.isfinite(mu):
        raise DataError(f"mu must be finite, got {mu}.")
    if not np.isfinite(sigma) or sigma < 0:
        raise DataError(f"sigma must be finite and >= 0, got {sigma}.")

    n_steps = round(n_years * steps_per_year)
    if n_steps < 1:
        raise DataError(
            f"n_years * steps_per_year rounds to {n_steps} steps; increase the horizon."
        )

    dt = 1.0 / steps_per_year
    rng = _coerce_rng(seed)

    # Per-step gross returns 1 + r = exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z).
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    shocks = rng.standard_normal(size=(n_steps, n_scenarios))
    gross = np.exp(drift + diffusion * shocks)

    # Row 0 is the seed (1.0 for returns, s_0 for prices); rows 1.. are the draws.
    rets = np.empty((n_steps + 1, n_scenarios), dtype=float)
    rets[0, :] = 1.0
    rets[1:, :] = gross

    index = pd.RangeIndex(n_steps + 1, name="step")
    columns = pd.RangeIndex(n_scenarios, name="scenario")
    if not prices:
        return pd.DataFrame(rets, index=index, columns=columns)

    levels = s_0 * np.cumprod(rets, axis=0)
    return pd.DataFrame(levels, index=index, columns=columns)


def gbm_price_data(
    *,
    n_years: float = 10.0,
    steps_per_year: int = 252,
    mu: float = 0.07,
    sigma: float = 0.15,
    n_scenarios: int = 1000,
    s_0: float = 100.0,
    seed: int | np.random.Generator,
    start: str | pd.Timestamp = "2000-01-03",
    freq: str = "B",
    scenario_prefix: str = "SCEN_",
) -> PriceData:
    """Simulate GBM and wrap the price paths as a :class:`PriceData` panel.

    Each scenario becomes a column (``SCEN_0000, SCEN_0001, ...``) and the
    integer steps are mapped to a real date index (``freq`` business days by
    default), so the result drops straight into the rest of the pipeline.

    Raises
    ------
    DataError
        On the same conditions as :func:`gbm`.
    """
    frame = gbm(
        n_years=n_years,
        steps_per_year=steps_per_year,
        mu=mu,
        sigma=sigma,
        n_scenarios=n_scenarios,
        s_0=s_0,
        seed=seed,
        prices=True,
    )
    n_rows, n_cols = frame.shape
    width = max(4, len(str(n_cols - 1)))
    frame = frame.copy()
    frame.columns = [f"{scenario_prefix}{i:0{width}d}" for i in range(n_cols)]
    try:
        frame.index = pd.date_range(start=start, periods=n_rows, freq=freq)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"Could not build a GBM date index (start={start!r}, freq={freq!r}): {exc}."
        ) from exc
    return PriceData(frame)


def terminal_values(paths: pd.DataFrame) -> np.ndarray:
    """Terminal (last-row) wealth for each scenario of a GBM *price* frame.

    Parameters
    ----------
    paths:
        A price-path DataFrame as returned by :func:`gbm` with ``prices=True``
        (rows = steps, columns = scenarios).

    Returns
    -------
    numpy.ndarray
        Length ``n_scenarios`` vector of terminal prices.

    Raises
    ------
    DataError
        If ``paths`` is empty or non-finite.
    """
    if not isinstance(paths, pd.DataFrame) or paths.shape[0] == 0 or paths.shape[1] == 0:
        raise DataError("terminal_values expects a non-empty price-path DataFrame.")
    terminal = paths.iloc[-1].to_numpy(dtype=float)
    if not np.isfinite(terminal).all():
        raise DataError("Terminal values contain NaN or infinite entries.")
    return terminal


def terminal_stats(
    paths: pd.DataFrame,
    *,
    floor: float | None = None,
    cap: float | None = None,
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
) -> dict[str, Any]:
    """Summarize terminal wealth across GBM scenarios.

    Parameters
    ----------
    paths:
        Price-path DataFrame from :func:`gbm` (``prices=True``).
    floor:
        Optional wealth floor. When set, the report includes ``p_breach_floor``,
        the fraction of scenarios whose terminal wealth fell *below* ``floor``,
        and ``e_shortfall``, the mean shortfall ``max(floor − W_T, 0)`` (zero when
        no breach). Mirrors the EDHEC CPPI/“violation” diagnostics.
    cap:
        Optional wealth cap. When set, ``p_reach_cap`` reports the fraction of
        scenarios whose terminal wealth reached/exceeded ``cap``.
    percentiles:
        Percentiles (in ``[0, 100]``) to report under ``"percentiles"``.

    Returns
    -------
    dict
        ``{"mean", "median", "std", "min", "max", "n_scenarios", "percentiles",
        ...optional floor/cap keys...}``. ``percentiles`` maps each requested
        percentile to its terminal-wealth value.

    Raises
    ------
    DataError
        On an empty frame, a non-finite floor/cap, or out-of-range percentiles.
    """
    terminal = terminal_values(paths)

    if floor is not None and not np.isfinite(floor):
        raise DataError("floor must be finite when provided.")
    if cap is not None and not np.isfinite(cap):
        raise DataError("cap must be finite when provided.")
    for p in percentiles:
        if not np.isfinite(p) or not (0.0 <= p <= 100.0):
            raise DataError(f"percentiles must lie in [0, 100], got {p}.")

    pct_values = np.percentile(terminal, list(percentiles))
    stats: dict[str, Any] = {
        "n_scenarios": int(terminal.size),
        "mean": float(np.mean(terminal)),
        "median": float(np.median(terminal)),
        "std": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
        "min": float(np.min(terminal)),
        "max": float(np.max(terminal)),
        "percentiles": {float(p): float(v) for p, v in zip(percentiles, pct_values, strict=True)},
    }

    if floor is not None:
        breach = terminal < floor
        stats["floor"] = float(floor)
        stats["p_breach_floor"] = float(np.mean(breach))
        shortfall = np.clip(floor - terminal, 0.0, None)
        stats["e_shortfall"] = float(np.mean(shortfall))

    if cap is not None:
        stats["cap"] = float(cap)
        stats["p_reach_cap"] = float(np.mean(terminal >= cap))

    return stats
