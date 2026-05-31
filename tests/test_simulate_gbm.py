"""Tests for GBM scenario generation (Agent 2).

Covers: parameterization recovery (log-return mu/sigma within sampling error),
determinism, the price/return path forms, ``terminal_values`` / ``terminal_stats``
summaries (including floor-breach probability), the :class:`PriceData` wrapper,
error handling, and a property-based invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import DataError
from riskbudget.core.types import PriceData
from riskbudget.simulate.gbm import gbm, gbm_price_data, terminal_stats, terminal_values

# ---------------------------------------------------------------------------
# Shape & basic structure
# ---------------------------------------------------------------------------


def test_price_path_shape_and_seed_row() -> None:
    f = gbm(n_years=2, steps_per_year=252, n_scenarios=10, s_0=100.0, seed=0)
    assert f.shape == (2 * 252 + 1, 10)
    # Row 0 is the seed price for every scenario.
    assert np.allclose(f.iloc[0].to_numpy(), 100.0)


def test_return_form_first_row_is_one() -> None:
    f = gbm(n_years=1, steps_per_year=12, n_scenarios=5, seed=0, prices=False)
    assert np.allclose(f.iloc[0].to_numpy(), 1.0)
    assert (f.iloc[1:].to_numpy() > 0).all()


def test_fractional_years() -> None:
    f = gbm(n_years=0.5, steps_per_year=252, n_scenarios=3, seed=0)
    assert f.shape[0] == round(0.5 * 252) + 1


# ---------------------------------------------------------------------------
# Parameterization recovery
# ---------------------------------------------------------------------------


def test_log_returns_recover_mu_and_sigma() -> None:
    mu, sigma = 0.08, 0.20
    steps = 252
    f = gbm(
        n_years=40,
        steps_per_year=steps,
        mu=mu,
        sigma=sigma,
        n_scenarios=400,
        s_0=100.0,
        seed=2024,
    )
    levels = f.to_numpy()
    log_rets = np.log(levels[1:] / levels[:-1])  # (n_steps, n_scenarios)
    # Annualized: log-drift -> mu - 0.5 sigma^2, vol -> sigma.
    ann_drift = log_rets.mean() * steps
    ann_vol = log_rets.std() * np.sqrt(steps)
    assert ann_drift == pytest.approx(mu - 0.5 * sigma**2, abs=0.01)
    assert ann_vol == pytest.approx(sigma, abs=0.01)


def test_simple_returns_recover_arithmetic_mu() -> None:
    # Per-step simple returns have mean ~ mu * dt (Itô correction vs. log mean).
    mu, sigma, steps = 0.10, 0.15, 252
    f = gbm(n_years=60, steps_per_year=steps, mu=mu, sigma=sigma, n_scenarios=300, seed=7)
    levels = f.to_numpy()
    simple = levels[1:] / levels[:-1] - 1.0
    assert simple.mean() * steps == pytest.approx(mu, abs=0.01)


def test_zero_sigma_is_deterministic_drift() -> None:
    f = gbm(n_years=1, steps_per_year=12, mu=0.12, sigma=0.0, n_scenarios=4, s_0=100.0, seed=1)
    # With no diffusion every scenario is identical and grows at the drift.
    assert np.allclose(f.to_numpy()[:, 0:1], f.to_numpy())
    dt = 1.0 / 12
    expected_step = np.exp((0.12) * dt)  # mu - 0.5*0^2 = mu
    assert f.iloc[1, 0] / f.iloc[0, 0] == pytest.approx(expected_step)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed() -> None:
    a = gbm(n_years=3, n_scenarios=20, seed=42)
    b = gbm(n_years=3, n_scenarios=20, seed=42)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_different_seed_differs() -> None:
    a = gbm(n_years=3, n_scenarios=20, seed=1)
    b = gbm(n_years=3, n_scenarios=20, seed=2)
    assert not np.array_equal(a.to_numpy(), b.to_numpy())


def test_generator_threads_through() -> None:
    a = gbm(n_years=1, n_scenarios=5, seed=np.random.default_rng(9))
    b = gbm(n_years=1, n_scenarios=5, seed=np.random.default_rng(9))
    assert np.array_equal(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# Terminal values & stats
# ---------------------------------------------------------------------------


def test_terminal_values() -> None:
    f = gbm(n_years=2, n_scenarios=50, s_0=100.0, seed=3)
    tv = terminal_values(f)
    assert tv.shape == (50,)
    assert np.allclose(tv, f.iloc[-1].to_numpy())
    assert (tv > 0).all()


def test_terminal_stats_basic() -> None:
    f = gbm(n_years=10, mu=0.07, sigma=0.15, n_scenarios=2000, s_0=100.0, seed=11)
    stats = terminal_stats(f)
    assert stats["n_scenarios"] == 2000
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["mean"] > 0
    # Median terminal wealth should exceed the starting 100 for a positive drift.
    assert stats["median"] > 100.0
    pct = stats["percentiles"]
    # Percentiles are monotone non-decreasing.
    keys = sorted(pct)
    vals = [pct[k] for k in keys]
    assert vals == sorted(vals)


def test_terminal_stats_floor_breach() -> None:
    f = gbm(n_years=10, mu=0.07, sigma=0.15, n_scenarios=3000, s_0=100.0, seed=5)
    stats = terminal_stats(f, floor=100.0)
    assert "p_breach_floor" in stats
    assert 0.0 <= stats["p_breach_floor"] <= 1.0
    assert stats["e_shortfall"] >= 0.0
    # A floor at/above the max can never be breached; below the min always is.
    tv = terminal_values(f)
    above_all = terminal_stats(f, floor=float(tv.min()) - 1.0)
    assert above_all["p_breach_floor"] == 0.0
    below_all = terminal_stats(f, floor=float(tv.max()) + 1.0)
    assert below_all["p_breach_floor"] == 1.0


def test_terminal_stats_cap() -> None:
    f = gbm(n_years=5, n_scenarios=1000, s_0=100.0, seed=6)
    tv = terminal_values(f)
    stats = terminal_stats(f, cap=float(np.median(tv)))
    assert 0.0 <= stats["p_reach_cap"] <= 1.0


def test_terminal_stats_custom_percentiles() -> None:
    f = gbm(n_years=2, n_scenarios=500, seed=8)
    stats = terminal_stats(f, percentiles=(10.0, 90.0))
    assert set(stats["percentiles"].keys()) == {10.0, 90.0}


# ---------------------------------------------------------------------------
# PriceData wrapper
# ---------------------------------------------------------------------------


def test_gbm_price_data() -> None:
    pdat = gbm_price_data(n_years=2, n_scenarios=6, seed=0)
    assert isinstance(pdat, PriceData)
    assert pdat.shape[1] == 6
    assert all(a.startswith("SCEN_") for a in pdat.assets)
    assert isinstance(pdat.dates, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_seed_none_rejected() -> None:
    with pytest.raises(DataError, match="explicit seed"):
        gbm(n_years=1, n_scenarios=5, seed=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_years": 0}, "n_years"),
        ({"n_years": -1}, "n_years"),
        ({"steps_per_year": 0}, "steps_per_year"),
        ({"n_scenarios": 0}, "n_scenarios"),
        ({"s_0": 0}, "s_0"),
        ({"s_0": -1}, "s_0"),
        ({"sigma": -0.1}, "sigma"),
        ({"mu": np.inf}, "mu"),
    ],
)
def test_invalid_params(kwargs: dict[str, float], match: str) -> None:
    base = {"n_years": 1, "n_scenarios": 5, "seed": 0}
    base.update(kwargs)
    with pytest.raises(DataError, match=match):
        gbm(**base)  # type: ignore[arg-type]


def test_terminal_values_empty_rejected() -> None:
    with pytest.raises(DataError, match="non-empty"):
        terminal_values(pd.DataFrame())


def test_terminal_stats_bad_percentile() -> None:
    f = gbm(n_years=1, n_scenarios=5, seed=0)
    with pytest.raises(DataError, match="percentiles"):
        terminal_stats(f, percentiles=(150.0,))


def test_terminal_stats_bad_floor() -> None:
    f = gbm(n_years=1, n_scenarios=5, seed=0)
    with pytest.raises(DataError, match="floor"):
        terminal_stats(f, floor=float("nan"))


# ---------------------------------------------------------------------------
# Property-based invariants (deterministic loop; hypothesis unavailable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(8))
def test_property_paths_positive_and_finite(trial: int) -> None:
    rng = np.random.default_rng(trial)
    mu = float(rng.uniform(-0.1, 0.2))
    sigma = float(rng.uniform(0.0, 0.4))
    n_scen = int(rng.integers(2, 30))
    f = gbm(n_years=2, steps_per_year=52, mu=mu, sigma=sigma, n_scenarios=n_scen, seed=trial)
    arr = f.to_numpy()
    assert np.isfinite(arr).all()
    assert (arr > 0).all()
    # Row 0 is exactly s_0 (default 100) everywhere.
    assert np.allclose(arr[0], 100.0)


@pytest.mark.parametrize("trial", range(5))
def test_property_log_drift_matches(trial: int) -> None:
    rng = np.random.default_rng(100 + trial)
    mu = float(rng.uniform(0.02, 0.15))
    sigma = float(rng.uniform(0.05, 0.25))
    steps = 252
    f = gbm(n_years=50, steps_per_year=steps, mu=mu, sigma=sigma, n_scenarios=200, seed=trial)
    levels = f.to_numpy()
    log_rets = np.log(levels[1:] / levels[:-1])
    ann_drift = log_rets.mean() * steps
    assert ann_drift == pytest.approx(mu - 0.5 * sigma**2, abs=0.02)
