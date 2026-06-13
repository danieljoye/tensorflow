"""Regression tests for fetch_prices default windows (run.py).

A spec that leaves ``start``/``end`` unset must get the SOURCE'S FULL available
range for recorded-history sources — not a hardcoded 2015-2021 window — while the
synthetic source keeps its bounded deterministic default (it *generates* data over
the requested window).
"""

from __future__ import annotations

import riskbudget as rb


def test_recorded_source_defaults_to_full_range() -> None:
    spec = rb.StrategySpec(
        name="full",
        assets=["STOCKS_TR", "BONDS", "GOLD"],
        data_source={"name": "daily_panel"},
        method="erc",
        risk_model="sample",
        periods_per_year=252,
    )
    prices = rb.fetch_prices(spec)
    # the committed daily panel starts 1968 and runs to ~present; the default
    # window must not truncate it to the old 2015-2021 synthetic default.
    assert prices.dates.min().year == 1968
    assert prices.dates.max().year >= 2024
    assert prices.shape[0] > 10_000


def test_synthetic_source_keeps_bounded_default() -> None:
    cov = [[4e-4, 1e-4, 0.0], [1e-4, 9e-4, 2e-4], [0.0, 2e-4, 1.6e-3]]
    spec = rb.StrategySpec(
        name="syn",
        assets=["A", "B", "C"],
        data_source={"name": "synthetic", "params": {"cov": cov}},
        method="erc",
        risk_model="sample",
        periods_per_year=252,
    )
    prices = rb.fetch_prices(spec)
    # the synthetic default window is ~6 years (2015-2021), not centuries.
    assert prices.dates.min().year == 2015
    assert prices.dates.max().year <= 2021


def test_explicit_dates_still_honored() -> None:
    from datetime import date

    spec = rb.StrategySpec(
        name="windowed",
        assets=["GOLD"],
        data_source={"name": "daily_panel"},
        method="erc",
        risk_model="sample",
        periods_per_year=252,
        start=date(2000, 1, 1),
        end=date(2005, 1, 1),
    )
    prices = rb.fetch_prices(spec)
    assert prices.dates.min().year == 2000
    assert prices.dates.max().year <= 2005
