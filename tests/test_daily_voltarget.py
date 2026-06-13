"""Offline test: vol targeting estimates volatility from 252 daily returns.

Loads the committed real daily sector panel and verifies that, run on daily data
with a 252-period lookback, the covariance/vol window is exactly 252 daily
observations and the vol-targeted book realizes near the target.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import riskbudget as rb


def _load_example() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "daily_voltarget_demo.py"
    spec = importlib.util.spec_from_file_location("daily_voltarget_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vol_window_is_252_daily_returns() -> None:
    ex = _load_example()
    panel = ex.load_panel()
    assets = list(panel.assets)
    res = rb.backtest(ex.make_spec("ERC", "erc", assets=assets, target_vol=0.10), prices=panel)
    per = res.diagnostics["per_rebalance"]
    # every covariance/vol window uses exactly 252 daily observations
    assert {d["window_len"] for d in per} == {252}
    # monthly rebalancing on daily data -> dozens of rebalances over ~4 usable years
    assert res.diagnostics["n_rebalances"] >= 36
    # realized annualized vol (daily) lands near the 10% target
    realized = float(res.returns.std() * np.sqrt(252))
    assert 0.07 < realized < 0.14


def test_daily_panel_is_real_and_daily() -> None:
    ex = _load_example()
    panel = ex.load_panel()
    dates = panel.dates
    assert len(dates) > 1000  # ~5 years of trading days
    # median gap between observations is ~1 day (daily), not ~30 (monthly)
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    assert np.median(gaps) <= 3
