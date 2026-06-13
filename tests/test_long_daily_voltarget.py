"""Offline smoke test for the long daily total-return vol-targeting example.

Loads ``examples/long_daily_voltarget_demo.py`` by file path (the ``examples``
package is not importable under pytest) and runs it against the committed offline
fixture so CI never touches the network. Verifies the demo wires the long daily
``STOCKS_TR``/``BONDS``/``GOLD`` panel through every method, vol-targeted to 10%.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import riskbudget as rb


def _load_example() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "long_daily_voltarget_demo.py"
    spec = importlib.util.spec_from_file_location("long_daily_voltarget_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_panel_is_long_daily_total_return() -> None:
    ex = _load_example()
    panel = ex.load_panel(offline=True)
    assert list(panel.assets) == ["STOCKS_TR", "BONDS", "GOLD"]
    # Long daily history: tens of thousands... at least >10k daily rows.
    assert panel.shape[0] > 10_000
    assert (panel.values > 0).all()


def test_offline_backtest_vol_targets_each_method() -> None:
    ex = _load_example()
    panel = ex.load_panel(offline=True)
    assets = list(panel.assets)
    for method in ex.METHODS.values():
        res = rb.backtest(
            ex.make_spec(method, method, assets=assets, target_vol=0.10), prices=panel
        )
        per = res.diagnostics["per_rebalance"]
        assert {d["window_len"] for d in per} == {252}
        realized = float(res.returns.std() * np.sqrt(252))
        assert 0.05 < realized < 0.16
