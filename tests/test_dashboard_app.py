"""Dashboard import/build tests (Agent 7): no live Streamlit server.

Importing the module must stay headless-safe (no ``streamlit`` at import time),
and the pure data-assembly helpers must run on the synthetic backbone.
"""

from __future__ import annotations

import pandas as pd
from riskbudget.dashboard import app as dashboard_app


def test_module_imports_without_streamlit() -> None:
    # The module must import cleanly; streamlit is only pulled inside render().
    assert hasattr(dashboard_app, "render")
    assert hasattr(dashboard_app, "build_dashboard_payload")


def test_default_covariance_is_square_and_symmetric() -> None:
    assets = ["A", "B", "C", "D"]
    cov = dashboard_app.default_covariance(assets, seed=1)
    assert len(cov) == len(assets)
    assert all(len(row) == len(assets) for row in cov)
    for i in range(len(assets)):
        for j in range(len(assets)):
            assert abs(cov[i][j] - cov[j][i]) < 1e-12


def test_build_payload_assembles_weights_and_summary() -> None:
    spec = dashboard_app.build_spec(
        name="ERC",
        assets=["A", "B", "C", "D"],
        method="erc",
        risk_model="sample",
        frequency="monthly",
        lookback=60,
    )
    payload = dashboard_app.build_dashboard_payload(
        spec, benchmark_method="equal_weight", make_figures=False
    )
    assert set(payload.weights) == {"A", "B", "C", "D"}
    assert abs(sum(payload.weights.values()) - 1.0) < 1e-6
    assert payload.volatility > 0.0
    assert isinstance(payload.summary, pd.DataFrame)
    # Strategy + benchmark rows.
    assert payload.summary.shape[0] == 2
    assert "ERC" in payload.summary.index
    assert payload.result.returns is not None


def test_build_payload_with_figures() -> None:
    # make_figures=True pulls plotly (a declared dependency) via reporting.
    spec = dashboard_app.build_spec(
        name="GMV",
        assets=["A", "B", "C"],
        method="gmv",
        risk_model="sample",
        frequency="monthly",
        lookback=60,
    )
    payload = dashboard_app.build_dashboard_payload(
        spec, benchmark_method="equal_weight", make_figures=True
    )
    assert "equity" in payload.figures
    assert "drawdown" in payload.figures
