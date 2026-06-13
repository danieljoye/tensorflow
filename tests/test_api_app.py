"""FastAPI surface tests (Agent 7): /health, /methods, /construct, /backtest.

All offline: the synthetic data source is seeded from a hand-written covariance so
the responses are deterministic. The web stack is imported lazily inside the app,
so these tests are the only place ``fastapi`` is pulled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Importing the Starlette/FastAPI TestClient emits a deprecation warning under the
# pinned httpx; the project runs pytest with ``filterwarnings = ["error"]`` so we
# suppress it here (it is a third-party import-time warning, not our code).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from riskbudget.api.app import create_app


def _cov(n: int = 3) -> list[list[float]]:
    """A small, well-conditioned per-period covariance for ``n`` assets."""
    base = np.eye(n) * 0.04 + 0.005
    return base.tolist()


def _synthetic_source(n: int = 3) -> dict[str, object]:
    return {"name": "synthetic", "params": {"cov": _cov(n)}}


@pytest.fixture
def client() -> TestClient:
    """A TestClient over a freshly built app."""
    return TestClient(create_app())


@pytest.fixture
def assets() -> list[str]:
    return ["AAA", "BBB", "CCC"]


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_methods_lists_registry(client: TestClient) -> None:
    resp = client.get("/methods")
    assert resp.status_code == 200
    body = resp.json()
    method_names = {m["name"] for m in body["methods"]}
    # The full documented method set must be selectable.
    for expected in ("erc", "risk_budget", "gmv", "msr", "hrp", "mdp", "equal_weight"):
        assert expected in method_names
    assert "synthetic" in body["data_sources"]
    assert "sample" in body["risk_models"]
    # msr is flagged as requiring a mean model.
    msr = next(m for m in body["methods"] if m["name"] == "msr")
    assert msr["requires_mu"] is True


def test_construct_returns_schema_valid_response(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "erc",
        "risk_model": "sample",
        "data_source": _synthetic_source(),
    }
    resp = client.post("/construct", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["method"] == "erc"
    assert set(data["weights"]) == set(assets)
    assert set(data["risk_contributions"]) == set(assets)
    # Long-only ERC: weights are non-negative and sum to ~1.
    assert all(w >= -1e-9 for w in data["weights"].values())
    assert data["weights"] and abs(sum(data["weights"].values()) - 1.0) < 1e-6
    assert data["volatility"] > 0.0


def test_construct_with_explicit_budget(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "risk_budget",
        "data_source": _synthetic_source(),
        "budget": {"AAA": 2.0, "BBB": 1.0, "CCC": 1.0},
    }
    resp = client.post("/construct", json=body)
    assert resp.status_code == 200, resp.text
    rc = resp.json()["risk_contributions"]
    # The asset with the larger budget carries the larger risk contribution.
    assert rc["AAA"] > rc["BBB"]


def test_backtest_returns_metrics_and_curve(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "erc",
        "data_source": _synthetic_source(),
        "schedule": {"frequency": "monthly", "lookback": 60},
        "include_equity_curve": True,
    }
    resp = client.post("/backtest", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["n_periods"] > 0
    assert "sharpe_ratio" in data["metrics"]
    assert "annualized_volatility" in data["metrics"]
    assert data["equity_curve"] is not None
    assert len(data["equity_curve"]) >= data["n_periods"]


def test_backtest_without_equity_curve(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "equal_weight",
        "data_source": _synthetic_source(),
        "schedule": {"frequency": "monthly", "lookback": 60},
        "include_equity_curve": False,
    }
    resp = client.post("/backtest", json=body)
    assert resp.status_code == 200, resp.text
    assert resp.json()["equity_curve"] is None


def test_bad_method_name_is_4xx(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "does_not_exist",
        "data_source": _synthetic_source(),
    }
    resp = client.post("/construct", json=body)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error"] == "ConfigurationError"
    assert "does_not_exist" in detail["detail"]


def test_msr_without_mean_model_is_4xx(client: TestClient, assets: list[str]) -> None:
    body = {
        "assets": assets,
        "method": "msr",
        "data_source": _synthetic_source(),
    }
    resp = client.post("/construct", json=body)
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "ConfigurationError"


def test_empty_assets_is_422(client: TestClient) -> None:
    resp = client.post("/construct", json={"assets": [], "method": "erc"})
    # Pydantic request validation (min_length) -> FastAPI 422.
    assert resp.status_code == 422
