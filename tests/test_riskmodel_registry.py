"""Registry / factory convention tests for the riskmodel package (BUILD_PLAN §5.2).

Verifies the §5.2 factory names resolve, the ``register`` hook works with both
registry shapes, and the market-implied ``δ`` + extra error paths behave.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import RiskModelError
from riskbudget.core.interfaces import MeanModel, RiskModel
from riskbudget.core.types import ReturnMatrix
from riskbudget.riskmodel import (
    MEAN_MODEL_FACTORIES,
    RISK_MODEL_FACTORIES,
    register,
)
from riskbudget.riskmodel.black_litterman import BlackLitterman, BLInputs, Views


def test_expected_registry_names_present() -> None:
    assert set(RISK_MODEL_FACTORIES) == {
        "sample",
        "ewma",
        "semicov",
        "ledoit_wolf",
        "oas",
        "pca",
    }
    assert set(MEAN_MODEL_FACTORIES) == {
        "mean_historical",
        "ewma_mean",
        "capm",
        "risk_based",
        "black_litterman",
    }


def test_factories_produce_protocol_instances() -> None:
    for factory in RISK_MODEL_FACTORIES.values():
        assert isinstance(factory(), RiskModel)
    for factory in MEAN_MODEL_FACTORIES.values():
        assert isinstance(factory(), MeanModel)


class _TypedRegistry:
    def __init__(self) -> None:
        self.risk: dict[str, object] = {}
        self.mean: dict[str, object] = {}

    def register_risk_model(self, name: str, factory: object) -> None:
        self.risk[name] = factory

    def register_mean_model(self, name: str, factory: object) -> None:
        self.mean[name] = factory


class _GenericRegistry:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], object] = {}

    def register(self, kind: str, name: str, factory: object) -> None:
        self.items[(kind, name)] = factory


def test_register_hook_typed_registry() -> None:
    reg = _TypedRegistry()
    register(reg)
    assert set(reg.risk) == set(RISK_MODEL_FACTORIES)
    assert set(reg.mean) == set(MEAN_MODEL_FACTORIES)


def test_register_hook_generic_registry() -> None:
    reg = _GenericRegistry()
    register(reg)
    assert ("risk_model", "sample") in reg.items
    assert ("mean_model", "black_litterman") in reg.items


def _returns() -> ReturnMatrix:
    rng = np.random.default_rng(3)
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]])
    draws = rng.multivariate_normal(np.zeros(3), cov, size=400)
    return ReturnMatrix(
        pd.DataFrame(
            draws,
            index=pd.date_range("2020-01-01", periods=400, freq="B"),
            columns=["A", "B", "C"],
        )
    )


def test_bl_market_implied_delta_path() -> None:
    """δ implied from market_return / market_variance when delta is None."""
    inp = BLInputs(market_return=0.08, market_variance=0.04, risk_free_rate=0.0)
    bl = BlackLitterman(views=None, inputs=inp)
    post = bl.estimate_posterior(_returns())
    assert np.isfinite(post.prior).all()


def test_bl_default_delta_fallback() -> None:
    """No delta and no market figures -> δ = 2.5 default, runs cleanly."""
    bl = BlackLitterman(views=None, inputs=BLInputs())
    post = bl.estimate_posterior(_returns())
    assert np.isfinite(post.prior).all()


def test_bl_relative_view_shifts_spread() -> None:
    views = Views.relative("A", "C", 0.10, ["A", "B", "C"], confidence=0.9)
    bl = BlackLitterman(views=views, inputs=BLInputs(delta=2.5))
    post = bl.estimate_posterior(_returns())
    # The A−C spread in the posterior moves toward the 0.10 view.
    prior_spread = post.prior[0] - post.prior[2]
    post_spread = post.mu[0] - post.mu[2]
    assert abs(post_spread - 0.10) < abs(prior_spread - 0.10)


def test_blinputs_validation() -> None:
    with pytest.raises(RiskModelError):
        BLInputs(tau=0.0)
    with pytest.raises(RiskModelError):
        BLInputs(delta=-1.0)
    with pytest.raises(RiskModelError):
        BLInputs(market_caps={"A": -1.0})


def test_bl_market_caps_missing_asset() -> None:
    bl = BlackLitterman(views=None, inputs=BLInputs(market_caps={"A": 1.0}, delta=2.5))
    with pytest.raises(RiskModelError):
        bl.estimate(_returns())


def test_views_require_a_view() -> None:
    with pytest.raises(RiskModelError):
        Views(assets=["A", "B"], P=np.empty((0, 2)), Q=np.empty((0,)))
