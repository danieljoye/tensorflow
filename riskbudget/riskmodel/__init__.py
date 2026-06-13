"""Risk and return models (BUILD_PLAN §4, §5.2, Agent 3).

Covariance estimators (implementing :class:`~riskbudget.core.interfaces.RiskModel`)
and expected-return estimators (implementing
:class:`~riskbudget.core.interfaces.MeanModel`), plus the universal nearest-PSD fix
(:func:`nearest_psd`) routed through *every* covariance output (BUILD_PLAN §12.3).

Public factory callables are exposed under the §5.2 registry names so Agent 11's
registry can pick them up without importing internals:

- Risk models: ``"sample"``, ``"ewma"``, ``"semicov"``, ``"ledoit_wolf"``,
  ``"oas"``, ``"pca"``.
- Mean models: ``"mean_historical"``, ``"ewma_mean"``, ``"capm"``,
  ``"risk_based"``, ``"black_litterman"``.

The :data:`RISK_MODEL_FACTORIES` / :data:`MEAN_MODEL_FACTORIES` maps and the
:func:`register` hook (BUILD_PLAN §5.2 registration convention) let the integration
agent assemble its registry from one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from riskbudget.riskmodel.black_litterman import (
    BlackLitterman,
    BLInputs,
    BLPosterior,
    Views,
    black_litterman,
)
from riskbudget.riskmodel.ewma import EWMACovariance, ewma_covariance
from riskbudget.riskmodel.factor import PCAFactorModel, pca_factor_model
from riskbudget.riskmodel.psd import is_psd, nearest_psd
from riskbudget.riskmodel.returns_model import (
    CAPMReturns,
    EWMAMeanReturns,
    HistoricalMeanReturns,
    RiskBasedReturns,
    capm,
    ewma_mean,
    mean_historical,
    risk_based,
)
from riskbudget.riskmodel.sample import SampleCovariance, sample_covariance
from riskbudget.riskmodel.semicov import SemiCovariance, semi_covariance
from riskbudget.riskmodel.shrinkage import (
    OAS,
    LedoitWolfConstantCorrelation,
    LedoitWolfConstantVariance,
    ledoit_wolf,
    oas_shrinkage,
)

# §5.2 name -> factory maps. The factory call (no args) yields a default-configured
# estimator; callers may pass keyword overrides documented on each factory.
RISK_MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "sample": sample_covariance,
    "ewma": ewma_covariance,
    "semicov": semi_covariance,
    "ledoit_wolf": ledoit_wolf,
    "oas": oas_shrinkage,
    "pca": pca_factor_model,
}

MEAN_MODEL_FACTORIES: dict[str, Callable[..., Any]] = {
    "mean_historical": mean_historical,
    "ewma_mean": ewma_mean,
    "capm": capm,
    "risk_based": risk_based,
    "black_litterman": black_litterman,
}


def register(registry: Any) -> None:
    """Register this package's factories with the integration registry (§5.2).

    The integration agent (Agent 11) owns the concrete registry object. This hook
    expects it to expose ``register_risk_model(name, factory)`` and
    ``register_mean_model(name, factory)`` methods, and falls back to a generic
    ``register(kind, name, factory)`` if those are not present. Unknown registry
    shapes raise ``AttributeError`` (surfaced to the integrator), never silently
    no-op.
    """
    for name, factory in RISK_MODEL_FACTORIES.items():
        if hasattr(registry, "register_risk_model"):
            registry.register_risk_model(name, factory)
        else:
            registry.register("risk_model", name, factory)
    for name, factory in MEAN_MODEL_FACTORIES.items():
        if hasattr(registry, "register_mean_model"):
            registry.register_mean_model(name, factory)
        else:
            registry.register("mean_model", name, factory)


__all__ = [
    "BLInputs",
    "BLPosterior",
    "BlackLitterman",
    "CAPMReturns",
    "EWMACovariance",
    "EWMAMeanReturns",
    "HistoricalMeanReturns",
    "LedoitWolfConstantCorrelation",
    "LedoitWolfConstantVariance",
    "MEAN_MODEL_FACTORIES",
    "OAS",
    "PCAFactorModel",
    "RISK_MODEL_FACTORIES",
    "RiskBasedReturns",
    "SampleCovariance",
    "SemiCovariance",
    "Views",
    "black_litterman",
    "capm",
    "ewma_covariance",
    "ewma_mean",
    "is_psd",
    "ledoit_wolf",
    "mean_historical",
    "nearest_psd",
    "oas_shrinkage",
    "pca_factor_model",
    "register",
    "risk_based",
    "sample_covariance",
    "semi_covariance",
]
