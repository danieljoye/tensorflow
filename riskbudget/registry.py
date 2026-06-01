"""The single name→factory registry that wires every implementation together.

This module is the *one place* (BUILD_PLAN §5.2) that imports the public factory
callables exposed by each implementation package and maps a string name to each.
Every surface — :mod:`riskbudget.spec`, :mod:`riskbudget.compare`, the public
``construct``/``backtest`` helpers, and Agent 7's API/CLI/dashboard — resolves
methods and models through *this* registry rather than re-inventing the wiring.

Registration convention (BUILD_PLAN §5.2)
-----------------------------------------
Each implementation package either exposes a ``register(registry)`` hook (which
calls the typed ``register_*`` methods on :class:`Registry`) or simply exposes
``*_FACTORIES`` maps / named factories that this module imports directly. Packages
never edit a shared registry file concurrently; the integration agent assembles it
here.

Categories
----------
- **risk models** — :class:`~riskbudget.core.interfaces.RiskModel` factories
  (``sample``, ``ewma``, ``semicov``, ``ledoit_wolf``, ``oas``, ``pca``).
- **mean models** — :class:`~riskbudget.core.interfaces.MeanModel` factories
  (``mean_historical``, ``ewma_mean``, ``capm``, ``risk_based``,
  ``black_litterman``).
- **methods** — portfolio constructors / optimizers (``erc``, ``risk_budget``,
  ``gmv``, ``msr``, ``efficient_msr``, ``equal_weight``, ``mdp``, ``max_enb``,
  ``factor_risk_budget``, ``hrp``, ``ensemble``).
- **allocators** — dynamic (temporal) allocators (``cppi``, ``fixed_mix``,
  ``glidepath``, ``floor``, ``drawdown``, ``fund_separation``).
- **data sources** — :class:`~riskbudget.core.interfaces.DataSource` factories
  (``synthetic``, ``csv``, ``tiingo``, ``shiller``, ``gold``).
- **backtesters** — ``walkforward``.
- **cost models** — ``proportional``, ``none``.

An unknown name raises :class:`~riskbudget.core.errors.ConfigurationError` listing
the valid options for that category.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# --- implementation factories (imported in ONE place) ----------------------
from riskbudget.backtest import (
    BACKTESTER_FACTORIES,
    COST_MODEL_FACTORIES,
)
from riskbudget.backtest import register as _register_backtest
from riskbudget.clustered import hrp
from riskbudget.core.errors import ConfigurationError
from riskbudget.data import csv_data_source, synthetic_data_source
from riskbudget.diversification import (
    factor_risk_budget,
    max_enb,
    mdp,
)
from riskbudget.dynamic import (
    cppi,
    drawdown,
    fixed_mix,
    floor,
    fund_separation,
    glidepath,
)
from riskbudget.optimize import (
    efficient_msr,
    ensemble,
    equal_weight,
    erc,
    gmv,
    msr,
    risk_budget,
)
from riskbudget.riskmodel import (
    MEAN_MODEL_FACTORIES,
    RISK_MODEL_FACTORIES,
)
from riskbudget.riskmodel import register as _register_riskmodel

Factory = Callable[..., Any]

# Methods (portfolio constructors / optimizers). Each factory returns an object
# implementing ``PortfolioConstructor.construct`` and/or ``Optimizer.solve`` — the
# walk-forward backtester dispatches over whichever it finds.
METHOD_FACTORIES: dict[str, Factory] = {
    "erc": erc,
    "risk_budget": risk_budget,
    "gmv": gmv,
    "msr": msr,
    "efficient_msr": efficient_msr,
    "equal_weight": equal_weight,
    "mdp": mdp,
    "max_enb": max_enb,
    "factor_risk_budget": factor_risk_budget,
    "hrp": hrp,
    "ensemble": ensemble,
}

# Dynamic (temporal) allocators — implement ``Allocator.allocate``.
ALLOCATOR_FACTORIES: dict[str, Factory] = {
    "cppi": cppi,
    "fixed_mix": fixed_mix,
    "glidepath": glidepath,
    "floor": floor,
    "drawdown": drawdown,
    "fund_separation": fund_separation,
}

# Data sources — implement ``DataSource.get_prices``. ``tiingo`` is imported
# lazily (its provider module is optional / network-gated).
DATA_SOURCE_FACTORIES: dict[str, Factory] = {
    "synthetic": synthetic_data_source,
    "csv": csv_data_source,
}

# Method names that require an expected-returns (``mu``) input, so the backtester
# must be given a mean model.
METHODS_REQUIRING_MU: frozenset[str] = frozenset({"msr", "efficient_msr"})


def _tiingo_factory() -> Factory:
    """Lazily import the optional Tiingo provider factory."""
    from riskbudget.data.providers import tiingo_data_source

    return tiingo_data_source


def _shiller_factory() -> Factory:
    """Lazily import the Shiller deep-history provider factory."""
    from riskbudget.data.providers import shiller_data_source

    return shiller_data_source


def _gold_factory() -> Factory:
    """Lazily import the gold deep-history provider factory."""
    from riskbudget.data.providers import gold_data_source

    return gold_data_source


class Registry:
    """A mutable name→factory registry over every implementation category.

    The default, fully-populated instance is :data:`REGISTRY` (and a fresh copy
    is built by :func:`default_registry`). Implementation packages register via
    their ``register(registry)`` hooks, which call the typed ``register_*``
    methods below; categories without a hook are seeded directly from their
    ``*_FACTORIES`` maps.
    """

    def __init__(self) -> None:
        self._risk_models: dict[str, Factory] = {}
        self._mean_models: dict[str, Factory] = {}
        self._methods: dict[str, Factory] = {}
        self._allocators: dict[str, Factory] = {}
        self._data_sources: dict[str, Factory] = {}
        self._backtesters: dict[str, Factory] = {}
        self._cost_models: dict[str, Factory] = {}

    # -- typed registration hooks (called by package ``register`` hooks) -----

    def register_risk_model(self, name: str, factory: Factory) -> None:
        """Register a :class:`RiskModel` factory under ``name``."""
        self._risk_models[name] = factory

    def register_mean_model(self, name: str, factory: Factory) -> None:
        """Register a :class:`MeanModel` factory under ``name``."""
        self._mean_models[name] = factory

    def register_method(self, name: str, factory: Factory) -> None:
        """Register a portfolio-constructor/optimizer factory under ``name``."""
        self._methods[name] = factory

    def register_allocator(self, name: str, factory: Factory) -> None:
        """Register a dynamic :class:`Allocator` factory under ``name``."""
        self._allocators[name] = factory

    def register_data_source(self, name: str, factory: Factory) -> None:
        """Register a :class:`DataSource` factory under ``name``."""
        self._data_sources[name] = factory

    def register_backtester(self, name: str, factory: Factory) -> None:
        """Register a :class:`Backtester` factory under ``name``."""
        self._backtesters[name] = factory

    def register_cost_model(self, name: str, factory: Factory) -> None:
        """Register a cost-model factory under ``name``."""
        self._cost_models[name] = factory

    # -- resolution ----------------------------------------------------------

    @staticmethod
    def _resolve(table: dict[str, Factory], name: str, *, kind: str) -> Factory:
        try:
            return table[name]
        except KeyError:
            valid = ", ".join(sorted(table))
            raise ConfigurationError(f"Unknown {kind} {name!r}. Valid options: {valid}.") from None

    def risk_model(self, name: str) -> Factory:
        """Resolve a risk-model factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._risk_models, name, kind="risk model")

    def mean_model(self, name: str) -> Factory:
        """Resolve a mean-model factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._mean_models, name, kind="mean model")

    def method(self, name: str) -> Factory:
        """Resolve a portfolio-method factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._methods, name, kind="method")

    def allocator(self, name: str) -> Factory:
        """Resolve a dynamic-allocator factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._allocators, name, kind="allocator")

    def data_source(self, name: str) -> Factory:
        """Resolve a data-source factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._data_sources, name, kind="data source")

    def backtester(self, name: str) -> Factory:
        """Resolve a backtester factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._backtesters, name, kind="backtester")

    def cost_model(self, name: str) -> Factory:
        """Resolve a cost-model factory by name (``ConfigurationError`` if unknown)."""
        return self._resolve(self._cost_models, name, kind="cost model")

    # -- introspection -------------------------------------------------------

    def risk_models(self) -> list[str]:
        """Sorted list of registered risk-model names."""
        return sorted(self._risk_models)

    def mean_models(self) -> list[str]:
        """Sorted list of registered mean-model names."""
        return sorted(self._mean_models)

    def methods(self) -> list[str]:
        """Sorted list of registered portfolio-method names."""
        return sorted(self._methods)

    def allocators(self) -> list[str]:
        """Sorted list of registered dynamic-allocator names."""
        return sorted(self._allocators)

    def data_sources(self) -> list[str]:
        """Sorted list of registered data-source names."""
        return sorted(self._data_sources)

    def backtesters(self) -> list[str]:
        """Sorted list of registered backtester names."""
        return sorted(self._backtesters)

    def cost_models(self) -> list[str]:
        """Sorted list of registered cost-model names."""
        return sorted(self._cost_models)

    def is_allocator_method(self, name: str) -> bool:
        """True if ``name`` is a dynamic allocator (vs. a cross-sectional method)."""
        return name in self._allocators

    def requires_mu(self, name: str) -> bool:
        """True if the named method needs an expected-returns (``mu``) input."""
        return name in METHODS_REQUIRING_MU


def default_registry() -> Registry:
    """Build a fresh, fully-populated :class:`Registry`.

    Risk models, mean models, backtesters, and cost models are wired through each
    package's ``register(registry)`` hook (BUILD_PLAN §5.2). Methods, allocators,
    and data sources are seeded from the maps in this module (those packages expose
    named factories rather than a registry hook). The optional Tiingo data source
    is wired lazily so importing the registry never pulls the provider module.
    """
    reg = Registry()

    # Hook-based registration (risk/mean models, backtester, cost models).
    _register_riskmodel(reg)
    _register_backtest(reg)

    # Sanity: the hooks should have populated exactly the documented factories.
    assert set(reg.risk_models()) == set(RISK_MODEL_FACTORIES)
    assert set(reg.mean_models()) == set(MEAN_MODEL_FACTORIES)
    assert set(reg.backtesters()) == set(BACKTESTER_FACTORIES)
    assert set(reg.cost_models()) == set(COST_MODEL_FACTORIES)

    # Direct registration (methods, allocators, data sources).
    for name, factory in METHOD_FACTORIES.items():
        reg.register_method(name, factory)
    for name, factory in ALLOCATOR_FACTORIES.items():
        reg.register_allocator(name, factory)
    for name, factory in DATA_SOURCE_FACTORIES.items():
        reg.register_data_source(name, factory)
    reg.register_data_source("tiingo", _LazyTiingo())
    reg.register_data_source("shiller", _LazyProvider(_shiller_factory))
    reg.register_data_source("gold", _LazyProvider(_gold_factory))

    return reg


class _LazyTiingo:
    """Defers importing the optional Tiingo provider until the factory is called."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _tiingo_factory()(*args, **kwargs)


class _LazyProvider:
    """Defers importing a provider module until the factory is actually called.

    Used to keep ``import riskbudget`` light: the deep-history providers (and any
    network transport they may use) are only imported when their data source is
    constructed.
    """

    def __init__(self, loader: Callable[[], Factory]) -> None:
        self._loader = loader

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._loader()(*args, **kwargs)


#: The process-wide default registry.
REGISTRY: Registry = default_registry()


__all__ = [
    "ALLOCATOR_FACTORIES",
    "DATA_SOURCE_FACTORIES",
    "METHODS_REQUIRING_MU",
    "METHOD_FACTORIES",
    "REGISTRY",
    "Registry",
    "default_registry",
]
