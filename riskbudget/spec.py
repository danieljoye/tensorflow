"""``StrategySpec`` — the single config object that defines a complete run.

Every surface (the API, CLI, dashboard, :func:`riskbudget.compare`, and the
``construct``/``backtest`` helpers) accepts a :class:`StrategySpec` and nothing
takes loose kwargs (BUILD_PLAN §5.2). The spec is a pydantic v2 model so it
validates itself on construction and round-trips to/from JSON; validation
failures and unknown method/model names raise
:class:`~riskbudget.core.errors.ConfigurationError`.

The spec carries *names* (resolved against :data:`riskbudget.registry.REGISTRY`)
plus the configuration each name needs, and provides ``build_*`` methods that
turn those names into the live ``core`` objects the backtester consumes:

- :meth:`StrategySpec.build_data_source` → a :class:`DataSource`.
- :meth:`StrategySpec.build_risk_model` → a :class:`RiskModel`.
- :meth:`StrategySpec.build_mean_model` → a :class:`MeanModel` or ``None``.
- :meth:`StrategySpec.build_method` → a ``PortfolioConstructor`` / ``Optimizer``.
- :meth:`StrategySpec.build_budget` → a :class:`RiskBudget`.
- :meth:`StrategySpec.build_constraints` / :meth:`build_schedule` /
  :meth:`build_cost_model`.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from riskbudget.core.errors import ConfigurationError
from riskbudget.core.interfaces import Constraints, RebalanceSchedule
from riskbudget.core.types import RiskBudget
from riskbudget.registry import REGISTRY, Registry

# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]


class BlackLittermanViewsSpec(BaseModel):
    """Black-Litterman views + market inputs (BUILD_PLAN §12.7).

    Only used when ``mean_model == "black_litterman"``. ``absolute_views`` maps an
    asset to its view return ``Q``; ``confidences`` (optional) maps an asset to an
    Idzorek confidence in ``(0, 1]``. ``relative_views`` expresses
    "outperformer beats underperformer by spread". Market inputs feed the
    reverse-optimization prior.
    """

    model_config = ConfigDict(extra="forbid")

    absolute_views: dict[str, float] = Field(default_factory=dict)
    relative_views: list[tuple[str, str, float]] = Field(default_factory=list)
    confidences: dict[str, float] = Field(default_factory=dict)
    market_caps: dict[str, float] | None = None
    tau: float = Field(default=0.05, gt=0)
    delta: float | None = None
    risk_free_rate: float = 0.0
    market_return: float | None = None
    market_variance: float | None = None


class DataSourceSpec(BaseModel):
    """Which :class:`DataSource` to use and how to configure it.

    ``name`` resolves against the registry (``synthetic``, ``csv``, ``tiingo``);
    ``params`` are forwarded as keyword arguments to that factory. For the
    ``synthetic`` source the caller typically supplies a ``cov`` (or the factor
    triple) and a ``seed`` through ``params``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "synthetic"
    params: dict[str, Any] = Field(default_factory=dict)


class ConstraintsSpec(BaseModel):
    """Feasibility limits mirroring :class:`~riskbudget.core.interfaces.Constraints`."""

    model_config = ConfigDict(extra="forbid")

    long_only: bool = True
    leverage: float | None = 1.0
    min_weight: float | None = None
    max_weight: float | None = None
    group_caps: dict[str, float] = Field(default_factory=dict)
    groups: dict[str, str] = Field(default_factory=dict)
    max_turnover: float | None = None

    def to_constraints(self) -> Constraints:
        """Build the core :class:`Constraints` (re-raising as ``ConfigurationError``)."""
        try:
            return Constraints(
                long_only=self.long_only,
                leverage=self.leverage,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                group_caps=dict(self.group_caps),
                groups=dict(self.groups),
                max_turnover=self.max_turnover,
            )
        except ConfigurationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"Invalid constraints: {exc}") from exc


RebalanceFrequencyLit = Literal["none", "daily", "weekly", "monthly", "quarterly", "annual"]


class RebalanceScheduleSpec(BaseModel):
    """When the backtester re-solves and how far it may look back."""

    model_config = ConfigDict(extra="forbid")

    frequency: RebalanceFrequencyLit = "monthly"
    lookback: int | None = Field(default=None, gt=0)
    min_lookback: int | None = Field(default=None, gt=0)
    start: date | None = None
    end: date | None = None

    def to_schedule(self) -> RebalanceSchedule:
        """Build the core :class:`RebalanceSchedule` (re-raising as ``ConfigurationError``)."""
        try:
            return RebalanceSchedule(
                frequency=self.frequency,
                lookback=self.lookback,
                min_lookback=self.min_lookback,
                start=self.start,
                end=self.end,
            )
        except ConfigurationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"Invalid rebalance schedule: {exc}") from exc


class CostModelSpec(BaseModel):
    """Transaction-cost model: ``proportional`` (bps on turnover) or ``none``."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["proportional", "none"] = "none"
    bps: float = Field(default=0.0, ge=0)
    one_way: bool = True


# ---------------------------------------------------------------------------
# StrategySpec
# ---------------------------------------------------------------------------


class StrategySpec(BaseModel):
    """The complete definition of one strategy run (BUILD_PLAN §5.2).

    Carries the universe, the data source, the risk/mean models, the construction
    ``method``, the risk ``budget``, constraints, the rebalance schedule, the cost
    model, and a ``seed``. Validated on construction; unknown names or malformed
    fields raise :class:`~riskbudget.core.errors.ConfigurationError`.

    The ``budget`` field is optional and only meaningful for the risk-budget path
    (``erc`` / ``risk_budget`` / ``factor_risk_budget``): ``None`` (or an empty
    mapping) means the equal-risk-contribution (ERC) budget ``bᵢ = 1/N`` over the
    universe. A non-empty mapping of *relative* positive weights is normalized via
    :meth:`RiskBudget.from_weights`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "strategy"
    assets: list[str] = Field(min_length=1)

    data_source: DataSourceSpec = Field(default_factory=DataSourceSpec)

    risk_model: str = "sample"
    risk_model_params: dict[str, Any] = Field(default_factory=dict)

    mean_model: str | None = None
    mean_model_params: dict[str, Any] = Field(default_factory=dict)
    black_litterman: BlackLittermanViewsSpec | None = None

    method: str = "erc"
    method_params: dict[str, Any] = Field(default_factory=dict)

    budget: dict[str, float] | None = None

    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)
    schedule: RebalanceScheduleSpec = Field(default_factory=RebalanceScheduleSpec)
    cost_model: CostModelSpec = Field(default_factory=CostModelSpec)

    return_method: Literal["simple", "log"] = "simple"
    periods_per_year: PositiveInt = 252
    risk_free_rate: float = 0.0
    seed: int = 0

    # Optional volatility-targeting overlay: when set, the constructed book is
    # re-levered each rebalance to this annualized volatility (ex-ante, capped at
    # ``target_vol_max_leverage``). None disables it (fully-invested as solved).
    target_volatility: PositiveFloat | None = None
    target_vol_max_leverage: PositiveFloat = 3.0

    start: date | None = None
    end: date | None = None

    # -- field/model validation ---------------------------------------------

    @field_validator("assets")
    @classmethod
    def _no_duplicate_assets(cls, value: list[str]) -> list[str]:
        cleaned = [str(a) for a in value]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("StrategySpec.assets contains duplicate ids.")
        return cleaned

    @model_validator(mode="after")
    def _validate_names(self) -> StrategySpec:
        """Resolve every referenced name against the registry up front."""
        reg = self._registry
        # ``method`` may name a cross-sectional constructor or a dynamic allocator.
        if not reg.is_allocator_method(self.method):
            reg.method(self.method)  # raises ConfigurationError on unknown
        reg.risk_model(self.risk_model)
        reg.data_source(self.data_source.name)
        if self.mean_model is not None:
            reg.mean_model(self.mean_model)
        if self.cost_model.name not in {"proportional", "none"}:
            raise ConfigurationError(  # pragma: no cover - Literal guards this
                f"Unknown cost model {self.cost_model.name!r}."
            )
        # Methods needing μ must have a mean model.
        if reg.requires_mu(self.method) and self.mean_model is None:
            raise ConfigurationError(
                f"Method {self.method!r} needs expected returns; set a mean_model "
                "(e.g. 'mean_historical', 'risk_based', 'capm', or 'black_litterman')."
            )
        if self.black_litterman is not None and self.mean_model != "black_litterman":
            raise ConfigurationError(
                "black_litterman views supplied but mean_model is not 'black_litterman'."
            )
        return self

    @property
    def _registry(self) -> Registry:
        return REGISTRY

    # -- builders (spec → live core objects) --------------------------------

    def build_budget(self) -> RiskBudget:
        """Build the :class:`RiskBudget` over the universe (ERC default)."""
        try:
            if not self.budget:
                return RiskBudget.equal(self.assets)
            extra = set(self.budget) - set(self.assets)
            if extra:
                raise ConfigurationError(f"Budget references assets not in the universe: {extra}.")
            relative = {a: self.budget[a] for a in self.assets if a in self.budget}
            return RiskBudget.from_weights(relative)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(f"Invalid risk budget: {exc}") from exc

    def build_constraints(self) -> Constraints:
        """Build the core :class:`Constraints`."""
        return self.constraints.to_constraints()

    def build_schedule(self) -> RebalanceSchedule:
        """Build the core :class:`RebalanceSchedule`."""
        return self.schedule.to_schedule()

    def build_data_source(self) -> Any:
        """Instantiate the configured :class:`DataSource`."""
        factory = self._registry.data_source(self.data_source.name)
        params = dict(self.data_source.params)
        # The synthetic source is seeded from the spec seed unless overridden.
        if self.data_source.name == "synthetic":
            params.setdefault("seed", self.seed)
            params.setdefault("assets", list(self.assets))
        try:
            return factory(**params)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not build data source {self.data_source.name!r}: {exc}"
            ) from exc

    def _propagate_periods_per_year(self, factory: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Inject the run's ``periods_per_year`` into ``params`` when the factory
        accepts it and the user hasn't pinned it explicitly.

        Keeps every estimator annualized to the *data* frequency rather than a
        hardcoded 252 — covariance and expected returns must use the same factor
        or μ and Σ are on different scales.
        """
        import inspect

        if "periods_per_year" in params:
            return params
        try:
            sig = inspect.signature(factory)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return params
        if "periods_per_year" in sig.parameters:
            params = {**params, "periods_per_year": self.periods_per_year}
        return params

    def build_risk_model(self) -> Any:
        """Instantiate the configured :class:`RiskModel`.

        Propagates the run's ``periods_per_year`` to factories that accept it (so
        the covariance is annualized to the *data* frequency, not a hardcoded 252),
        unless the user pinned it explicitly in ``risk_model_params``.
        """
        factory = self._registry.risk_model(self.risk_model)
        params = self._propagate_periods_per_year(factory, dict(self.risk_model_params))
        try:
            return factory(**params)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not build risk model {self.risk_model!r}: {exc}"
            ) from exc

    def build_mean_model(self) -> Any | None:
        """Instantiate the configured :class:`MeanModel`, or ``None`` if unset.

        Propagates the run's ``periods_per_year`` exactly like
        :meth:`build_risk_model` — μ and Σ must annualize by the same factor.
        """
        if self.mean_model is None:
            return None
        factory = self._registry.mean_model(self.mean_model)
        params = dict(self.mean_model_params)
        if self.mean_model == "black_litterman":
            params = {**self._black_litterman_kwargs(), **params}
        params = self._propagate_periods_per_year(factory, params)
        try:
            return factory(**params)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Could not build mean model {self.mean_model!r}: {exc}"
            ) from exc

    def build_method(self) -> Any:
        """Instantiate the configured portfolio constructor / optimizer."""
        factory = self._registry.method(self.method)
        try:
            return factory(**self.method_params)
        except Exception as exc:
            raise ConfigurationError(f"Could not build method {self.method!r}: {exc}") from exc

    def build_cost_model(self) -> Any:
        """Instantiate the configured cost model."""
        factory = self._registry.cost_model(self.cost_model.name)
        if self.cost_model.name == "proportional":
            return factory(self.cost_model.bps, one_way=self.cost_model.one_way)
        return factory()

    def _black_litterman_kwargs(self) -> dict[str, Any]:
        """Translate the BL views spec into ``black_litterman`` factory kwargs."""
        from riskbudget.riskmodel import Views
        from riskbudget.riskmodel.black_litterman import BLInputs

        bl = self.black_litterman
        kwargs: dict[str, Any] = {}
        if bl is None:
            return kwargs
        try:
            inputs = BLInputs(
                market_caps=dict(bl.market_caps) if bl.market_caps else None,
                tau=bl.tau,
                delta=bl.delta,
                risk_free_rate=bl.risk_free_rate,
                market_return=bl.market_return,
                market_variance=bl.market_variance,
            )
        except Exception as exc:
            raise ConfigurationError(f"Invalid Black-Litterman inputs: {exc}") from exc
        kwargs["inputs"] = inputs

        views = None
        if bl.absolute_views:
            confidences = dict(bl.confidences) if bl.confidences else None
            try:
                views = Views.absolute(
                    dict(bl.absolute_views), self.assets, confidences=confidences
                )
            except Exception as exc:
                raise ConfigurationError(f"Invalid absolute Black-Litterman views: {exc}") from exc
        elif bl.relative_views:
            if len(bl.relative_views) != 1:
                raise ConfigurationError(
                    "Only a single relative Black-Litterman view is supported via the spec; "
                    "build a Views object directly for multiple relative views."
                )
            out, under, spread = bl.relative_views[0]
            try:
                views = Views.relative(out, under, spread, self.assets)
            except Exception as exc:
                raise ConfigurationError(f"Invalid relative Black-Litterman view: {exc}") from exc
        if views is not None:
            kwargs["views"] = views
        return kwargs

    # -- convenience --------------------------------------------------------

    @property
    def is_dynamic(self) -> bool:
        """True if ``method`` is a dynamic (temporal) allocator rather than a constructor."""
        return self._registry.is_allocator_method(self.method)


def strategy_spec_from_dict(data: dict[str, Any]) -> StrategySpec:
    """Build a :class:`StrategySpec` from a plain dict (``ConfigurationError`` on bad input)."""
    try:
        return StrategySpec.model_validate(data)
    except ConfigurationError:
        raise
    except PydanticValidationError as exc:
        raise ConfigurationError(f"Invalid StrategySpec: {exc}") from exc


__all__ = [
    "BlackLittermanViewsSpec",
    "ConstraintsSpec",
    "CostModelSpec",
    "DataSourceSpec",
    "RebalanceScheduleSpec",
    "StrategySpec",
    "strategy_spec_from_dict",
]
