"""Pydantic request/response models for the FastAPI service (BUILD_PLAN §5.2, §7).

These models wrap / mirror :class:`riskbudget.StrategySpec` so the HTTP surface
shares the *one* config object every other surface uses. A request carries the
universe + the strategy knobs; the handler turns it into a ``StrategySpec`` (so
the registry validates every name and the spec validates every field) and drives
:func:`riskbudget.construct` / :func:`riskbudget.backtest`. Responses are plain
JSON-serialisable views of the resulting :class:`~riskbudget.core.types.Portfolio`
and :class:`~riskbudget.core.types.BacktestResult`.

``fastapi`` is imported by :mod:`riskbudget.api.app`, never here — these are pure
pydantic models so the schemas can be imported (and unit-tested) without the web
stack. They are *not* re-exported from the top-level ``riskbudget`` package, which
must stay free of the optional API deps.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from riskbudget.spec import (
    BlackLittermanViewsSpec,
    ConstraintsSpec,
    CostModelSpec,
    DataSourceSpec,
    RebalanceScheduleSpec,
    StrategySpec,
)

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class _SpecRequestBase(BaseModel):
    """Shared body fields that mirror the knobs of a :class:`StrategySpec`.

    Every field maps one-to-one onto a ``StrategySpec`` attribute; the nested
    config objects (data source, constraints, cost model, BL views) reuse the spec
    models directly so the request and the spec never drift. :meth:`to_spec`
    assembles the spec, which validates names against :data:`riskbudget.REGISTRY`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "strategy"
    assets: list[str] = Field(min_length=1, description="Universe of asset ids.")

    data_source: DataSourceSpec = Field(default_factory=DataSourceSpec)

    risk_model: str = "sample"
    risk_model_params: dict[str, Any] = Field(default_factory=dict)

    mean_model: str | None = None
    mean_model_params: dict[str, Any] = Field(default_factory=dict)
    black_litterman: BlackLittermanViewsSpec | None = None

    method: str = "erc"
    method_params: dict[str, Any] = Field(default_factory=dict)

    budget: dict[str, float] | None = Field(
        default=None,
        description="Relative risk budget per asset; None/empty => equal (ERC).",
    )

    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)
    cost_model: CostModelSpec = Field(default_factory=CostModelSpec)

    return_method: str = "simple"
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    seed: int = 0

    start: date | None = None
    end: date | None = None

    def _spec_fields(self) -> dict[str, Any]:
        """The subset of fields common to every spec-backed request."""
        return {
            "name": self.name,
            "assets": list(self.assets),
            "data_source": self.data_source,
            "risk_model": self.risk_model,
            "risk_model_params": dict(self.risk_model_params),
            "mean_model": self.mean_model,
            "mean_model_params": dict(self.mean_model_params),
            "black_litterman": self.black_litterman,
            "method": self.method,
            "method_params": dict(self.method_params),
            "budget": dict(self.budget) if self.budget else None,
            "constraints": self.constraints,
            "cost_model": self.cost_model,
            "return_method": self.return_method,
            "periods_per_year": self.periods_per_year,
            "risk_free_rate": self.risk_free_rate,
            "seed": self.seed,
            "start": self.start,
            "end": self.end,
        }


class ConstructRequest(_SpecRequestBase):
    """Body of ``POST /construct`` — a point-in-time solve over the data window."""

    def to_spec(self) -> StrategySpec:
        """Build the validated :class:`StrategySpec` (``ConfigurationError`` on bad input)."""
        return StrategySpec.model_validate(self._spec_fields())


class BacktestRequest(_SpecRequestBase):
    """Body of ``POST /backtest`` — adds the rebalance schedule and curve toggle."""

    schedule: RebalanceScheduleSpec = Field(default_factory=RebalanceScheduleSpec)
    include_equity_curve: bool = Field(
        default=True,
        description="Return the (date -> value) equity curve in the response.",
    )
    include_weights: bool = Field(
        default=False,
        description="Return the per-rebalance weights frame in the response.",
    )

    def to_spec(self) -> StrategySpec:
        """Build the validated :class:`StrategySpec` (``ConfigurationError`` on bad input)."""
        fields = self._spec_fields()
        fields["schedule"] = self.schedule
        return StrategySpec.model_validate(fields)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ConstructResponse(BaseModel):
    """Weights + risk decomposition for a single constructed portfolio."""

    model_config = ConfigDict(extra="forbid")

    name: str
    method: str
    assets: list[str]
    weights: dict[str, float]
    risk_contributions: dict[str, float]
    volatility: float
    leverage: float
    net_exposure: float


class BacktestResponse(BaseModel):
    """Backtest metrics plus optional equity curve / weights series."""

    model_config = ConfigDict(extra="forbid")

    name: str
    method: str
    assets: list[str]
    metrics: dict[str, float]
    start: str | None = None
    end: str | None = None
    n_periods: int
    equity_curve: dict[str, float] | None = None
    weights: dict[str, dict[str, float]] | None = None


class MethodInfo(BaseModel):
    """One method's registry metadata for the discovery endpoint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    requires_mu: bool
    is_dynamic: bool


class CatalogResponse(BaseModel):
    """The available registry names (drives UI dropdowns and validation)."""

    model_config = ConfigDict(extra="forbid")

    methods: list[MethodInfo]
    risk_models: list[str]
    mean_models: list[str]
    allocators: list[str]
    data_sources: list[str]
    cost_models: list[str]


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    version: str


class ErrorResponse(BaseModel):
    """Uniform error envelope for mapped library exceptions."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str


__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "CatalogResponse",
    "ConstructRequest",
    "ConstructResponse",
    "ErrorResponse",
    "HealthResponse",
    "MethodInfo",
]
