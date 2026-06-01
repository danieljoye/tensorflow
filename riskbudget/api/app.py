"""FastAPI application for the risk-budgeting system (Agent 7, BUILD_PLAN §5.2, §7).

Endpoints
---------
- ``GET  /health``    — liveness + version.
- ``GET  /methods``   — the registry's available methods / models / sources, so a
  UI (or a client) can populate dropdowns and validate a request before sending.
- ``POST /construct`` — a point-in-time solve: assets + budget + risk model +
  method + constraints → weights + risk contributions + volatility.
- ``POST /backtest``  — the above plus a date range + schedule → metrics and an
  optional equity curve / weights series.

Every handler turns its request body into a :class:`~riskbudget.spec.StrategySpec`
and drives the *one* shared wiring path (``riskbudget.construct`` /
``riskbudget.backtest``) — it never re-implements method dispatch. Library errors
are mapped to clean HTTP status codes via the error taxonomy (§3.1):
``ConfigurationError`` → 400, ``ValidationError`` → 422, ``OptimizationError`` /
``RiskModelError`` / ``DataError`` / ``BacktestError`` → 422, and any other
``RiskBudgetError`` → 500.

``fastapi`` is imported here (lazily, behind :func:`create_app`) so that importing
``riskbudget`` never pulls the web stack (BUILD_PLAN §3.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

import riskbudget
from riskbudget import REGISTRY
from riskbudget.analytics import summary_stats
from riskbudget.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    CatalogResponse,
    ConstructRequest,
    ConstructResponse,
    HealthResponse,
    MethodInfo,
)
from riskbudget.core.errors import (
    BacktestError,
    ConfigurationError,
    DataError,
    OptimizationError,
    RiskBudgetError,
    RiskModelError,
    ValidationError,
)
from riskbudget.core.types import BacktestResult, Portfolio

if TYPE_CHECKING:
    from fastapi import FastAPI

# Map each error class in the taxonomy (§3.1) to an HTTP status code.
_STATUS_BY_ERROR: list[tuple[type[RiskBudgetError], int]] = [
    (ConfigurationError, 400),
    (ValidationError, 422),
    (OptimizationError, 422),
    (RiskModelError, 422),
    (DataError, 422),
    (BacktestError, 422),
]


def _status_for(exc: RiskBudgetError) -> int:
    """Resolve the HTTP status code for a library error (defaults to 500)."""
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return status
    return 500


def _index_to_str(index: pd.Index) -> list[str]:
    """Stringify a (possibly datetime) index for JSON-friendly keys."""
    if isinstance(index, pd.DatetimeIndex):
        return [ts.strftime("%Y-%m-%d") for ts in index]
    return [str(v) for v in index]


def _construct_payload(spec_request: ConstructRequest) -> ConstructResponse:
    """Run a construct request through the library and shape the response."""
    spec = spec_request.to_spec()
    prices = riskbudget.fetch_prices(spec)
    portfolio: Portfolio = riskbudget.construct(spec, prices=prices)

    returns = prices.to_returns(method=spec.return_method).select(spec.assets)
    cov = spec.build_risk_model().estimate(returns)

    risk_contributions = portfolio.risk_contributions(cov, spec.assets)
    volatility = portfolio.volatility(cov, spec.assets)

    return ConstructResponse(
        name=spec.name,
        method=spec.method,
        assets=list(spec.assets),
        weights={a: float(portfolio.weights[a]) for a in spec.assets},
        risk_contributions={a: float(risk_contributions[a]) for a in spec.assets},
        volatility=float(volatility),
        leverage=float(portfolio.leverage),
        net_exposure=float(portfolio.net_exposure),
    )


def _backtest_payload(spec_request: BacktestRequest) -> BacktestResponse:
    """Run a backtest request through the library and shape the response."""
    spec = spec_request.to_spec()
    result: BacktestResult = riskbudget.backtest(spec)

    metrics = _result_metrics(result, spec)

    equity_curve: dict[str, float] | None = None
    if spec_request.include_equity_curve:
        curve = result.equity_curve.astype(float)
        equity_curve = dict(zip(_index_to_str(curve.index), curve.tolist(), strict=True))

    weights: dict[str, dict[str, float]] | None = None
    if spec_request.include_weights:
        frame = result.weights.astype(float)
        weights = {
            row_key: {str(c): float(v) for c, v in row.items()}
            for row_key, row in zip(
                _index_to_str(frame.index), frame.to_dict(orient="records"), strict=True
            )
        }

    start = result.start
    end = result.end
    return BacktestResponse(
        name=spec.name,
        method=spec.method,
        assets=list(result.assets),
        metrics=metrics,
        start=None if start is None else str(start),
        end=None if end is None else str(end),
        n_periods=int(result.returns.shape[0]) if result.returns is not None else 0,
        equity_curve=equity_curve,
        weights=weights,
    )


def _result_metrics(result: BacktestResult, spec: riskbudget.StrategySpec) -> dict[str, float]:
    """Headline metrics from a backtest: ``result.metrics`` plus summary stats."""
    metrics: dict[str, float] = {str(k): float(v) for k, v in result.metrics.items()}
    if result.returns is not None and result.returns.shape[0] > 1:
        table = summary_stats(
            result.returns.rename(spec.name),
            risk_free_rate=spec.risk_free_rate,
            periods_per_year=spec.periods_per_year,
        )
        row = table.loc[spec.name].astype(float).to_dict()
        for column, value in row.items():
            metrics.setdefault(str(column), float(value))
    return metrics


def _catalog() -> CatalogResponse:
    """Snapshot the registry's available names for the discovery endpoint."""
    methods = [
        MethodInfo(
            name=name,
            requires_mu=REGISTRY.requires_mu(name),
            is_dynamic=REGISTRY.is_allocator_method(name),
        )
        for name in REGISTRY.methods()
    ]
    return CatalogResponse(
        methods=methods,
        risk_models=REGISTRY.risk_models(),
        mean_models=REGISTRY.mean_models(),
        allocators=REGISTRY.allocators(),
        data_sources=REGISTRY.data_sources(),
        cost_models=REGISTRY.cost_models(),
    )


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Imported lazily so ``import riskbudget`` stays free of FastAPI. The handlers
    catch the whole :class:`RiskBudgetError` family and translate it into clean
    4xx/5xx responses via :class:`fastapi.HTTPException`.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="riskbudget API",
        version=riskbudget.__version__,
        summary="Construct and backtest risk-budgeted portfolios.",
    )

    def _handle(exc: RiskBudgetError) -> HTTPException:
        return HTTPException(
            status_code=_status_for(exc),
            detail={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Liveness probe."""
        return HealthResponse(status="ok", version=riskbudget.__version__)

    @app.get("/methods", response_model=CatalogResponse, tags=["meta"])
    def methods() -> CatalogResponse:
        """List the available methods / risk models / mean models / data sources."""
        return _catalog()

    @app.post("/construct", response_model=ConstructResponse, tags=["portfolio"])
    def construct(request: ConstructRequest) -> ConstructResponse:
        """Solve a single portfolio and return weights + risk contributions."""
        try:
            return _construct_payload(request)
        except RiskBudgetError as exc:
            raise _handle(exc) from exc

    @app.post("/backtest", response_model=BacktestResponse, tags=["portfolio"])
    def backtest(request: BacktestRequest) -> BacktestResponse:
        """Run the walk-forward backtest and return metrics + optional curve."""
        try:
            return _backtest_payload(request)
        except RiskBudgetError as exc:
            raise _handle(exc) from exc

    return app


__all__ = ["create_app"]
