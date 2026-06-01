"""FastAPI service for the Portfolio Risk Budgeting system (Agent 7).

The HTTP surface mirrors the library: a request is translated into a
:class:`~riskbudget.spec.StrategySpec` and driven through the shared
``riskbudget.construct`` / ``riskbudget.backtest`` wiring path. The heavy
``fastapi`` / ``uvicorn`` imports live inside :mod:`riskbudget.api.app` so that
``import riskbudget`` stays free of the web stack (BUILD_PLAN §3.1).

Use :func:`riskbudget.api.app.create_app` to build the ASGI application.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    """Lazily expose :func:`create_app` without importing FastAPI eagerly."""
    if name == "create_app":
        from riskbudget.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
