"""Command-line interface for the risk-budgeting system (Agent 7).

``python -m riskbudget.cli`` exposes ``construct`` and ``backtest`` subcommands
mirroring the API: they read a price panel from CSV (or use the synthetic
backbone), build a :class:`~riskbudget.spec.StrategySpec`, drive the shared
``riskbudget.construct`` / ``riskbudget.backtest`` wiring, and write JSON.

The CLI uses only the standard library plus the (already-required) numeric stack;
no web deps are imported, so it stays usable in a headless environment.
"""

from __future__ import annotations

from riskbudget.cli.main import main

__all__ = ["main"]
