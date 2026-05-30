"""Portfolio Risk Budgeting system.

A Python toolkit that constructs portfolios by *risk budget* rather than by
capital weight: given a target allocation of risk across assets, it estimates a
risk model, solves for weights whose risk contributions match the budget,
backtests the strategy, and reports performance and risk-contribution analytics.

The :mod:`riskbudget.core` subpackage holds the shared, frozen contracts (data
types, pipeline protocols, and errors) that the rest of the system builds on.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
