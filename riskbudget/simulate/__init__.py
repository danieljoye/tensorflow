"""Scenario generation for stress testing and dynamic allocation (BUILD_PLAN §4).

Geometric Brownian Motion (GBM) Monte-Carlo per the EDHEC course (BUILD_PLAN
§11). Exposes:

- :func:`~riskbudget.simulate.gbm.gbm` — simulate price (or gross-return) paths.
- :func:`~riskbudget.simulate.gbm.gbm_price_data` — the same as a
  :class:`~riskbudget.core.types.PriceData` panel.
- :func:`~riskbudget.simulate.gbm.terminal_values` /
  :func:`~riskbudget.simulate.gbm.terminal_stats` — terminal-wealth summaries
  (mean/median/percentiles, floor-breach probability) feeding the
  dynamic-allocation agent (8) and stress testing.

Every draw takes an explicit seed (BUILD_PLAN §3.1 determinism).
"""

from __future__ import annotations

from riskbudget.simulate.gbm import (
    gbm,
    gbm_price_data,
    terminal_stats,
    terminal_values,
)

__all__ = [
    "gbm",
    "gbm_price_data",
    "terminal_stats",
    "terminal_values",
]
