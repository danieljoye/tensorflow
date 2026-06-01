"""Dynamic (temporal) risk-budgeting allocators (BUILD_PLAN §2, §11; Agent 8).

The *temporal* sense of risk budgeting: distribute a risk budget over time between
a risky (performance-seeking) and a safe asset, subject to a floor or a
maximum-drawdown constraint. Every allocator implements the
:class:`~riskbudget.core.interfaces.Allocator` protocol
(``allocate(risky, safe, params) -> BacktestResult``).

Modules
-------
- :mod:`~riskbudget.dynamic.cppi` — Constant Proportion Portfolio Insurance with
  fixed- and drawdown-floor variants (EDHEC ``run_cppi``).
- :mod:`~riskbudget.dynamic.allocators` — fixed-mix, glidepath (calendar +
  state-dependent), floor, and drawdown allocators, plus the :func:`bt_mix`
  scenario driver.
- :mod:`~riskbudget.dynamic.fund_separation` — three-fund PSP / liability-hedging /
  safe-asset allocator with a funding-ratio floor (Martellini–Milhau 2012).

Factory callables exposed under the BUILD_PLAN §5.2 registry names: ``cppi``,
``fixed_mix``, ``glidepath``, ``floor``, ``drawdown``, ``fund_separation``.
"""

from __future__ import annotations

from riskbudget.dynamic.allocators import (
    DrawdownAllocator,
    FixedMixAllocator,
    FloorAllocator,
    GlidepathAllocator,
    bt_mix,
    drawdown,
    fixed_mix,
    floor,
    glidepath,
)
from riskbudget.dynamic.cppi import (
    CPPI,
    CPPIHistory,
    cppi,
    run_cppi_path,
)
from riskbudget.dynamic.fund_separation import (
    FundSeparationAllocator,
    FundSeparationHistory,
    fund_separation,
    run_fund_separation_path,
)

__all__ = [
    "CPPI",
    "CPPIHistory",
    "DrawdownAllocator",
    "FixedMixAllocator",
    "FloorAllocator",
    "FundSeparationAllocator",
    "FundSeparationHistory",
    "GlidepathAllocator",
    "bt_mix",
    "cppi",
    "drawdown",
    "fixed_mix",
    "floor",
    "fund_separation",
    "glidepath",
    "run_cppi_path",
    "run_fund_separation_path",
]
