# Agent 8 — Dynamic Allocation (CPPI & Portfolio Insurance)

**Wave:** 2. **Owns:** `riskbudget/dynamic/*`.
**Depends on:** Agent 0 (`core/`), Agent 0.5 (`Allocator`, `AllocatorParams`),
Agent 2's data/simulation interfaces.

## Scope
The *temporal* sense of risk budgeting (BUILD_PLAN §2): dynamically allocate a
risk budget between a risky and a safe asset over time, subject to a floor or
drawdown constraint. Follow the EDHEC course `run_cppi` / allocator formulations.
1. `dynamic/cppi.py` — Constant Proportion Portfolio Insurance: given a risky-asset
   return series, a safe asset (series or constant rate), a multiplier `m`, and a
   floor, run the backtest. At each step: `cushion = (account − floor)/account`,
   risky weight `= clip(m × cushion, 0, 1)` (with optional leverage cap), rebalance,
   accrue. Support a **drawdown-based floor** (floor = `(1 − maxDD) × running peak`).
   Return a `BacktestResult` plus the CPPI-specific history (account, floor,
   cushion, risky weight over time).
2. `dynamic/allocators.py` — pluggable `Allocator`s: `fixedmix_allocator`
   (constant weights), `glidepath_allocator` (linear shift start→end weights;
   support a **state-dependent** variant driven by the funding cushion, not just
   calendar — Martellini–Milhau life-cycle), `floor_allocator` (CPPI-style floor),
   `drawdown_allocator` (peak-drawdown limit), and a `bt_mix` driver that runs an
   allocator over a set of scenarios (e.g. GBM paths from Agent 2's `simulate/`).
3. `dynamic/fund_separation.py` — **three-fund PSP / LHP / safe-asset allocator**
   (Martellini–Milhau, "Dynamic Allocation Decisions in the Presence of Funding
   Ratio Constraints"): given a performance-seeking portfolio, a liability-hedging
   portfolio (or its return stream), and a safe asset, allocate dynamically with the
   risky (PSP) weight = `m × cushion` where `cushion` is distance of the funding
   ratio above its floor. Generalizes CPPI from risky/safe to PSP/LHP/safe. Cite the
   paper in the docstring. (Full ALM — liability PV, duration matching, surplus-risk
   optimization — is roadmap per BUILD_PLAN §10; build only this allocator now.)

All strategies implement the `Allocator` Protocol from Agent 0.5.

## Interfaces to honor
`Allocator.allocate(risky, safe, params) -> BacktestResult` and `AllocatorParams`
from `core/`. Produce a `BacktestResult` consumable by Agent 6's analytics so CPPI
runs report through the same summary-stats/report path.

## Deliverables
CPPI + allocators + fund-separation + scenario driver, with tests: CPPI never
breaches its floor on a monotone-down synthetic path; cushion/multiplier math
matches a hand-worked step; glidepath weights interpolate correctly; drawdown
allocator respects its limit; the three-fund PSP/LHP/safe allocator keeps the
funding ratio above its floor on synthetic paths; `bt_mix` over GBM scenarios
produces sane terminal-wealth distributions.

## Out of scope
Cross-sectional ERC/optimizers (Agent 4), analytics internals (Agent 6),
API/dashboard (Agent 7). Do not change `core/`.

## Done when
All allocators satisfy `Allocator`, CPPI respects its floor, results flow through
Agent 6 analytics, and tests pass offline.
