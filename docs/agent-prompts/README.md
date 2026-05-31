# Agent Prompt Pack

Self-contained prompts for each build agent of the Portfolio Risk Budgeting
system. Each agent reads `docs/BUILD_PLAN.md` first, then its own prompt below.

**Dispatch order**
- Wave 0: Agent 0 (Foundation) — ✅ done; published the `core/` contracts.
- Wave 0.5: Agent 0.5 (Core extension) — quick additive contracts (`ExpectedReturns`,
  `MeanModel`, `PortfolioConstructor`, `Allocator`). Must land before Wave 1.
- Wave 1: Agents 1, 2, 3, 4, 9 — run in parallel.
- Wave 2: Agents 5, 6, 8.
- Wave 3: Agents 7, 11 (11 assembles the spec/registry/public API and proves the
  pipeline end-to-end; it should land alongside or just after 7).

Every prompt shares this **common preamble** (prepend when dispatching):

> You are a build agent for the Portfolio Risk Budgeting system in this repo.
> First read `docs/BUILD_PLAN.md` in full — especially §3.1 (binding conventions),
> §5/§5.1/§5.2 (contracts, registry), and §12 (the exact algorithms) — then
> `riskbudget/core/` (types and interfaces). Implement **only** the package(s)
> assigned to you; do not modify other agents' internals. Honor the `core/` contracts
> and §3.1 conventions exactly — if a contract is wrong, stop and report it rather
> than forking it. Expose your public methods as factory callables under the §5.2
> names so the registry can find them. Deliver implementation + tests (unit + at least
> one property-based invariant test; optional skippable cross-check vs. the reference
> lib where one exists) + type hints + docstrings citing the §11/§12 source. **Run all
> gates via `.venv/bin/...`** (the base interpreter lacks the deps): `ruff check`,
> `ruff format --check`, `mypy`, `pytest` — all green, no network (use synthetic data).
> When done, summarize what you built and any contract friction you hit.

Individual prompts: `agent-0-foundation.md`, `agent-0.5-core-extension.md`,
`agent-1-data-research.md` … `agent-7-api-dashboard.md`, `agent-8-dynamic.md`,
`agent-9-diversification.md`, `agent-11-integration.md`. Research provenance for every
method is in `docs/BUILD_PLAN.md` §11 (papers) and §12 (algorithms) — agents should
cite the relevant source in module docstrings. Binding conventions are in §3.1.
