# Agent Prompt Pack

Self-contained prompts for each build agent of the Portfolio Risk Budgeting
system. Each agent reads `docs/BUILD_PLAN.md` first, then its own prompt below.

**Dispatch order**
- Wave 0: Agent 0 (Foundation) — must finish first; it publishes `core/` contracts.
- Wave 1: Agents 1, 2, 3, 4 — run in parallel.
- Wave 2: Agents 5, 6.
- Wave 3: Agent 7.

Every prompt shares this **common preamble** (prepend when dispatching):

> You are a build agent for the Portfolio Risk Budgeting system in this repo.
> First read `docs/BUILD_PLAN.md` in full, then `riskbudget/core/` (types and
> interfaces). Implement **only** the package(s) assigned to you; do not modify
> other agents' internals. Honor the `core/` contracts exactly — if a contract is
> wrong, stop and report it rather than forking it. Deliver implementation +
> pytest tests (>90% coverage of your package) + type hints + docstrings. Your
> code must pass `ruff check`, `ruff format --check`, `mypy`, and `pytest` with no
> network access (use the synthetic data source). When done, summarize what you
> built and any contract friction you hit.

Individual prompts: `agent-0-foundation.md` … `agent-7-api-dashboard.md`.
