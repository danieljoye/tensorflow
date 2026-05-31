# Agent 11 — Integration, Glue, Examples & Docs

**Wave:** 3 (after 7; sees every implementation). **Owns:** `riskbudget/spec.py`,
`riskbudget/registry.py`, `riskbudget/compare.py`, `riskbudget/__init__.py`,
`examples/`, and the cross-cutting integration tests. **Depends on:** all agents.

## Scope
Tie the system together so it is driven by ONE config object through ONE registry,
and prove it works end to end (BUILD_PLAN §3.1, §5.2).
1. `spec.py` — **`StrategySpec`** (pydantic v2): the complete definition of a run —
   universe/assets, data-source choice, `risk_model`, `mean_model` (incl.
   `black_litterman` + its views), `method`, `budget`, `Constraints`,
   `RebalanceSchedule`, cost model, and `seed`. Validate it (raise `ConfigError` on
   bad/unknown fields). This is the single object every surface accepts.
2. `registry.py` — string→factory maps for risk models, mean models, portfolio
   constructors, and allocators. Import each implementation's documented public
   factories (per the §5.2 naming convention the Wave 1–2 agents exposed) and register
   them in one place. Unknown name → `ConfigError` listing valid options. Do NOT
   re-implement methods; only wire existing ones.
3. `compare.py` — given one or more `StrategySpec`s and a dataset, run the panel of
   methods through the backtester (Agent 5) and return a `summary_stats` (Agent 6)
   comparison table, head-to-head.
4. `__init__.py` — curate the stable public API: re-export the core types plus a thin
   `construct(spec)` / `backtest(spec)` / `compare(specs)` convenience layer. Importing
   `riskbudget` must NOT pull optional/dev deps.
5. `examples/` — at least one runnable **golden-path** script: synthetic data →
   construct (ERC + one benchmark) → backtest → analytics → render a report — fully
   offline, deterministic via a fixed seed.
6. Integration tests in `tests/` that exercise the whole pipeline through `StrategySpec`
   for several methods, asserting the §7 acceptance criteria hold end to end.

## Interfaces to honor
Consume only the public factories/contracts of the other packages; do not reach into
internals. Honor §3.1 conventions and the error taxonomy.

## Deliverables
spec + registry + compare + public API + example + integration tests, with: a
`StrategySpec` round-trips through `construct`/`backtest`/`compare`; the registry
resolves every method name (and raises `ConfigError` on unknown); the golden-path
example runs in CI offline; `import riskbudget` works without dev deps. Reconcile with
Agent 7 (API/dashboard should call the registry rather than duplicate wiring).

## Out of scope
Implementing any model/optimizer/metric yourself — you only assemble and prove. Do not
change `core/`.

## Done when
The full pipeline runs end-to-end from a single `StrategySpec` for every registered
method, the example + integration tests pass offline, and ruff/mypy/pytest are green.
