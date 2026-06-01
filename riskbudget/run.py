"""The thin execution layer: turn a :class:`StrategySpec` into results.

These helpers are the *one* wiring path shared by the public API
(:func:`riskbudget.construct` / :func:`riskbudget.backtest`),
:func:`riskbudget.compare.compare`, and Agent 7's API/CLI/dashboard. They resolve
the spec's names through :data:`riskbudget.registry.REGISTRY`, fetch the data, and
drive the existing risk model / constructor / backtester — never re-implementing
any method.

- :func:`construct` — a single point-in-time solve over the spec's full data
  window (estimate cov / μ, then construct weights).
- :func:`backtest` — the walk-forward backtest over the same data.
- :func:`fetch_prices` — resolve the spec's data source into a ``PriceData`` panel.
"""

from __future__ import annotations

from datetime import date

from riskbudget.core.errors import BacktestError, ConfigurationError
from riskbudget.core.types import BacktestResult, Portfolio, PriceData
from riskbudget.spec import StrategySpec

# A generous default synthetic window when a spec leaves dates unset: ~6 years of
# business days, enough history for any lookback in the test/example suite.
_DEFAULT_START = date(2015, 1, 1)
_DEFAULT_END = date(2021, 1, 1)


def fetch_prices(spec: StrategySpec) -> PriceData:
    """Resolve ``spec``'s data source into a :class:`PriceData` panel.

    Uses ``spec.start`` / ``spec.end`` when set; otherwise falls back to a default
    multi-year window (the synthetic source is deterministic given the seed).
    """
    source = spec.build_data_source()
    start = spec.start if spec.start is not None else _DEFAULT_START
    end = spec.end if spec.end is not None else _DEFAULT_END
    try:
        prices = source.get_prices(list(spec.assets), start, end)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Data source {spec.data_source.name!r} could not supply prices: {exc}"
        ) from exc
    if not isinstance(prices, PriceData):  # pragma: no cover - defensive
        raise ConfigurationError(
            f"Data source {spec.data_source.name!r} did not return a PriceData panel."
        )
    return prices


def construct(spec: StrategySpec, *, prices: PriceData | None = None) -> Portfolio:
    """Solve a single :class:`Portfolio` from ``spec`` (point-in-time).

    Estimates the covariance (and, when configured, μ) over the full data window
    and constructs weights via the spec's ``method``. For the risk-budget path the
    spec's ``budget`` (ERC by default) is used.

    Parameters
    ----------
    spec:
        The strategy definition.
    prices:
        Optional pre-fetched price panel (skips the data-source call); must cover
        exactly ``spec.assets``.

    Raises
    ------
    ConfigurationError
        If ``spec.method`` is a dynamic allocator (use :func:`backtest` instead).
    """
    if spec.is_dynamic:
        raise ConfigurationError(
            f"Method {spec.method!r} is a dynamic allocator; "
            "use backtest() rather than construct()."
        )

    price_panel = prices if prices is not None else fetch_prices(spec)
    returns = price_panel.to_returns(method=spec.return_method).select(spec.assets)

    risk_model = spec.build_risk_model()
    cov = risk_model.estimate(returns)

    mean_model = spec.build_mean_model()
    mu = mean_model.estimate(returns) if mean_model is not None else None

    method = spec.build_method()
    if spec.target_volatility is not None:
        from riskbudget.optimize.voltarget import VolatilityTargetConstructor

        method = VolatilityTargetConstructor(
            method,
            target_volatility=spec.target_volatility,
            max_leverage=spec.target_vol_max_leverage,
        )
    budget = spec.build_budget()
    constraints = spec.build_constraints()

    try:
        if hasattr(method, "construct"):
            return method.construct(cov, mu=mu, budget=budget, constraints=constraints)  # type: ignore[no-any-return]
        if hasattr(method, "solve"):
            return method.solve(cov, budget, constraints)  # type: ignore[no-any-return]
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Construction failed for method {spec.method!r}: {exc}") from exc
    raise ConfigurationError(f"Method {spec.method!r} implements neither construct() nor solve().")


def backtest(spec: StrategySpec, *, prices: PriceData | None = None) -> BacktestResult:
    """Run the walk-forward backtest defined by ``spec``.

    Parameters
    ----------
    spec:
        The strategy definition.
    prices:
        Optional pre-fetched price panel (skips the data-source call); must cover
        exactly ``spec.assets``.

    Returns
    -------
    BacktestResult
        Equity curve, weights-through-time, per-rebalance diagnostics, and the
        net-of-cost return series.

    Raises
    ------
    ConfigurationError
        If ``spec.method`` is a dynamic allocator (not yet driven through the
        walk-forward backtester here).
    """
    if spec.is_dynamic:
        raise ConfigurationError(
            f"Method {spec.method!r} is a dynamic allocator; drive it via riskbudget.dynamic "
            "directly (the walk-forward backtester runs cross-sectional constructors)."
        )

    price_panel = prices if prices is not None else fetch_prices(spec)
    price_panel = PriceData(price_panel.frame.loc[:, list(spec.assets)])

    risk_model = spec.build_risk_model()
    mean_model = spec.build_mean_model()
    method = spec.build_method()
    if spec.target_volatility is not None:
        from riskbudget.optimize.voltarget import VolatilityTargetConstructor

        method = VolatilityTargetConstructor(
            method,
            target_volatility=spec.target_volatility,
            max_leverage=spec.target_vol_max_leverage,
        )
    budget = spec.build_budget()
    constraints = spec.build_constraints()
    schedule = spec.build_schedule()
    cost_model = spec.build_cost_model()

    from riskbudget.backtest import WalkForwardBacktester

    engine = WalkForwardBacktester(
        constraints=constraints,
        cost_model=cost_model,
        mean_model=mean_model,
        return_method=spec.return_method,
    )
    try:
        result = engine.run(price_panel, risk_model, method, budget, schedule)
    except BacktestError:
        raise
    except ConfigurationError:
        raise
    except Exception as exc:
        raise BacktestError(f"Backtest failed for {spec.name!r}: {exc}") from exc

    result.metadata.setdefault("strategy_name", spec.name)
    result.metadata.setdefault("method", spec.method)
    return result


__all__ = ["backtest", "construct", "fetch_prices"]
