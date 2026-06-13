"""Streamlit UI for constructing + backtesting risk-budgeted portfolios.

Run it with::

    streamlit run riskbudget/dashboard/app.py

The page lets a user choose a universe, a construction ``method``, a ``risk_model``,
a relative risk ``budget``, and a date range, then shows: the constructed weights,
a risk-contribution bar chart, the backtest equity curve, the drawdown, and a
``summary_stats`` table comparing the chosen method against a benchmark
(equal-weight by default).

Design
------
The data assembly is split out of the rendering so it can be unit-tested without a
live Streamlit server:

- :func:`build_dashboard_payload` — pure: spec → portfolio + backtest + comparison
  table + Plotly figures. Drives the shared ``riskbudget`` wiring only.
- :func:`render` / :func:`main` — the Streamlit page; ``streamlit`` is imported
  lazily inside them so importing this module stays headless-safe (BUILD_PLAN §3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

import riskbudget
from riskbudget import REGISTRY, StrategySpec
from riskbudget.compare import compare
from riskbudget.core.types import BacktestResult, Portfolio

# A small, well-conditioned default covariance for the synthetic backbone so the
# dashboard works out-of-the-box with no uploaded data.
_DEFAULT_ASSETS = ["EQUITY", "BONDS", "GOLD", "CREDIT"]


def default_covariance(assets: list[str], *, seed: int = 0) -> list[list[float]]:
    """A deterministic, positive-definite per-period covariance for ``assets``.

    Used to seed the synthetic data source when the user has not supplied their
    own panel. Distinct asset vols and mild positive correlation keep the
    risk-budget solver well posed.
    """
    n = len(assets)
    rng = np.random.default_rng(seed)
    vols = 0.008 + 0.012 * rng.random(n)  # daily vols ~ 0.8%-2%
    corr = np.full((n, n), 0.2)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    cov = (cov + cov.T) / 2.0
    return [[float(x) for x in row] for row in cov.tolist()]


@dataclass
class DashboardPayload:
    """Everything the page renders, assembled off the Streamlit thread."""

    spec: StrategySpec
    portfolio: Portfolio
    weights: dict[str, float]
    risk_contributions: dict[str, float]
    volatility: float
    result: BacktestResult
    benchmark_result: BacktestResult
    summary: pd.DataFrame
    figures: dict[str, Any] = field(default_factory=dict)


def build_spec(
    *,
    name: str,
    assets: list[str],
    method: str,
    risk_model: str,
    budget: dict[str, float] | None = None,
    mean_model: str | None = None,
    frequency: str = "monthly",
    lookback: int = 60,
    seed: int = 0,
    start: date | None = None,
    end: date | None = None,
    cov: list[list[float]] | None = None,
    data_source: dict[str, Any] | None = None,
) -> StrategySpec:
    """Assemble a :class:`StrategySpec` for the dashboard (synthetic by default)."""
    if data_source is None:
        params: dict[str, Any] = {"cov": cov if cov is not None else default_covariance(assets)}
        data_source = {"name": "synthetic", "params": params}
    payload: dict[str, Any] = {
        "name": name,
        "assets": assets,
        "method": method,
        "risk_model": risk_model,
        "data_source": data_source,
        "schedule": {"frequency": frequency, "lookback": lookback},
        "seed": seed,
    }
    if budget:
        payload["budget"] = budget
    if mean_model is not None:
        payload["mean_model"] = mean_model
    if start is not None:
        payload["start"] = start
    if end is not None:
        payload["end"] = end
    return riskbudget.strategy_spec_from_dict(payload)


def build_dashboard_payload(
    spec: StrategySpec,
    *,
    benchmark_method: str = "equal_weight",
    make_figures: bool = False,
) -> DashboardPayload:
    """Run the chosen strategy + a benchmark and assemble the renderable payload.

    Parameters
    ----------
    spec:
        The chosen strategy.
    benchmark_method:
        A second method run on the *same* data for a head-to-head summary
        (``equal_weight`` by default; ``gmv`` is the other natural choice).
    make_figures:
        When ``True`` build the Plotly figures via :mod:`riskbudget.reporting`
        (kept off by default so the headless import/build test does not need
        plotly).
    """
    prices = riskbudget.fetch_prices(spec)
    portfolio = riskbudget.construct(spec, prices=prices)

    returns = prices.to_returns(method=spec.return_method).select(spec.assets)
    cov = spec.build_risk_model().estimate(returns)
    risk_contributions = portfolio.risk_contributions(cov, spec.assets)
    volatility = portfolio.volatility(cov, spec.assets)

    result = riskbudget.backtest(spec, prices=prices)

    benchmark_spec = _benchmark_spec(spec, benchmark_method)
    benchmark_result = riskbudget.backtest(benchmark_spec, prices=prices)

    summary = compare([spec, benchmark_spec])

    figures: dict[str, Any] = {}
    if make_figures:
        from riskbudget.reporting import build_report

        report = build_report(
            result,
            cov=cov,
            title=f"{spec.name} ({spec.method})",
            strategy_name=spec.name,
            risk_free_rate=spec.risk_free_rate,
            periods_per_year=spec.periods_per_year,
        )
        figures = dict(report.figures)

    return DashboardPayload(
        spec=spec,
        portfolio=portfolio,
        weights={a: float(portfolio.weights[a]) for a in spec.assets},
        risk_contributions={a: float(risk_contributions[a]) for a in spec.assets},
        volatility=float(volatility),
        result=result,
        benchmark_result=benchmark_result,
        summary=summary,
        figures=figures,
    )


def _benchmark_spec(spec: StrategySpec, benchmark_method: str) -> StrategySpec:
    """A same-data benchmark spec (distinct name; plain weighting rule)."""
    payload = spec.model_dump()
    payload["name"] = f"{spec.name} [{benchmark_method}]"
    payload["method"] = benchmark_method
    payload["method_params"] = {}
    payload["mean_model"] = None
    payload["mean_model_params"] = {}
    payload["black_litterman"] = None
    payload["budget"] = None
    return riskbudget.strategy_spec_from_dict(payload)


# ---------------------------------------------------------------------------
# Streamlit rendering (streamlit imported lazily)
# ---------------------------------------------------------------------------


def render() -> None:  # pragma: no cover - exercised only under a live server
    """Render the Streamlit page. Imports ``streamlit`` lazily."""
    import streamlit as st

    st.set_page_config(page_title="riskbudget", layout="wide")
    st.title("Portfolio Risk Budgeting")
    st.caption(
        "Construct portfolios by risk budget, backtest them on the synthetic "
        "backbone, and compare against a benchmark."
    )

    methods = REGISTRY.methods()
    risk_models = REGISTRY.risk_models()

    with st.sidebar:
        st.header("Configuration")
        assets_text = st.text_input("Assets (comma-separated)", ",".join(_DEFAULT_ASSETS))
        assets = [a.strip() for a in assets_text.split(",") if a.strip()]
        method = st.selectbox("Method", methods, index=methods.index("erc"))
        risk_model = st.selectbox("Risk model", risk_models, index=risk_models.index("sample"))

        mean_model = None
        if REGISTRY.requires_mu(method):
            mean_models = REGISTRY.mean_models()
            mean_model = st.selectbox("Mean model (required)", mean_models)

        benchmark = st.selectbox(
            "Benchmark",
            [m for m in methods if m != method] or ["equal_weight"],
            index=0,
        )
        frequency = st.selectbox(
            "Rebalance frequency",
            ["monthly", "weekly", "quarterly", "annual"],
            index=0,
        )
        lookback = int(st.number_input("Lookback (periods)", min_value=10, value=60, step=10))
        col_a, col_b = st.columns(2)
        start = col_a.date_input("Start", value=date(2016, 1, 1))
        end = col_b.date_input("End", value=date(2021, 1, 1))
        seed = int(st.number_input("Seed", min_value=0, value=0, step=1))
        run = st.button("Run", type="primary")

    if not run:
        st.info("Set the configuration in the sidebar and press **Run**.")
        return

    if len(assets) < 2:
        st.error("Provide at least two assets.")
        return

    try:
        spec = build_spec(
            name=f"{method.upper()}",
            assets=assets,
            method=method,
            risk_model=risk_model,
            mean_model=mean_model,
            frequency=frequency,
            lookback=lookback,
            seed=seed,
            start=start if isinstance(start, date) else None,
            end=end if isinstance(end, date) else None,
        )
        payload = build_dashboard_payload(spec, benchmark_method=benchmark, make_figures=True)
    except riskbudget.RiskBudgetError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Weights")
        st.bar_chart(pd.Series(payload.weights, name="weight"))
        st.metric("Portfolio volatility (per period)", f"{payload.volatility:.4%}")
    with right:
        st.subheader("Risk contributions")
        st.bar_chart(pd.Series(payload.risk_contributions, name="risk contribution"))

    st.subheader("Equity curve")
    if "equity" in payload.figures:
        st.plotly_chart(payload.figures["equity"], use_container_width=True)
    st.subheader("Drawdown")
    if "drawdown" in payload.figures:
        st.plotly_chart(payload.figures["drawdown"], use_container_width=True)

    st.subheader("Summary statistics (strategy vs. benchmark)")
    st.dataframe(payload.summary)


def main() -> None:  # pragma: no cover - exercised only under a live server
    """Module entry point for ``streamlit run``."""
    render()


# Streamlit executes the script top-to-bottom; ``render`` is called only when this
# file is run as the Streamlit entrypoint (``__main__``), not on import.
if __name__ == "__main__":  # pragma: no cover - live server only
    main()
