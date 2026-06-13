"""Regression tests for the correctness-review findings M2, M3, L3-L8.

Each test pins the FIXED behavior so the original bug cannot silently return.
(Round 1 — H1/H2/M1/L1/L2 — lives in ``test_review_regressions.py``.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import riskbudget as rb
from riskbudget.core.errors import ConfigurationError, DataError
from riskbudget.core.interfaces import Constraints
from riskbudget.core.types import BacktestResult, Portfolio, RiskBudget
from riskbudget.data.providers.daily import (
    _indices_spx_close,
    _parse_flexible_decimal,
    _parse_two_col,
    _splice_stocks,
    _stocks_tr_index,
)
from riskbudget.optimize.router import RiskBudgetOptimizer
from riskbudget.optimize.voltarget import VolatilityTargetConstructor
from riskbudget.reporting.comparison import build_comparison_report
from riskbudget.spec import StrategySpec

COV = [[4e-4, 1e-4, 0.0], [1e-4, 9e-4, 2e-4], [0.0, 2e-4, 1.6e-3]]

# ---------------------------------------------------------------------------
# M2: Constraints.max_turnover is enforced at backtest rebalances
# ---------------------------------------------------------------------------

_M2_COMMON: dict = {
    "assets": ["A", "B", "C"],
    "data_source": {"name": "synthetic", "params": {"cov": COV}},
    "risk_model": "sample",
    # short lookback -> noisy sample cov -> real month-to-month churn
    "schedule": {"frequency": "monthly", "lookback": 60, "min_lookback": 60},
    "periods_per_year": 252,
}


def test_max_turnover_binds_on_every_rebalance_after_the_first() -> None:
    """End-to-end: max_turnover=0.05 caps per-rebalance turnover (bug M2)."""
    free = rb.backtest(StrategySpec(name="free", method="erc", **_M2_COMMON))
    free_turnover = [d["turnover"] for d in free.diagnostics["per_rebalance"]]
    # Premise: without the cap, some rebalance genuinely churns more than 5%
    # (otherwise this test would be vacuous). diag turnover is ONE-WAY.
    assert max(free_turnover[1:]) > 0.05

    capped = rb.backtest(
        StrategySpec(
            name="capped",
            method="erc",
            constraints={"max_turnover": 0.05},
            **_M2_COMMON,
        )
    )
    capped_turnover = [d["turnover"] for d in capped.diagnostics["per_rebalance"]]
    assert len(capped_turnover) == len(free_turnover)
    # First rebalance deploys from cash: deliberately exempt from the cap.
    assert capped_turnover[0] == pytest.approx(1.0)
    # Every later rebalance: two-way turnover Σ|Δw| <= 0.05 + tol. The diag value
    # is one-way (max of buy/sell legs), and Σ|Δw| <= 2 * one-way always.
    tol = 1e-6
    for t in capped_turnover[1:]:
        assert 2.0 * t <= 0.05 + tol


def test_optimizer_set_prev_weights_constrains_solve() -> None:
    """Unit: the duck-typed seam threads prev weights into the convex path."""
    cov = np.asarray(COV)
    budget = RiskBudget.equal(["A", "B", "C"])
    constraints = Constraints(max_turnover=0.10)
    opt = RiskBudgetOptimizer()

    # No prev weights (first rebalance): unconstrained-by-turnover ERC solve.
    free = opt.solve(cov, budget, constraints).as_array(["A", "B", "C"])

    # Prev book far from the ERC solution: the solve must stay within turnover.
    prev = {"A": 0.05, "B": 0.05, "C": 0.90}
    opt.set_prev_weights(prev)
    held = opt.solve(cov, budget, constraints).as_array(["A", "B", "C"])
    prev_arr = np.array([prev[a] for a in ["A", "B", "C"]])
    assert np.abs(held - prev_arr).sum() <= 0.10 + 1e-6
    # ... and is genuinely different from the unconstrained answer.
    assert np.abs(free - prev_arr).sum() > 0.10
    assert not np.allclose(held, free)

    # Clearing the state restores the unconstrained solve.
    opt.set_prev_weights(None)
    again = opt.solve(cov, budget, constraints).as_array(["A", "B", "C"])
    np.testing.assert_allclose(again, free, atol=1e-6)


def test_voltarget_forwards_prev_weights_to_inner() -> None:
    inner = RiskBudgetOptimizer()
    vt = VolatilityTargetConstructor(inner, target_volatility=0.10)
    vt.set_prev_weights({"A": 0.5, "B": 0.5})
    assert inner._prev_weights == {"A": 0.5, "B": 0.5}
    vt.set_prev_weights(None)
    assert inner._prev_weights is None


# ---------------------------------------------------------------------------
# M3: vol-target overlay vs inner constraints
# ---------------------------------------------------------------------------


def test_voltarget_with_explicit_nondefault_leverage_raises() -> None:
    with pytest.raises(ConfigurationError, match=r"target_volatility.*conflicts"):
        StrategySpec(
            name="conflict",
            assets=["A", "B"],
            data_source={"name": "synthetic", "params": {"cov": [[4e-4, 0.0], [0.0, 9e-4]]}},
            target_volatility=0.10,
            constraints={"leverage": 2.0},
        )


def test_voltarget_with_default_or_unset_leverage_is_fine() -> None:
    common: dict = {
        "assets": ["A", "B"],
        "data_source": {"name": "synthetic", "params": {"cov": [[4e-4, 0.0], [0.0, 9e-4]]}},
        "target_volatility": 0.10,
    }
    # constraints untouched (default leverage 1.0 not user-set): no conflict.
    StrategySpec(name="ok-default", **common)
    # leverage explicitly the 1.0 default value: no conflict (JSON round-trip safe).
    StrategySpec(name="ok-explicit-default", constraints={"leverage": 1.0}, **common)
    # leverage explicitly None (unconstrained): the overlay governs, no conflict.
    StrategySpec(name="ok-none", constraints={"leverage": None}, **common)
    # explicit non-default leverage WITHOUT a vol target is still allowed.
    StrategySpec(
        name="ok-no-target",
        assets=["A", "B"],
        data_source={"name": "synthetic", "params": {"cov": [[4e-4, 0.0], [0.0, 9e-4]]}},
        constraints={"leverage": 2.0},
    )


class _EqualWeight:
    def construct(self, cov, *, mu=None, budget=None, constraints):
        n = cov.shape[0]
        return Portfolio({f"A{i}": 1.0 / n for i in range(n)})


def test_voltarget_scale_capped_by_max_weight() -> None:
    """After scaling, no weight may exceed constraints.max_weight (bug M3c)."""
    tiny = np.asarray(COV) * 1e-6  # would want max_leverage (3x) -> 1.0 per asset
    vt = VolatilityTargetConstructor(_EqualWeight(), target_volatility=0.10, max_leverage=3.0)
    port = vt.construct(tiny, constraints=Constraints(max_weight=0.6))
    # k reduced from 3.0 to max_weight / max(inner weight) = 0.6 / (1/3) = 1.8
    assert max(port.weights.values()) == pytest.approx(0.6)
    assert port.leverage == pytest.approx(1.8)


def test_voltarget_max_weight_not_binding_leaves_scale_alone() -> None:
    tiny = np.asarray(COV) * 1e-6
    vt = VolatilityTargetConstructor(_EqualWeight(), target_volatility=0.10, max_leverage=2.0)
    port = vt.construct(tiny, constraints=Constraints(max_weight=0.9))
    # 2.0 * (1/3) = 0.667 <= 0.9: the cap does not bind, k stays at max_leverage.
    assert port.leverage == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# L3: dividend yield is lagged one month (no intra-month look-ahead)
# ---------------------------------------------------------------------------


def test_stocks_tr_uses_previous_months_dividend_yield() -> None:
    dates = pd.bdate_range("2000-01-03", "2000-02-29")
    stocks = pd.Series(100.0, index=dates)  # flat price: TR growth == carry only
    y_jan, y_feb = 0.0252, 0.0504
    div_yield = pd.Series(
        [y_jan, y_feb],
        index=pd.DatetimeIndex([pd.Timestamp("2000-01-01"), pd.Timestamp("2000-02-01")]),
    )
    tr = _stocks_tr_index(stocks, div_yield)
    # January days: month M-1 (Dec 1999) has no observation -> zero carry, NOT
    # January's own (look-ahead) yield.
    jan = tr[tr.index < pd.Timestamp("2000-02-01")]
    assert jan.iloc[-1] / jan.iloc[0] == pytest.approx(1.0)
    # February days carry JANUARY's yield (y/252 per day), not February's.
    feb_first = tr.loc[pd.Timestamp("2000-02-01")]
    feb_next = tr.loc[pd.Timestamp("2000-02-02")]
    assert feb_next / feb_first == pytest.approx(1.0 + y_jan / 252.0)


# ---------------------------------------------------------------------------
# L4: per-value decimal-convention detection + splice sanity checks
# ---------------------------------------------------------------------------


def test_parse_flexible_decimal_conventions() -> None:
    assert _parse_flexible_decimal("6.800,26") == pytest.approx(6800.26)  # European
    assert _parse_flexible_decimal("6800,26") == pytest.approx(6800.26)  # decimal comma
    assert _parse_flexible_decimal("6800.26") == pytest.approx(6800.26)  # US decimal
    assert _parse_flexible_decimal("6800") == pytest.approx(6800.0)
    with pytest.raises(ValueError):
        _parse_flexible_decimal("not-a-number")


def test_indices_spx_parses_us_format_text() -> None:
    # A US-decimal dump must parse as-is — the old code blindly stripped '.'
    # and would have read 4756.50 as 475650.0.
    text = (
        "Dates;EMB US Equity;SPX Index\n02.01.2024;#N/A N/A;4756.50\n03.01.2024;#N/A N/A;4763.25\n"
    )
    series = _indices_spx_close(text)
    assert series.tolist() == [4756.50, 4763.25]


def test_indices_spx_still_parses_european_format_text() -> None:
    text = (
        "Dates;EMB US Equity;SPX Index\n02.01.2024;#N/A N/A;4.756,50\n03.01.2024;#N/A N/A;4763,25\n"
    )
    series = _indices_spx_close(text)
    assert series.tolist() == [4756.50, 4763.25]


def test_splice_absurd_seam_scale_raises() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp(2024, 2, 26)])
    deep = pd.Series([200.0], index=idx)
    current = pd.Series(
        [100.0, 110.0], index=idx.append(pd.DatetimeIndex([pd.Timestamp(2024, 2, 27)]))
    )
    with pytest.raises(DataError, match="seam scale"):
        _splice_stocks(deep, current)


def test_splice_disagreeing_overlap_window_raises() -> None:
    # Seam scale is exactly 1.0 (fine) but earlier common days disagree by ~50%,
    # so the overlap-agreement check must reject the splice.
    idx = pd.date_range("2024-02-20", periods=3, freq="B")
    deep = pd.Series([100.0, 100.0, 100.0], index=idx)
    current = pd.Series(
        [50.0, 50.0, 100.0, 101.0],
        index=idx.append(pd.DatetimeIndex([pd.Timestamp(2024, 2, 23)])),
    )
    with pytest.raises(DataError, match="overlap check"):
        _splice_stocks(deep, current)


# ---------------------------------------------------------------------------
# L5: a regressed (shorter) current leg must not truncate the deep leg
# ---------------------------------------------------------------------------


def test_splice_keeps_deep_tail_when_current_leg_ends_earlier() -> None:
    deep_idx = pd.date_range("2024-02-19", periods=5, freq="B")  # ends 2024-02-23
    deep = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=deep_idx)
    # Current leg overlaps the first three days then STOPS (upstream regression).
    current = pd.Series([100.0, 101.0, 102.0], index=deep_idx[:3])
    spliced = _splice_stocks(deep, current)
    # The deep leg's tail (2024-02-22, 2024-02-23) is preserved verbatim.
    assert list(spliced.index) == list(deep_idx)
    assert spliced.tolist() == deep.tolist()


def test_splice_current_extends_deep_only_past_deep_end() -> None:
    deep_idx = pd.date_range("2024-02-19", periods=3, freq="B")
    deep = pd.Series([100.0, 101.0, 102.0], index=deep_idx)
    ext_idx = deep_idx.append(pd.DatetimeIndex([pd.Timestamp(2024, 2, 22)]))
    current = pd.Series([100.0, 101.0, 102.0, 103.0], index=ext_idx)
    spliced = _splice_stocks(deep, current)
    assert list(spliced.index) == list(ext_idx)
    assert spliced.iloc[-1] == pytest.approx(103.0)


# ---------------------------------------------------------------------------
# L6: gold parser locates the USD column by header name
# ---------------------------------------------------------------------------


def test_gold_parser_survives_reordered_columns() -> None:
    reordered = (
        "date,gold_pm_gbp,gold_pm_eur,gold_pm_usd\n1968-01-02,15.0,0,35.0\n1968-01-03,15.2,0,36.0\n"
    )
    series = _parse_two_col(reordered, "GOLD", prefer_columns=("gold_pm_usd",))
    assert series.tolist() == [35.0, 36.0]  # USD, not the GBP column-1 values


def test_two_col_parser_falls_back_to_column_one() -> None:
    # Generic payloads without the preferred header keep the old behavior.
    text = "observation_date,DGS10\n1968-01-02,6.0\n1968-01-03,5.0\n"
    series = _parse_two_col(text, "DGS10", prefer_columns=("gold_pm_usd",))
    assert series.tolist() == [6.0, 5.0]


# ---------------------------------------------------------------------------
# L7/L8: comparison report — mixed-cadence subtitle + composition vs leverage
# ---------------------------------------------------------------------------


def _result(
    name: str,
    *,
    frequency: str = "monthly",
    lookback: int = 60,
    n_rebalances: int = 24,
    gross: float = 1.0,
    assets: tuple[str, ...] = ("STOCKS", "BONDS"),
) -> BacktestResult:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2000-01-31", periods=n_rebalances, freq="ME")
    rets = pd.Series(rng.normal(0.005, 0.02, len(idx)), index=idx)
    mix = rng.dirichlet(np.ones(len(assets)), len(idx))  # rows sum to 1
    weights = pd.DataFrame(mix * gross, index=idx, columns=list(assets))
    return BacktestResult(
        equity_curve=(1.0 + rets).cumprod(),
        weights=weights,
        returns=rets,
        metadata={"name": name, "frequency": frequency, "lookback": lookback},
        diagnostics={"n_rebalances": n_rebalances},
    )


def test_subtitle_reports_mixed_cadence_and_lookback_range() -> None:
    results = {
        "ERC-m": _result("ERC-m", frequency="monthly", lookback=60, n_rebalances=24),
        "ERC-q": _result("ERC-q", frequency="quarterly", lookback=252, n_rebalances=8),
    }
    rep = build_comparison_report(results, periods_per_year=12)
    assert "Mixed rebalancing (monthly, quarterly)" in rep.subtitle
    assert "lookbacks 60-252" in rep.subtitle
    assert "8-24 rebalances" in rep.subtitle
    assert rep.metadata["rebalance"] == rep.subtitle


def test_subtitle_unchanged_when_results_agree() -> None:
    results = {
        "A": _result("A", frequency="monthly", lookback=60, n_rebalances=24),
        "B": _result("B", frequency="monthly", lookback=60, n_rebalances=24),
    }
    rep = build_comparison_report(results, periods_per_year=12)
    assert "Monthly rebalancing" in rep.subtitle
    assert "60-month lookback" in rep.subtitle
    assert "24 rebalances" in rep.subtitle


def test_composition_normalizes_by_gross_and_annotates_leverage() -> None:
    results = {
        "1x": _result("1x", gross=1.0),
        "2x": _result("2x", gross=2.0),
    }
    rep = build_comparison_report(results, periods_per_year=12)
    comp = rep.figures["composition"]
    # Stacked composition mix sums to ~100% per strategy REGARDLESS of leverage
    # (the old chart stacked the 2x book to 200%).
    totals = np.zeros(2)
    for trace in comp.data:
        totals = totals + np.asarray(trace.y, dtype=float)
    np.testing.assert_allclose(totals, [1.0, 1.0], atol=1e-9)
    # Mean gross leverage is annotated per strategy.
    notes = {a.text for a in comp.layout.annotations}
    assert "1.0x gross" in notes
    assert "2.0x gross" in notes


def test_composition_missing_asset_renders_zero_percent() -> None:
    results = {
        "AB": _result("AB", assets=("STOCKS", "BONDS")),
        "AG": _result("AG", assets=("STOCKS", "GOLD")),
    }
    rep = build_comparison_report(results, periods_per_year=12)
    comp = rep.figures["composition"]
    by_name = {trace.name: trace for trace in comp.data}
    # GOLD is missing from "AB": its share renders as 0% (not NaN).
    gold = by_name["GOLD"]
    strategies = list(gold.x)
    gold_y = dict(zip(strategies, np.asarray(gold.y, dtype=float), strict=True))
    assert gold_y["AB"] == pytest.approx(0.0)
    assert "0%" in list(gold.text)
    assert np.isfinite(np.asarray(gold.y, dtype=float)).all()
