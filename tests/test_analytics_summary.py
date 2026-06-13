"""Tests for analytics/summary.py: summary_stats canonical table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.analytics.downside import cvar_historic, max_drawdown, var_gaussian
from riskbudget.analytics.performance import annualized_return, sharpe_ratio
from riskbudget.analytics.summary import SUMMARY_COLUMNS, summary_stats
from riskbudget.core.errors import ValidationError


def _returns(seed: int, n: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n) * 0.01 + 0.0003)


def test_series_returns_one_row_with_expected_columns() -> None:
    table = summary_stats(_returns(1).rename("erc"))
    assert list(table.columns) == SUMMARY_COLUMNS
    assert list(table.index) == ["erc"]
    assert table.shape == (1, len(SUMMARY_COLUMNS))


def test_unnamed_series_defaults_label() -> None:
    table = summary_stats(_returns(2))
    assert list(table.index) == ["strategy"]


def test_dataframe_one_row_per_strategy() -> None:
    df = pd.DataFrame({"erc": _returns(3), "gmv": _returns(4)})
    table = summary_stats(df)
    assert list(table.index) == ["erc", "gmv"]
    assert table.shape == (2, len(SUMMARY_COLUMNS))


def test_values_delegate_to_underlying_metrics() -> None:
    r = _returns(5)
    table = summary_stats(r.rename("s"), periods_per_year=252)
    assert table.loc["s", "annualized_return"] == pytest.approx(annualized_return(r, 252))
    assert table.loc["s", "sharpe_ratio"] == pytest.approx(sharpe_ratio(r, 0.0, 252))
    assert table.loc["s", "max_drawdown"] == pytest.approx(max_drawdown(r))
    assert table.loc["s", "cornish_fisher_var_5"] == pytest.approx(
        var_gaussian(r, 0.05, modified=True)
    )
    assert table.loc["s", "historic_cvar_5"] == pytest.approx(cvar_historic(r, 0.05))


def test_var_level_propagates() -> None:
    r = _returns(6).rename("s")
    t1 = summary_stats(r, var_level=0.05)
    t10 = summary_stats(r, var_level=0.10)
    # A wider tail (10%) is a less extreme quantile ⇒ smaller VaR magnitude.
    assert t10.loc["s", "cornish_fisher_var_5"] <= t1.loc["s", "cornish_fisher_var_5"]


def test_empty_dataframe_raises() -> None:
    with pytest.raises(ValidationError):
        summary_stats(pd.DataFrame())


def test_bad_type_raises() -> None:
    with pytest.raises(ValidationError):
        summary_stats([0.1, 0.2])  # type: ignore[arg-type]
