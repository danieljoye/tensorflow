"""End-to-end integration tests for the StrategySpec → registry → pipeline glue.

These exercise the whole system through a single :class:`StrategySpec` for several
methods and assert the BUILD_PLAN §7 acceptance criteria for the
integration/glue area:

- a ``StrategySpec`` round-trips through ``construct``/``backtest``/``compare``;
- the registry resolves every method name and raises ``ConfigurationError`` on an
  unknown one;
- importing ``riskbudget`` does not pull the optional API/dashboard/plot deps;
- the golden-path example runs end-to-end offline.

Everything is offline (synthetic data) and seeded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import riskbudget as rb
from riskbudget.core.errors import ConfigurationError
from riskbudget.core.types import BacktestResult, Portfolio
from riskbudget.registry import REGISTRY, Registry, default_registry
from riskbudget.spec import StrategySpec, strategy_spec_from_dict

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def assets() -> list[str]:
    return ["EQ", "INTL", "BOND", "CREDIT"]


@pytest.fixture
def cov() -> np.ndarray:
    """A realistic per-period (daily) covariance for a 4-asset universe."""
    vol = np.array([0.011, 0.013, 0.004, 0.006])
    corr = np.array(
        [
            [1.00, 0.75, -0.20, 0.30],
            [0.75, 1.00, -0.15, 0.35],
            [-0.20, -0.15, 1.00, 0.50],
            [0.30, 0.35, 0.50, 1.00],
        ]
    )
    return corr * np.outer(vol, vol)


def _spec(
    name: str, method: str, assets: list[str], cov: np.ndarray, **extra: object
) -> StrategySpec:
    return StrategySpec(
        name=name,
        assets=assets,
        method=method,
        data_source={"name": "synthetic", "params": {"cov": cov.tolist()}},
        schedule={"frequency": "monthly", "lookback": 120},
        seed=42,
        **extra,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_every_method() -> None:
    expected = {
        "erc",
        "risk_budget",
        "gmv",
        "msr",
        "efficient_msr",
        "equal_weight",
        "mdp",
        "max_enb",
        "factor_risk_budget",
        "hrp",
        "ensemble",
    }
    assert set(REGISTRY.methods()) == expected
    for name in expected:
        assert callable(REGISTRY.method(name))


def test_registry_resolves_models_and_sources() -> None:
    assert set(REGISTRY.risk_models()) == {
        "sample",
        "ewma",
        "semicov",
        "ledoit_wolf",
        "oas",
        "pca",
    }
    assert set(REGISTRY.mean_models()) == {
        "mean_historical",
        "ewma_mean",
        "capm",
        "risk_based",
        "black_litterman",
    }
    assert set(REGISTRY.allocators()) == {
        "cppi",
        "fixed_mix",
        "glidepath",
        "floor",
        "drawdown",
        "fund_separation",
    }
    assert "synthetic" in REGISTRY.data_sources()
    assert "csv" in REGISTRY.data_sources()
    assert "tiingo" in REGISTRY.data_sources()
    assert REGISTRY.backtesters() == ["walkforward"]
    assert set(REGISTRY.cost_models()) == {"proportional", "none"}


@pytest.mark.parametrize(
    "resolver",
    ["method", "risk_model", "mean_model", "allocator", "data_source", "backtester", "cost_model"],
)
def test_unknown_name_raises_configuration_error(resolver: str) -> None:
    with pytest.raises(ConfigurationError) as exc:
        getattr(REGISTRY, resolver)("does_not_exist")
    # The error lists the valid options.
    assert "Valid options" in str(exc.value)


def test_default_registry_is_fresh_copy() -> None:
    reg = default_registry()
    assert isinstance(reg, Registry)
    assert reg is not REGISTRY
    assert reg.methods() == REGISTRY.methods()


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_rejects_unknown_method(assets: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        StrategySpec(assets=assets, method="bogus_method")


def test_spec_rejects_duplicate_assets() -> None:
    with pytest.raises(ConfigurationError):
        strategy_spec_from_dict({"assets": ["A", "A"], "method": "erc"})


def test_msr_requires_mean_model(assets: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        StrategySpec(assets=assets, method="msr")


def test_bl_views_require_bl_mean_model(assets: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        StrategySpec(
            assets=assets,
            method="erc",
            black_litterman={"absolute_views": {"EQ": 0.1}},
        )


def test_spec_round_trips_through_json(assets: list[str], cov: np.ndarray) -> None:
    spec = _spec("ERC", "erc", assets, cov)
    payload = spec.model_dump()
    rebuilt = strategy_spec_from_dict(payload)
    assert rebuilt.name == spec.name
    assert rebuilt.assets == spec.assets
    assert rebuilt.method == spec.method
    # JSON text round-trips too.
    assert StrategySpec.model_validate_json(spec.model_dump_json()).name == "ERC"


# ---------------------------------------------------------------------------
# construct / backtest through the spec
# ---------------------------------------------------------------------------


def test_construct_erc_weights_are_long_only_and_normalized(
    assets: list[str], cov: np.ndarray
) -> None:
    spec = _spec("ERC", "erc", assets, cov)
    portfolio = rb.construct(spec)
    assert isinstance(portfolio, Portfolio)
    weights = portfolio.as_array(assets)
    assert weights.min() >= 0.0  # long-only
    assert pytest.approx(weights.sum(), abs=1e-8) == 1.0


def test_construct_erc_equalizes_risk_contributions(assets: list[str], cov: np.ndarray) -> None:
    spec = _spec("ERC", "erc", assets, cov)
    portfolio = rb.construct(spec)
    # ERC equalizes contributions under the *estimated* covariance the solver saw.
    prices = rb.fetch_prices(spec)
    returns = prices.to_returns(method=spec.return_method).select(assets)
    est_cov = spec.build_risk_model().estimate(returns)
    rc = np.array(list(portfolio.risk_contributions(est_cov, assets).values()))
    assert (rc.max() - rc.min()) / rc.mean() < 1e-4


def test_gmv_has_lowest_volatility(assets: list[str], cov: np.ndarray) -> None:
    gmv_w = rb.construct(_spec("GMV", "gmv", assets, cov)).as_array(assets)
    ew_w = rb.construct(_spec("EW", "equal_weight", assets, cov)).as_array(assets)
    gmv_vol = float(np.sqrt(gmv_w @ cov @ gmv_w))
    ew_vol = float(np.sqrt(ew_w @ cov @ ew_w))
    assert gmv_vol <= ew_vol + 1e-12


@pytest.mark.parametrize(
    "method,extra",
    [
        ("erc", {}),
        ("risk_budget", {}),
        ("gmv", {}),
        ("equal_weight", {}),
        ("hrp", {}),
        ("mdp", {}),
        ("max_enb", {}),
        ("msr", {"mean_model": "risk_based"}),
        ("efficient_msr", {"mean_model": "risk_based"}),
    ],
)
def test_backtest_runs_for_every_method(
    assets: list[str], cov: np.ndarray, method: str, extra: dict[str, object]
) -> None:
    spec = _spec(method, method, assets, cov, **extra)
    result = rb.backtest(spec)
    assert isinstance(result, BacktestResult)
    assert result.returns is not None
    assert len(result.equity_curve) > 0
    assert float(result.equity_curve.iloc[-1]) > 0.0
    # Metadata is annotated with the strategy provenance.
    assert result.metadata["strategy_name"] == method
    assert result.metadata["method"] == method


def test_backtest_is_deterministic(assets: list[str], cov: np.ndarray) -> None:
    spec = _spec("ERC", "erc", assets, cov)
    a = rb.backtest(spec).equity_curve
    b = rb.backtest(spec).equity_curve
    pd.testing.assert_series_equal(a, b)


def test_black_litterman_views_through_spec(assets: list[str], cov: np.ndarray) -> None:
    spec = _spec(
        "BL",
        "msr",
        assets,
        cov,
        mean_model="black_litterman",
        black_litterman={"absolute_views": {"EQ": 0.12}, "confidences": {"EQ": 0.7}},
    )
    result = rb.backtest(spec)
    assert float(result.equity_curve.iloc[-1]) > 0.0


def test_costs_reduce_terminal_wealth(assets: list[str], cov: np.ndarray) -> None:
    free = _spec("free", "erc", assets, cov)
    costed = _spec(
        "costed",
        "erc",
        assets,
        cov,
        cost_model={"name": "proportional", "bps": 50.0},
    )
    assert float(rb.backtest(costed).equity_curve.iloc[-1]) <= float(
        rb.backtest(free).equity_curve.iloc[-1]
    )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_returns_one_row_per_strategy(assets: list[str], cov: np.ndarray) -> None:
    specs = [
        _spec("ERC", "erc", assets, cov),
        _spec("EqualWeight", "equal_weight", assets, cov),
        _spec("GMV", "gmv", assets, cov),
    ]
    table = rb.compare(specs)
    assert list(table.index) == ["ERC", "EqualWeight", "GMV"]
    for col in ("annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"):
        assert col in table.columns


def test_compare_rejects_empty_panel() -> None:
    with pytest.raises(ConfigurationError):
        rb.compare([])


def test_compare_rejects_duplicate_names(assets: list[str], cov: np.ndarray) -> None:
    dup = [_spec("same", "erc", assets, cov), _spec("same", "gmv", assets, cov)]
    with pytest.raises(ConfigurationError):
        rb.compare(dup)


# ---------------------------------------------------------------------------
# Dynamic methods route away from construct/backtest
# ---------------------------------------------------------------------------


def test_dynamic_method_rejected_by_construct(cov: np.ndarray) -> None:
    spec = StrategySpec(
        assets=["RISKY", "SAFE"],
        method="cppi",
        data_source={"name": "synthetic", "params": {"cov": (np.eye(2) * 1e-4).tolist()}},
    )
    assert spec.is_dynamic is True
    with pytest.raises(ConfigurationError):
        rb.construct(spec)
    with pytest.raises(ConfigurationError):
        rb.backtest(spec)


# ---------------------------------------------------------------------------
# Public API hygiene + universe + example
# ---------------------------------------------------------------------------


def test_import_riskbudget_does_not_pull_optional_deps() -> None:
    code = (
        "import sys, riskbudget; "
        "leaked=[m for m in ('fastapi','uvicorn','streamlit','plotly') if m in sys.modules]; "
        "print(leaked); assert not leaked, leaked"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


def test_default_universe_loads_symbols() -> None:
    symbols = rb.default_universe(limit=10)
    assert len(symbols) == 10
    assert all(isinstance(s, str) and s for s in symbols)
    # No duplicates in the returned slice.
    assert len(set(symbols)) == len(symbols)


def test_golden_path_example_runs_offline() -> None:
    script = REPO_ROOT / "examples" / "golden_path.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ERC weights" in proc.stdout
    assert "Comparison" in proc.stdout
