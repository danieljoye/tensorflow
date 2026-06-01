"""Golden-path example: synthetic data → ERC vs. benchmarks → backtest → report.

A fully offline, deterministic end-to-end run of the Portfolio Risk Budgeting
system (BUILD_PLAN §5.2, §7). It builds a small correlated universe, defines an
equal-risk-contribution (ERC) strategy plus equal-weight and GMV benchmarks
through a single :class:`~riskbudget.StrategySpec` each, backtests them on the
synthetic data source, compares them head-to-head, and renders a report for the
ERC strategy.

Run it with::

    python examples/golden_path.py

Everything is seeded, so the printed numbers are reproducible and no network is
touched.
"""

from __future__ import annotations

import numpy as np
import riskbudget as rb


def build_universe() -> tuple[list[str], np.ndarray]:
    """A 5-asset universe with a realistic daily covariance (per-period)."""
    assets = ["EQ_US", "EQ_INTL", "BONDS", "CREDIT", "GOLD"]
    # Per-period (daily) volatilities and a plausible correlation matrix.
    vol = np.array([0.011, 0.013, 0.004, 0.006, 0.010])
    corr = np.array(
        [
            [1.00, 0.80, -0.20, 0.30, 0.10],
            [0.80, 1.00, -0.15, 0.35, 0.15],
            [-0.20, -0.15, 1.00, 0.50, 0.05],
            [0.30, 0.35, 0.50, 1.00, 0.10],
            [0.10, 0.15, 0.05, 0.10, 1.00],
        ]
    )
    cov = corr * np.outer(vol, vol)
    return assets, cov


def make_spec(
    name: str, method: str, assets: list[str], cov: np.ndarray, **extra: object
) -> rb.StrategySpec:
    """Build a :class:`StrategySpec` on the seeded synthetic source."""
    return rb.StrategySpec(
        name=name,
        assets=assets,
        method=method,
        data_source={"name": "synthetic", "params": {"cov": cov.tolist(), "mu": 0.0003}},
        risk_model="ledoit_wolf",
        schedule={"frequency": "monthly", "lookback": 120},
        cost_model={"name": "proportional", "bps": 5.0},
        seed=20240601,
        **extra,
    )


def main() -> None:
    """Run the golden path and print the comparison + a report summary."""
    assets, cov = build_universe()

    erc = make_spec("ERC", "erc", assets, cov)
    equal_weight = make_spec("EqualWeight", "equal_weight", assets, cov)
    gmv = make_spec("GMV", "gmv", assets, cov)

    # 1) Single point-in-time construction (risk contributions of the ERC book).
    portfolio = rb.construct(erc)
    print("ERC weights:")
    for asset, weight in portfolio.weights.items():
        print(f"  {asset:>8}: {weight:6.2%}")

    # 2) Walk-forward backtest of the ERC strategy.
    result = rb.backtest(erc)
    print(
        f"\nERC backtest: {result.diagnostics['n_rebalances']} rebalances, "
        f"final growth-of-$1 = {float(result.equity_curve.iloc[-1]):.4f}"
    )

    # 3) Head-to-head comparison vs. the benchmarks.
    table = rb.compare([erc, equal_weight, gmv])
    print("\nComparison (summary_stats):")
    cols = ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]
    print(table[cols].round(4))

    # 4) Render a report for the ERC strategy (tables + Plotly figures, built lazily).
    # The reporting package is imported here rather than via the top-level surface so
    # that ``import riskbudget`` does not pull plotly (BUILD_PLAN §3.1).
    from riskbudget.reporting import build_report

    report = build_report(result, title="ERC golden-path report", strategy_name="ERC")
    print(f"\nReport assembled: {report.title!r} with {len(report.figures)} figures.")


if __name__ == "__main__":
    main()
