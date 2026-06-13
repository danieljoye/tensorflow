"""Daily volatility-targeting demo.

Rebuilds the risk-budgeting methods on **real daily data** (S&P 500 sector
composites, 2013-2018) with the volatility-targeting overlay, where the target
volatility is estimated from a trailing **252-day (1-year) window of daily
returns** — not monthly returns. Run on daily prices with ``periods_per_year=252``
and a 252-period lookback, the covariance feeding both the allocation and the
vol-target overlay is exactly the last 252 daily observations.

Offline by default (reads the committed daily fixture). Writes a self-contained
HTML comparison report.

Usage::

    python examples/daily_voltarget_demo.py [--out report.html] [--target-vol 0.10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import riskbudget as rb
from riskbudget.reporting import build_comparison_report

FIXTURE = Path(__file__).resolve().parent / "data" / "daily_sectors_2013_2018.csv"
METHODS = {
    "ERC (risk parity)": "erc",
    "GMV": "gmv",
    "Equal Weight": "equal_weight",
    "HRP": "hrp",
    "Max-ENB": "max_enb",
    "Max Diversification": "mdp",
}


def load_panel() -> rb.PriceData:
    """Load the committed daily sector-composite price panel."""
    frame = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    return rb.PriceData(frame)


def make_spec(name: str, method: str, *, assets: list[str], target_vol: float) -> rb.StrategySpec:
    """Daily spec: 252-day lookback, daily annualization, vol-targeted."""
    return rb.StrategySpec(
        name=name,
        assets=assets,
        method=method,
        risk_model="ledoit_wolf",
        schedule={"frequency": "monthly", "lookback": 252, "min_lookback": 252},
        return_method="simple",
        periods_per_year=252,
        target_volatility=target_vol,
        target_vol_max_leverage=3.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/daily_voltarget_report.html")
    ap.add_argument("--target-vol", type=float, default=0.10)
    args = ap.parse_args()

    panel = load_panel()
    assets = list(panel.assets)
    results = {
        label: rb.backtest(
            make_spec(label, m, assets=assets, target_vol=args.target_vol), prices=panel
        )
        for label, m in METHODS.items()
    }
    rep = build_comparison_report(
        results,
        periods_per_year=252,
        title=f"Daily vol-targeted to {args.target_vol:.0%} "
        "— vol from 252 daily returns (US sectors, 2013-2018)",
    )
    print(rep.subtitle)
    print(rep.summary[["annualized_volatility", "sharpe_ratio", "max_drawdown"]].round(4))
    Path(args.out).write_text(rep.to_html(include_plotlyjs=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
