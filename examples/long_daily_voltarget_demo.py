"""Long-span daily volatility-targeting demo on the live total-return panel.

Runs every risk-budgeting method (ERC / GMV / Equal Weight / HRP / Max-ENB /
Max Diversification) on the **longest daily** multi-asset panel reachable from
this build environment — the **dividend-adjusted** S&P total return
(``STOCKS_TR``), the 10Y Treasury total-return proxy (``BONDS``) and gold
(``GOLD``) — vol-targeted to 10% annualized, monthly rebalancing with a 252-day
(1-year) trailing window and daily annualization (``periods_per_year=252``).

By default this fetches **fresh, live** data on every run via
:meth:`DailyPanelDataSource.from_github` so the panel stays current with the
upstream GitHub mirrors. Pass ``--offline`` to read the committed offline cache
(``examples/data/daily_panel_long.csv``) instead — that path needs no network and
is what CI uses.

Usage::

    python examples/long_daily_voltarget_demo.py            # live (fresh) data
    python examples/long_daily_voltarget_demo.py --offline  # committed fixture
    python examples/long_daily_voltarget_demo.py --out report.html --target-vol 0.10
"""

from __future__ import annotations

import argparse
from datetime import date

import riskbudget as rb
from riskbudget.data.providers.daily import DailyPanelDataSource
from riskbudget.reporting import build_comparison_report

ASSETS = ["STOCKS_TR", "BONDS", "GOLD"]
START = date(1968, 1, 1)
END = date(2100, 1, 1)
METHODS = {
    "ERC (risk parity)": "erc",
    "GMV": "gmv",
    "Equal Weight": "equal_weight",
    "HRP": "hrp",
    "Max-ENB": "max_enb",
    "Max Diversification": "mdp",
}


def load_panel(*, offline: bool = False) -> rb.PriceData:
    """Load the long daily total-return panel (live by default, fixture if offline)."""
    source = (
        DailyPanelDataSource.from_fixtures() if offline else DailyPanelDataSource.from_github()
    )
    return source.get_prices(ASSETS, START, END)


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
    ap.add_argument("--out", default="/tmp/long_daily_voltarget_report.html")
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Use the committed offline fixture instead of fetching live.",
    )
    args = ap.parse_args()

    panel = load_panel(offline=args.offline)
    assets = list(panel.assets)
    results = {
        label: rb.backtest(
            make_spec(label, m, assets=assets, target_vol=args.target_vol), prices=panel
        )
        for label, m in METHODS.items()
    }
    transport = "offline fixture" if args.offline else "live GitHub"
    rep = build_comparison_report(
        results,
        periods_per_year=252,
        title=f"Long daily vol-targeted to {args.target_vol:.0%} "
        f"— STOCKS_TR/BONDS/GOLD ({transport})",
    )
    print(rep.subtitle)
    print(rep.summary[["annualized_volatility", "sharpe_ratio", "max_drawdown"]].round(4))
    with open(args.out, "w") as fh:
        fh.write(rep.to_html(include_plotlyjs=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
