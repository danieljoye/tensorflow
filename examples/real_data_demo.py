"""Real deep-history demo: Stocks / Bonds / Gold → ERC vs benchmarks → HTML report.

A reusable, **offline-by-default** end-to-end run on *real* (free, GitHub-mirrored)
deep-history market data:

- ``STOCKS`` = ``SP500_TR``  — S&P 500 total-return index (Shiller dataset).
- ``BONDS``  = ``US10Y_TR``  — a duration-8 10Y Treasury total-return **proxy**
  derived from Shiller's Long Interest Rate column (NOT an actual bond index;
  see :mod:`riskbudget.data.providers.shiller`).
- ``GOLD``   = ``GOLD``      — monthly gold spot price (datahub gold dataset).

The three monthly series are aligned from 1971 onward, combined into one
:class:`~riskbudget.PriceData` panel, and used to backtest an equal-risk-
contribution (ERC) strategy against equal-weight and global-minimum-variance
(GMV) benchmarks. A self-contained HTML report (Plotly inlined) is written to
disk.

By default everything reads the committed offline fixtures, so the script runs
with **no network**. Pass ``--live`` to refresh both series from GitHub.

Run it::

    python examples/real_data_demo.py                 # offline (committed fixtures)
    python examples/real_data_demo.py --live          # refresh from GitHub
    python examples/real_data_demo.py --out report.html

Renamed columns (``STOCKS`` / ``BONDS`` / ``GOLD``) keep the report readable.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import riskbudget as rb
from riskbudget.data.providers import GoldDataSource, ShillerDataSource

# Universe: friendly id -> (source key, raw asset id).
_STOCKS = "STOCKS"
_BONDS = "BONDS"
_GOLD = "GOLD"
_ASSETS = [_STOCKS, _BONDS, _GOLD]

_START = date(1971, 1, 1)
_END = date(2024, 12, 31)

# Monthly data => 12 periods/year for annualized analytics.
_PERIODS_PER_YEAR = 12


def build_panel(*, live: bool = False) -> rb.PriceData:
    """Build the aligned monthly STOCKS/BONDS/GOLD price panel.

    Loads ``SP500_TR`` + ``US10Y_TR`` from the Shiller source and ``GOLD`` from
    the gold source (offline fixtures by default; GitHub when ``live=True``),
    renames them to the friendly universe ids, aligns on the common monthly
    index from 1971, and returns a validated :class:`~riskbudget.PriceData`.
    """
    if live:
        shiller = ShillerDataSource.from_github()
        gold = GoldDataSource.from_github()
    else:
        shiller = ShillerDataSource.from_fixtures()
        gold = GoldDataSource.from_fixtures()

    equities = shiller.get_prices(["SP500_TR", "US10Y_TR"], _START, _END).frame
    equities = equities.rename(columns={"SP500_TR": _STOCKS, "US10Y_TR": _BONDS})
    metals = gold.get_prices(["GOLD"], _START, _END).frame.rename(columns={"GOLD": _GOLD})

    combined = pd.concat([equities, metals], axis=1, join="inner").sort_index()
    combined = combined.dropna(how="any")
    if combined.empty:
        raise rb.DataError("No overlapping monthly dates across STOCKS/BONDS/GOLD.")
    return rb.PriceData(combined.loc[:, _ASSETS])


def make_spec(name: str, method: str) -> rb.StrategySpec:
    """Build a :class:`StrategySpec` over the real deep-history universe.

    The data source is irrelevant here (we pass a pre-built panel to
    :func:`riskbudget.backtest`), so it defaults to ``synthetic``; only the
    method, risk model, and (monthly) schedule matter.
    """
    return rb.StrategySpec(
        name=name,
        assets=_ASSETS,
        method=method,
        risk_model="ledoit_wolf",
        schedule={"frequency": "monthly", "lookback": 60, "min_lookback": 24},
        return_method="simple",
    )


def main() -> None:
    """Run the offline (or ``--live``) deep-history demo and write an HTML report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh STOCKS/BONDS/GOLD from GitHub instead of the offline fixtures.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("real_data_demo_report.html"),
        help="Where to write the self-contained HTML report.",
    )
    args = parser.parse_args()

    panel = build_panel(live=args.live)
    print(
        f"Loaded {panel.shape[0]} monthly rows "
        f"({panel.dates.min().date()}..{panel.dates.max().date()}) for {panel.assets}."
    )

    erc = make_spec("ERC", "erc")
    equal_weight = make_spec("EqualWeight", "equal_weight")
    gmv = make_spec("GMV", "gmv")

    results = {spec.name: rb.backtest(spec, prices=panel) for spec in (erc, equal_weight, gmv)}
    print("\nFinal growth-of-$1:")
    for name, res in results.items():
        print(f"  {name:>12}: {float(res.equity_curve.iloc[-1]):.4f}")

    # Head-to-head comparison. ``compare()`` re-fetches each spec's own data
    # source, so instead we pool the backtested return series (all driven by the
    # shared real panel) into a single summary table.
    from riskbudget.analytics.summary import summary_stats

    pooled = pd.DataFrame(
        {name: res.returns for name, res in results.items() if res.returns is not None}
    ).dropna(how="any")
    table = summary_stats(pooled, periods_per_year=_PERIODS_PER_YEAR)
    cols = ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]
    print("\nComparison (ERC vs equal_weight vs GMV):")
    print(table[cols].round(4))

    from riskbudget.reporting import build_report

    report = build_report(
        results["ERC"],
        title="Deep-history ERC: Stocks / Bonds (10Y proxy) / Gold",
        strategy_name="ERC",
        periods_per_year=_PERIODS_PER_YEAR,
    )
    html = report.to_html(include_plotlyjs=True)
    args.out.write_text(html)
    print(f"\nWrote self-contained HTML report to {args.out.resolve()}.")


if __name__ == "__main__":
    main()
