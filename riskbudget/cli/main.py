"""``riskbudget`` command-line interface — ``construct`` and ``backtest``.

Mirrors the API over the shell. A run is defined by a JSON :class:`StrategySpec`
(``--spec spec.json``) and/or individual flags (``--assets``, ``--method``,
``--risk-model`` …); flags override the file. Input prices come from CSV via
``--csv prices.csv`` (wired through the ``csv`` data source) or from the synthetic
backbone (``--cov cov.csv`` / a JSON cov in the spec). Output is JSON to stdout or
``--output``.

Examples
--------
Construct an ERC portfolio on a CSV price panel::

    python -m riskbudget.cli construct --csv prices.csv \\
        --assets AAA,BBB,CCC --method erc --output weights.json

Backtest GMV monthly with a 60-period lookback::

    python -m riskbudget.cli backtest --csv prices.csv \\
        --assets AAA,BBB,CCC --method gmv --frequency monthly --lookback 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import riskbudget
from riskbudget import StrategySpec
from riskbudget.analytics import summary_stats
from riskbudget.core.errors import RiskBudgetError
from riskbudget.core.types import BacktestResult, Portfolio


def _load_spec_file(path: str | None) -> dict[str, Any]:
    """Load a JSON StrategySpec dict from ``path`` (empty dict when unset)."""
    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Spec file {path!r} must contain a JSON object.")
    return data


def _apply_overrides(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Overlay CLI flags onto the spec payload (flags win over the file)."""
    spec = dict(payload)
    if args.name is not None:
        spec["name"] = args.name
    if args.assets is not None:
        spec["assets"] = [a.strip() for a in args.assets.split(",") if a.strip()]
    if args.method is not None:
        spec["method"] = args.method
    if args.risk_model is not None:
        spec["risk_model"] = args.risk_model
    if args.mean_model is not None:
        spec["mean_model"] = args.mean_model
    if args.seed is not None:
        spec["seed"] = args.seed
    if args.start is not None:
        spec["start"] = args.start
    if args.end is not None:
        spec["end"] = args.end

    if args.csv is not None:
        spec["data_source"] = {"name": "csv", "params": {"path": args.csv}}

    if getattr(args, "frequency", None) is not None or getattr(args, "lookback", None) is not None:
        schedule = dict(spec.get("schedule") or {})
        if getattr(args, "frequency", None) is not None:
            schedule["frequency"] = args.frequency
        if getattr(args, "lookback", None) is not None:
            schedule["lookback"] = args.lookback
        spec["schedule"] = schedule

    return spec


def _build_spec(args: argparse.Namespace) -> StrategySpec:
    """Assemble + validate the StrategySpec from the file and CLI overrides."""
    payload = _apply_overrides(_load_spec_file(args.spec), args)
    if "assets" not in payload:
        raise SystemExit("No assets given. Provide --assets or an 'assets' key in --spec.")
    return riskbudget.strategy_spec_from_dict(payload)


def _index_to_str(index: pd.Index) -> list[str]:
    """Stringify a (possibly datetime) index for JSON-friendly keys."""
    if isinstance(index, pd.DatetimeIndex):
        return [ts.strftime("%Y-%m-%d") for ts in index]
    return [str(v) for v in index]


def _write(output: str | None, payload: dict[str, Any]) -> None:
    """Write ``payload`` as JSON to ``output`` (or stdout when unset)."""
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        sys.stdout.write(text + "\n")
    else:
        Path(output).write_text(text + "\n", encoding="utf-8")


def _run_construct(args: argparse.Namespace) -> dict[str, Any]:
    """Build the spec, solve a portfolio, and return a JSON-ready payload."""
    spec = _build_spec(args)
    prices = riskbudget.fetch_prices(spec)
    portfolio: Portfolio = riskbudget.construct(spec, prices=prices)

    returns = prices.to_returns(method=spec.return_method).select(spec.assets)
    cov = spec.build_risk_model().estimate(returns)

    return {
        "name": spec.name,
        "method": spec.method,
        "assets": list(spec.assets),
        "weights": {a: float(portfolio.weights[a]) for a in spec.assets},
        "risk_contributions": {
            a: float(v) for a, v in portfolio.risk_contributions(cov, spec.assets).items()
        },
        "volatility": float(portfolio.volatility(cov, spec.assets)),
    }


def _run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    """Build the spec, run the backtest, and return a JSON-ready payload."""
    spec = _build_spec(args)
    result: BacktestResult = riskbudget.backtest(spec)

    metrics: dict[str, float] = {str(k): float(v) for k, v in result.metrics.items()}
    if result.returns is not None and result.returns.shape[0] > 1:
        table = summary_stats(
            result.returns.rename(spec.name),
            risk_free_rate=spec.risk_free_rate,
            periods_per_year=spec.periods_per_year,
        )
        row = table.loc[spec.name].astype(float).to_dict()
        for column, value in row.items():
            metrics.setdefault(str(column), float(value))

    payload: dict[str, Any] = {
        "name": spec.name,
        "method": spec.method,
        "assets": list(result.assets),
        "metrics": metrics,
        "start": str(result.start),
        "end": str(result.end),
        "n_periods": int(result.returns.shape[0]) if result.returns is not None else 0,
    }
    if args.equity_curve:
        curve = result.equity_curve.astype(float)
        payload["equity_curve"] = dict(zip(_index_to_str(curve.index), curve.tolist(), strict=True))
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI with ``construct`` and ``backtest`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="riskbudget",
        description="Construct and backtest risk-budgeted portfolios.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--spec", help="Path to a JSON StrategySpec file.")
        p.add_argument("--name", help="Strategy name.")
        p.add_argument("--assets", help="Comma-separated asset ids.")
        p.add_argument("--method", help="Construction method (e.g. erc, gmv, hrp).")
        p.add_argument("--risk-model", dest="risk_model", help="Risk model name.")
        p.add_argument("--mean-model", dest="mean_model", help="Mean model name.")
        p.add_argument("--csv", help="Path to a CSV price panel (wired via the csv source).")
        p.add_argument("--seed", type=int, help="Random seed for the synthetic source.")
        p.add_argument("--start", help="Start date (YYYY-MM-DD).")
        p.add_argument("--end", help="End date (YYYY-MM-DD).")
        p.add_argument("-o", "--output", help="Write JSON here instead of stdout.")

    construct = sub.add_parser("construct", help="Solve a single portfolio.")
    _add_common(construct)
    construct.set_defaults(handler=_run_construct)

    backtest = sub.add_parser("backtest", help="Run a walk-forward backtest.")
    _add_common(backtest)
    backtest.add_argument("--frequency", help="Rebalance frequency (e.g. monthly).")
    backtest.add_argument("--lookback", type=int, help="Estimation lookback in periods.")
    backtest.add_argument(
        "--equity-curve",
        action="store_true",
        help="Include the equity curve in the JSON output.",
    )
    backtest.set_defaults(handler=_run_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success, 1 on error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except RiskBudgetError as exc:
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return 1
    _write(args.output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/smoke test
    raise SystemExit(main())
