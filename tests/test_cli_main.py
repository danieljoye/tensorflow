"""CLI smoke tests (Agent 7): ``construct`` and ``backtest`` produce valid JSON.

Offline: a JSON :class:`StrategySpec` over the synthetic source is written to a
temp file and driven through ``riskbudget.cli.main.main`` in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from riskbudget.cli.main import main


def _write_spec(tmp_path: Path, method: str = "erc") -> Path:
    """Write a JSON StrategySpec over the synthetic source and return its path."""
    cov = (np.eye(3) * 0.04 + 0.005).tolist()
    spec = {
        "name": "cli-test",
        "assets": ["AAA", "BBB", "CCC"],
        "method": method,
        "risk_model": "sample",
        "data_source": {"name": "synthetic", "params": {"cov": cov}},
        "schedule": {"frequency": "monthly", "lookback": 60},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_construct_writes_valid_json(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path)
    out = tmp_path / "weights.json"
    rc = main(["construct", "--spec", str(spec), "--output", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["method"] == "erc"
    assert set(data["weights"]) == {"AAA", "BBB", "CCC"}
    assert set(data["risk_contributions"]) == {"AAA", "BBB", "CCC"}
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-6
    assert data["volatility"] > 0.0


def test_backtest_writes_valid_json(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path)
    out = tmp_path / "bt.json"
    rc = main(["backtest", "--spec", str(spec), "--equity-curve", "--output", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["method"] == "erc"
    assert data["n_periods"] > 0
    assert "sharpe_ratio" in data["metrics"]
    assert len(data["equity_curve"]) >= data["n_periods"]


def test_cli_flag_overrides_spec_method(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, method="erc")
    out = tmp_path / "w.json"
    rc = main(["construct", "--spec", str(spec), "--method", "equal_weight", "--output", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["method"] == "equal_weight"
    # Equal weight over 3 assets.
    for w in data["weights"].values():
        assert abs(w - 1.0 / 3.0) < 1e-6


def test_cli_inline_flags_without_spec_file(tmp_path: Path) -> None:
    # Build a synthetic-source spec entirely from flags is not possible (synthetic
    # needs a cov), so drive a CSV panel through the csv data source instead.
    import pandas as pd

    dates = pd.date_range("2018-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(400, 3)), axis=0))
    frame = pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])
    csv = tmp_path / "prices.csv"
    frame.to_csv(csv)

    out = tmp_path / "w.json"
    rc = main(
        [
            "construct",
            "--assets",
            "AAA,BBB,CCC",
            "--method",
            "erc",
            "--csv",
            str(csv),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["weights"]) == {"AAA", "BBB", "CCC"}


def test_unknown_method_returns_nonzero(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    spec = _write_spec(tmp_path)
    rc = main(["construct", "--spec", str(spec), "--method", "nope"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ConfigurationError" in err
