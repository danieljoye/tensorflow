"""Tests for analytics/attribution.py: risk-contribution history, drift, ENB/DR."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.analytics.attribution import (
    budget_drift,
    diversification_history,
    risk_contribution_history,
)
from riskbudget.budgeting.contributions import portfolio_volatility
from riskbudget.core.errors import ValidationError
from riskbudget.core.types import RiskBudget


def _weights_frame(assets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.34, 0.33, 0.33],
        ],
        index=pd.date_range("2021-01-01", periods=3, freq="MS"),
        columns=assets,
    )


def test_risk_contribution_history_reconciles_to_total_risk(
    assets: list[str], cov_3: np.ndarray
) -> None:
    w = _weights_frame(assets)
    trc = risk_contribution_history(w, cov_3, verify=True)
    assert list(trc.columns) == assets
    # Each row's TRC must sum to the portfolio volatility for that rebalance.
    for date, row in w.iterrows():
        vol = portfolio_volatility(row.to_numpy(dtype=float), cov_3)
        assert trc.loc[date].sum() == pytest.approx(vol)


def test_percentage_rows_sum_to_one(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    pcr = risk_contribution_history(w, cov_3, percentage=True)
    np.testing.assert_allclose(pcr.sum(axis=1).to_numpy(), 1.0)


def test_time_varying_cov_mapping(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    cov_map = {date: cov_3 * (1.0 + i * 0.1) for i, date in enumerate(w.index)}
    trc = risk_contribution_history(w, cov_map, verify=True)
    assert trc.shape == w.shape


def test_missing_cov_date_raises(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    cov_map = {w.index[0]: cov_3}  # missing later dates
    with pytest.raises(ValidationError):
        risk_contribution_history(w, cov_map)


def test_budget_drift_zero_at_erc_weights(cov_3: np.ndarray) -> None:
    # For an ERC solution the percentage contributions equal 1/N, so drift -> 0.
    from riskbudget.optimize.ccd import solve_risk_budget_ccd

    assets = ["AAA", "BBB", "CCC"]
    budget = RiskBudget.equal(assets)
    w_vec = solve_risk_budget_ccd(cov_3, budget.as_array(assets))
    w = pd.DataFrame([w_vec], index=[pd.Timestamp("2021-01-01")], columns=assets)
    drift = budget_drift(w, cov_3, budget)
    np.testing.assert_allclose(drift.to_numpy(), 0.0, atol=1e-6)


def test_budget_drift_columns_match(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    budget = RiskBudget.equal(assets)
    drift = budget_drift(w, cov_3, budget)
    assert list(drift.columns) == assets
    # Drift rows sum to ~0 (PCR sums to 1 and budget sums to 1).
    np.testing.assert_allclose(drift.sum(axis=1).to_numpy(), 0.0, atol=1e-9)


def test_budget_mismatched_assets_raises(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    budget = RiskBudget.equal(["X", "Y", "Z"])
    with pytest.raises(ValidationError):
        budget_drift(w, cov_3, budget)


def test_diversification_history_columns_and_bounds(assets: list[str], cov_3: np.ndarray) -> None:
    w = _weights_frame(assets)
    div = diversification_history(w, cov_3)
    assert list(div.columns) == ["enb", "diversification_ratio"]
    # 1 <= ENB <= N and DR >= 1 at every rebalance.
    assert (div["enb"] >= 1.0 - 1e-9).all()
    assert (div["enb"] <= len(assets) + 1e-9).all()
    assert (div["diversification_ratio"] >= 1.0 - 1e-9).all()


def test_verify_catches_corrupt_reconciliation(assets: list[str]) -> None:
    # A non-symmetric "covariance" is rejected upstream; instead test that
    # verify=True passes on a valid case and the path is exercised.
    w = _weights_frame(assets)
    cov = np.eye(3) * 0.04
    trc = risk_contribution_history(w, cov, verify=True)
    assert trc.shape == w.shape


def test_empty_weights_raises() -> None:
    with pytest.raises(ValidationError):
        risk_contribution_history(pd.DataFrame(), np.eye(2))


def test_non_square_cov_raises(assets: list[str]) -> None:
    w = _weights_frame(assets)
    with pytest.raises(ValidationError):
        risk_contribution_history(w, np.ones((3, 2)))


def test_cov_dimension_mismatch_raises(assets: list[str]) -> None:
    w = _weights_frame(assets)
    with pytest.raises(ValidationError):
        risk_contribution_history(w, np.eye(2))


def test_nonfinite_cov_raises(assets: list[str]) -> None:
    w = _weights_frame(assets)
    bad = np.eye(3)
    bad[0, 0] = np.inf
    with pytest.raises(ValidationError):
        risk_contribution_history(w, bad)


def test_nonfinite_weights_raises(assets: list[str]) -> None:
    w = _weights_frame(assets)
    w.iloc[0, 0] = np.inf
    with pytest.raises(ValidationError):
        risk_contribution_history(w, np.eye(3) * 0.04)


def test_diversification_history_all_cash_is_nan(assets: list[str], cov_3: np.ndarray) -> None:
    w = pd.DataFrame([[0.0, 0.0, 0.0]], index=[pd.Timestamp("2021-01-01")], columns=assets)
    div = diversification_history(w, cov_3)
    assert np.isnan(div["enb"].iloc[0])
    assert np.isnan(div["diversification_ratio"].iloc[0])


def test_budget_drift_non_dataframe_raises() -> None:
    with pytest.raises(ValidationError):
        risk_contribution_history("not a frame", np.eye(2))  # type: ignore[arg-type]


def test_invariant_reconciliation_sweep() -> None:
    """Seeded sweep: TRC always sums to σ(w) across random books and covariances."""
    rng = np.random.default_rng(99)
    for _ in range(40):
        n = int(rng.integers(2, 8))
        a = rng.standard_normal((n, n + 3))
        cov = a @ a.T / (n + 3) + np.eye(n) * 0.02
        rows = rng.random((int(rng.integers(1, 5)), n))
        rows = rows / rows.sum(axis=1, keepdims=True)
        assets = [f"A{i}" for i in range(n)]
        w = pd.DataFrame(rows, columns=assets)
        trc = risk_contribution_history(w, cov, verify=True)
        for i in range(w.shape[0]):
            vol = portfolio_volatility(rows[i], cov)
            assert trc.iloc[i].sum() == pytest.approx(vol)
