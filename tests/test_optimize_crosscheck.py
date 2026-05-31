"""Optional, skippable cross-checks vs. reference risk-parity libraries.

These run only when the (dev-only) ``crosscheck`` extra is installed; otherwise
they skip cleanly. They validate our Spinu CCD ERC weights against the published
reference implementations on the same covariance (BUILD_PLAN §7, §11).
"""

from __future__ import annotations

import numpy as np
import pytest
from riskbudget.optimize.ccd import solve_risk_budget_ccd

COV = np.array(
    [
        [0.04, 0.006, 0.0, 0.01],
        [0.006, 0.09, 0.0, 0.0],
        [0.0, 0.0, 0.16, 0.02],
        [0.01, 0.0, 0.02, 0.0625],
    ]
)


def test_crosscheck_riskparityportfolio_erc() -> None:
    rpp = pytest.importorskip("riskparityportfolio")
    n = COV.shape[0]
    b = np.full(n, 1 / n)
    ours = solve_risk_budget_ccd(COV, b)
    ref = rpp.RiskParityPortfolio(covariance=COV, budget=b)
    ref.design()
    ref_w = np.asarray(ref.weights, dtype=float)
    ref_w = ref_w / ref_w.sum()
    assert np.allclose(ours, ref_w, atol=1e-5)


def test_crosscheck_pyrb_erc() -> None:
    pyrb = pytest.importorskip("pyrb")
    n = COV.shape[0]
    b = np.full(n, 1 / n)
    ours = solve_risk_budget_ccd(COV, b)
    ref_solver = pyrb.EqualRiskContribution(COV)
    ref_solver.solve()
    ref_w = np.asarray(ref_solver.x, dtype=float)
    ref_w = ref_w / ref_w.sum()
    assert np.allclose(ours, ref_w, atol=1e-5)
