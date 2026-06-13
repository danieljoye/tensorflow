"""Tests for the Wave 0.5 additive contracts (BUILD_PLAN §5.1).

Covers the two new core types (:class:`ExpectedReturns`, :class:`AllocatorParams`)
and that minimal in-test implementations satisfy the three new runtime-checkable
protocols (:class:`MeanModel`, :class:`PortfolioConstructor`, :class:`Allocator`).
The emphasis is on validation and explicit asset-ordering alignment — the same
conventions Wave 0 established for ``RiskBudget`` / ``Portfolio``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from riskbudget.core.errors import ValidationError
from riskbudget.core.interfaces import (
    Allocator,
    Constraints,
    MeanModel,
    PortfolioConstructor,
)
from riskbudget.core.types import (
    AllocatorParams,
    BacktestResult,
    ExpectedReturns,
    Portfolio,
    ReturnMatrix,
    RiskBudget,
)

# ---------------------------------------------------------------------------
# ExpectedReturns
# ---------------------------------------------------------------------------


def test_expected_returns_assets_preserve_insertion_order() -> None:
    mu = ExpectedReturns({"AAA": 0.08, "BBB": 0.05, "CCC": -0.01})
    assert mu.assets == ["AAA", "BBB", "CCC"]
    assert len(mu) == 3


def test_expected_returns_as_array_default_order() -> None:
    mu = ExpectedReturns({"AAA": 0.08, "BBB": 0.05})
    np.testing.assert_array_equal(mu.as_array(), np.array([0.08, 0.05]))


def test_expected_returns_as_array_respects_explicit_ordering() -> None:
    mu = ExpectedReturns({"AAA": 0.08, "BBB": 0.05, "CCC": -0.01})
    reordered = ["CCC", "AAA", "BBB"]
    np.testing.assert_array_equal(mu.as_array(reordered), np.array([-0.01, 0.08, 0.05]))


def test_expected_returns_allows_negative_and_zero() -> None:
    # Unlike RiskBudget, μ may be negative or zero and need not sum to anything.
    mu = ExpectedReturns({"AAA": -0.2, "BBB": 0.0})
    np.testing.assert_array_equal(mu.as_array(), np.array([-0.2, 0.0]))


def test_expected_returns_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        ExpectedReturns({})


def test_expected_returns_rejects_nonfinite() -> None:
    with pytest.raises(ValidationError):
        ExpectedReturns({"AAA": float("nan")})
    with pytest.raises(ValidationError):
        ExpectedReturns({"AAA": float("inf")})


def test_expected_returns_as_array_rejects_unknown_asset() -> None:
    mu = ExpectedReturns({"AAA": 0.08, "BBB": 0.05})
    with pytest.raises(ValidationError):
        mu.as_array(["AAA", "ZZZ"])


def test_expected_returns_as_array_rejects_subset() -> None:
    mu = ExpectedReturns({"AAA": 0.08, "BBB": 0.05})
    with pytest.raises(ValidationError):
        mu.as_array(["AAA"])


def test_expected_returns_keys_coerced_to_str() -> None:
    mu = ExpectedReturns({1: 0.08, 2: 0.05})  # type: ignore[dict-item]
    assert mu.assets == ["1", "2"]
    np.testing.assert_array_equal(mu.as_array(["2", "1"]), np.array([0.05, 0.08]))


def test_expected_returns_aligns_with_covariance(cov_3: np.ndarray, assets: list[str]) -> None:
    # The whole point of as_array(assets): line μ up with a label-free cov.
    mu = ExpectedReturns(dict(zip(reversed(assets), [0.1, 0.2, 0.3], strict=True)))
    aligned = mu.as_array(assets)
    # Sharpe-style numerator wᵀμ should be order-invariant when both align to `assets`.
    w = np.array([0.5, 0.3, 0.2])
    assert float(w @ aligned) == pytest.approx(float(w @ mu.as_array(assets)), rel=1e-12)
    assert cov_3.shape[0] == aligned.size


# ---------------------------------------------------------------------------
# AllocatorParams
# ---------------------------------------------------------------------------


def test_allocator_params_defaults() -> None:
    p = AllocatorParams()
    assert p.multiplier == 3.0
    assert p.floor == 0.8
    assert p.max_drawdown is None
    assert p.safe_rate == 0.0
    assert p.start_value == 1.0


def test_allocator_params_drawdown_variant() -> None:
    p = AllocatorParams(multiplier=5.0, floor=0.0, max_drawdown=0.2, safe_rate=0.03)
    assert p.max_drawdown == 0.2
    assert p.safe_rate == 0.03


def test_allocator_params_rejects_nonpositive_multiplier() -> None:
    with pytest.raises(ValidationError):
        AllocatorParams(multiplier=0.0)
    with pytest.raises(ValidationError):
        AllocatorParams(multiplier=-1.0)


def test_allocator_params_rejects_floor_out_of_range() -> None:
    with pytest.raises(ValidationError):
        AllocatorParams(floor=1.0)  # floor must be < 1
    with pytest.raises(ValidationError):
        AllocatorParams(floor=-0.1)


def test_allocator_params_floor_zero_is_allowed() -> None:
    assert AllocatorParams(floor=0.0).floor == 0.0


def test_allocator_params_rejects_bad_max_drawdown() -> None:
    with pytest.raises(ValidationError):
        AllocatorParams(max_drawdown=0.0)  # must be > 0
    with pytest.raises(ValidationError):
        AllocatorParams(max_drawdown=1.5)  # must be <= 1


def test_allocator_params_max_drawdown_one_is_allowed() -> None:
    assert AllocatorParams(max_drawdown=1.0).max_drawdown == 1.0


def test_allocator_params_rejects_nonfinite_safe_rate() -> None:
    with pytest.raises(ValidationError):
        AllocatorParams(safe_rate=float("nan"))


def test_allocator_params_rejects_nonpositive_start_value() -> None:
    with pytest.raises(ValidationError):
        AllocatorParams(start_value=0.0)
    with pytest.raises(ValidationError):
        AllocatorParams(start_value=-1.0)


def test_allocator_params_is_frozen() -> None:
    p = AllocatorParams()
    with pytest.raises(Exception):  # noqa: B017 - dataclass FrozenInstanceError
        p.multiplier = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol conformance (structural typing, runtime-checkable)
# ---------------------------------------------------------------------------


class _StubMeanModel:
    def estimate(self, returns: ReturnMatrix) -> ExpectedReturns:
        means = returns.values.mean(axis=0) * 252.0  # naive annualization
        return ExpectedReturns(dict(zip(returns.assets, means, strict=True)))


class _StubConstructor:
    def construct(
        self,
        cov: np.ndarray,
        *,
        mu: ExpectedReturns | None = None,
        budget: RiskBudget | None = None,
        constraints: Constraints,
    ) -> Portfolio:
        n = cov.shape[0]
        # Resolve an asset ordering from whatever input is present.
        if budget is not None:
            order = budget.assets
        elif mu is not None:
            order = mu.assets
        else:
            order = [str(i) for i in range(n)]
        return Portfolio(dict.fromkeys(order, 1.0 / n))


class _StubAllocator:
    def allocate(
        self,
        risky: ReturnMatrix,
        safe: ReturnMatrix | float,
        params: AllocatorParams,
    ) -> BacktestResult:
        eq = pd.Series([params.start_value], index=risky.dates[:1])
        weights = pd.DataFrame({risky.assets[0]: [1.0]}, index=risky.dates[:1])
        return BacktestResult(equity_curve=eq, weights=weights)


def test_stubs_satisfy_new_protocols() -> None:
    assert isinstance(_StubMeanModel(), MeanModel)
    assert isinstance(_StubConstructor(), PortfolioConstructor)
    assert isinstance(_StubAllocator(), Allocator)


def test_mean_model_produces_aligned_expected_returns(return_matrix: ReturnMatrix) -> None:
    model: MeanModel = _StubMeanModel()
    mu = model.estimate(return_matrix)
    assert mu.assets == return_matrix.assets
    assert mu.as_array(return_matrix.assets).size == len(return_matrix.assets)


def test_constructor_runs_for_each_method(cov_3: np.ndarray, assets: list[str]) -> None:
    ctor: PortfolioConstructor = _StubConstructor()
    # GMV-style: cov only.
    gmv = ctor.construct(cov_3, constraints=Constraints())
    assert len(gmv) == len(assets)
    # MSR-style: cov + mu.
    mu = ExpectedReturns(dict(zip(assets, [0.08, 0.05, 0.02], strict=True)))
    msr = ctor.construct(cov_3, mu=mu, constraints=Constraints())
    assert msr.assets == assets
    # ERC-style: cov + budget.
    erc = ctor.construct(cov_3, budget=RiskBudget.equal(assets), constraints=Constraints())
    assert erc.assets == assets


def test_allocator_returns_backtest_result(return_matrix: ReturnMatrix) -> None:
    alloc: Allocator = _StubAllocator()
    result = alloc.allocate(return_matrix, 0.02, AllocatorParams(start_value=100.0))
    assert isinstance(result, BacktestResult)
    assert result.equity_curve.iloc[0] == 100.0
