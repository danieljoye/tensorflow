"""Construction & validation helpers for :class:`RiskBudget` (BUILD_PLAN §2, §5).

The :class:`~riskbudget.core.types.RiskBudget` dataclass itself lives in ``core``
(the frozen contract). This module adds the *construction* utilities the optimizer
layer needs but ``core`` deliberately keeps out:

- :func:`equal_risk_budget` — the ERC default ``bᵢ = 1/N``.
- :func:`budget_from_weights` — normalize arbitrary positive relative weights.
- :func:`group_risk_budget` — expand per-*group* budgets to per-asset budgets,
  splitting each group's share equally (or by relative weights) across its members.
- :func:`align_budget` — reconcile a budget to an explicit asset ordering.

Conditional / state-dependent budgets — a callable ``state -> RiskBudget``
(Martellini–Milhau–Tarelli) — are modelled by :class:`ConditionalBudget`, which
falls back to ERC when no signal is present. See ``optimize/conditional.py`` for
the constructor that consumes it.

Source: ERC (Maillard–Roncalli–Teïletche); conditional risk parity
(Martellini–Milhau–Tarelli, JAI 2015); BUILD_PLAN §2, §11.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from riskbudget.core.errors import ConfigurationError, ValidationError
from riskbudget.core.types import RiskBudget


def equal_risk_budget(assets: Sequence[str]) -> RiskBudget:
    """The equal-risk-contribution (ERC) budget ``bᵢ = 1/N``.

    Thin wrapper over :meth:`RiskBudget.equal` for symmetry with the other
    helpers here.
    """
    return RiskBudget.equal(assets)


def budget_from_weights(weights: Mapping[str, float]) -> RiskBudget:
    """Normalize arbitrary strictly-positive relative weights into a budget."""
    return RiskBudget.from_weights(weights)


def group_risk_budget(
    group_budgets: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    within_group: Mapping[str, float] | None = None,
) -> RiskBudget:
    """Expand per-group risk budgets to a per-asset :class:`RiskBudget`.

    Each group's target share is split across its member assets — equally by
    default, or by the relative ``within_group`` weights when supplied.

    Parameters
    ----------
    group_budgets:
        Mapping ``group_label -> target risk fraction`` (relative; normalized for
        you, so they need not sum to one).
    groups:
        Mapping ``asset_id -> group_label`` assigning each asset to a group.
    within_group:
        Optional mapping ``asset_id -> relative weight`` controlling the split
        *inside* a group. Assets omitted here default to an equal split within
        their group.

    Raises
    ------
    ConfigurationError
        If a referenced group has no assets, an asset references an unknown
        group, or a within-group weight is non-positive.
    """
    if not group_budgets:
        raise ConfigurationError("group_risk_budget requires at least one group budget.")
    if not groups:
        raise ConfigurationError("group_risk_budget requires an asset->group mapping.")

    members: dict[str, list[str]] = {g: [] for g in group_budgets}
    for asset, label in groups.items():
        if label not in group_budgets:
            raise ConfigurationError(f"Asset {asset!r} references group {label!r} with no budget.")
        members[label].append(str(asset))

    total_group = float(sum(group_budgets.values()))
    if total_group <= 0 or not np.isfinite(total_group):
        raise ConfigurationError("group budgets must have a finite positive sum.")

    per_asset: dict[str, float] = {}
    for label, share in group_budgets.items():
        if not np.isfinite(share) or share <= 0:
            raise ConfigurationError(f"group budget {label!r} must be finite and positive.")
        group_assets = members[label]
        if not group_assets:
            raise ConfigurationError(f"group {label!r} has a budget but no member assets.")
        group_share = float(share) / total_group

        if within_group is None:
            rel = dict.fromkeys(group_assets, 1.0)
        else:
            rel = {a: float(within_group.get(a, 1.0)) for a in group_assets}
        if any((not np.isfinite(v)) or v <= 0 for v in rel.values()):
            raise ConfigurationError(
                f"within-group weights for group {label!r} must be finite and positive."
            )
        rel_total = float(sum(rel.values()))
        for asset in group_assets:
            per_asset[asset] = group_share * rel[asset] / rel_total

    return RiskBudget(per_asset)


def align_budget(budget: RiskBudget, assets: Sequence[str]) -> np.ndarray:
    """Return ``budget`` as a vector ordered by ``assets`` (validating coverage)."""
    return budget.as_array(assets)


class ConditionalBudget:
    """A pluggable ``state -> RiskBudget`` mapping (Martellini–Milhau–Tarelli).

    Wraps a callable that turns an observable *state* (a rate level, valuation
    score, yield-curve slope, ...) into a :class:`RiskBudget`. When the state is
    ``None`` — or the callable declines to produce a budget — it falls back to
    the equal-risk-contribution (ERC) budget over ``assets``, so an absent signal
    reduces conditional risk parity to plain risk parity (BUILD_PLAN §2).

    Parameters
    ----------
    mapping:
        Callable ``state -> RiskBudget``. May return ``None`` to defer to ERC.
    assets:
        The asset universe; used to build the ERC fallback and to validate that
        any produced budget covers exactly these assets.

    Examples
    --------
    >>> cb = ConditionalBudget(
    ...     lambda s: None if s is None else RiskBudget.from_weights({"A": s, "B": 1.0}),
    ...     assets=["A", "B"],
    ... )
    >>> cb.budget_for(None).budgets["A"]  # flat signal -> ERC
    0.5
    """

    def __init__(
        self,
        mapping: Callable[[Any], RiskBudget | None],
        assets: Sequence[str],
    ) -> None:
        self._mapping = mapping
        self._assets = [str(a) for a in assets]
        if not self._assets:
            raise ConfigurationError("ConditionalBudget requires a non-empty asset list.")
        if len(set(self._assets)) != len(self._assets):
            raise ConfigurationError("ConditionalBudget assets must be unique.")
        self._erc = RiskBudget.equal(self._assets)

    @property
    def assets(self) -> list[str]:
        """The asset universe this conditional budget is defined over."""
        return list(self._assets)

    @property
    def fallback(self) -> RiskBudget:
        """The ERC budget used when no state-dependent budget is available."""
        return self._erc

    def budget_for(self, state: Any) -> RiskBudget:
        """Resolve the :class:`RiskBudget` for ``state`` (ERC fallback on ``None``).

        Raises
        ------
        ConfigurationError
            If the mapping returns a budget that does not cover exactly this
            conditional budget's asset universe.
        """
        if state is None:
            return self._erc
        produced = self._mapping(state)
        if produced is None:
            return self._erc
        # Validate coverage so a misbuilt mapping fails loudly, not silently.
        try:
            produced.as_array(self._assets)
        except ValidationError as exc:
            raise ConfigurationError(
                f"ConditionalBudget mapping produced a budget that does not cover "
                f"the asset universe {self._assets}: {exc}"
            ) from exc
        return produced

    def __call__(self, state: Any) -> RiskBudget:
        return self.budget_for(state)


__all__ = [
    "ConditionalBudget",
    "align_budget",
    "budget_from_weights",
    "equal_risk_budget",
    "group_risk_budget",
]
