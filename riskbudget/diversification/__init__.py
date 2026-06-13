"""Diversification measurement and the *factor* sense of risk budgeting.

This package implements the Deguest–Martellini–Meucci / Choueifaty line of
diversification analysis (BUILD_PLAN §2, §11, §12.4–§12.5):

- :mod:`riskbudget.diversification.metrics` — the Diversification Ratio and the
  Effective Number of Bets (ENB) with the exact entropy guard.
- :mod:`riskbudget.diversification.torsion` — the minimum-torsion transform (the
  symmetric/eigendecomposition polar fixed-point iteration), PCA torsion, and the
  one-shot approximate torsion.
- :mod:`riskbudget.diversification.constructors` — :class:`PortfolioConstructor`
  implementations: the Most Diversified Portfolio (MDP), the max-ENB portfolio,
  and the factor-risk-budget portfolio on the min-torsion basis.
"""

from __future__ import annotations

from riskbudget.diversification.constructors import (
    FactorRiskBudgetConstructor,
    MaxENBConstructor,
    MostDiversifiedConstructor,
    factor_risk_budget,
    max_enb,
    mdp,
)
from riskbudget.diversification.metrics import (
    diversification_distribution,
    diversification_ratio,
    effective_number_of_bets,
)
from riskbudget.diversification.torsion import (
    TorsionMethod,
    approximate_torsion,
    cov_to_corr,
    minimum_torsion,
    pca_torsion,
    torsion,
)

__all__ = [
    "FactorRiskBudgetConstructor",
    "MaxENBConstructor",
    "MostDiversifiedConstructor",
    "TorsionMethod",
    "approximate_torsion",
    "cov_to_corr",
    "diversification_distribution",
    "diversification_ratio",
    "effective_number_of_bets",
    "factor_risk_budget",
    "max_enb",
    "mdp",
    "minimum_torsion",
    "pca_torsion",
    "torsion",
]
