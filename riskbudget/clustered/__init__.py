"""Hierarchical / clustered allocation (López de Prado).

v1 ships Hierarchical Risk Parity (HRP): correlation-distance clustering →
quasi-diagonalisation → recursive bisection with inverse-variance weights, with
no matrix inversion (so it tolerates a singular covariance).
"""

from __future__ import annotations

from riskbudget.clustered.hrp import HRPConstructor, hrp

__all__ = ["HRPConstructor", "hrp"]
