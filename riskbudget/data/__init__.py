"""Data-source implementations: the always-offline synthetic backbone + loaders.

This package provides the :class:`~riskbudget.core.interfaces.DataSource`
implementations every other agent's tests rely on (BUILD_PLAN §4, Agent 2):

- :class:`~riskbudget.data.synthetic.SyntheticDataSource` — a seeded,
  deterministic correlated-returns generator whose sample covariance converges
  to a known target. The offline backbone for the whole test suite.
- :class:`~riskbudget.data.csvsource.CsvDataSource` — a CSV/parquet price loader
  with a documented forward-fill policy.

Both satisfy the ``DataSource`` protocol (``get_prices(assets, start, end) ->
PriceData``) and are exposed as registry-friendly factory callables
(``synthetic_data_source`` / ``csv_data_source``) per BUILD_PLAN §5.2.
"""

from __future__ import annotations

from riskbudget.data.csvsource import CsvDataSource, csv_data_source
from riskbudget.data.synthetic import (
    SyntheticDataSource,
    generate_correlated_returns,
    synthetic_data_source,
)

__all__ = [
    "CsvDataSource",
    "SyntheticDataSource",
    "csv_data_source",
    "generate_correlated_returns",
    "synthetic_data_source",
]
