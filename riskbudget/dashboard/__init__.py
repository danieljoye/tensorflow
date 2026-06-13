"""Streamlit dashboard for the risk-budgeting system (Agent 7).

Launch with ``streamlit run riskbudget/dashboard/app.py``. The UI lets a user pick
a universe, method, risk model, budget, and date range, then renders the weights,
a risk-contribution chart, the equity curve, the drawdown, and a ``summary_stats``
table comparing the chosen method against a benchmark.

``streamlit`` is imported inside :mod:`riskbudget.dashboard.app` (never at import
time of this package or of ``riskbudget`` itself), so the headless import used in
CI does not require the dashboard stack.
"""

from __future__ import annotations

__all__: list[str] = []
