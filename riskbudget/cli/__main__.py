"""Enable ``python -m riskbudget.cli`` to invoke the CLI."""

from __future__ import annotations

from riskbudget.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
