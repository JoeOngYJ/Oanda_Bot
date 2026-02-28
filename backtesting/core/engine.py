"""Main backtest orchestrator."""
from typing import Optional


class BacktestEngine:
    """A minimal backtest engine skeleton.

    Responsibilities:
    - initialize data providers and event bus
    - run the event loop
    - collect results
    """

    def __init__(self, context: Optional[dict] = None):
        self.context = context or {}

    def run(self):
        """Run the backtest (placeholder)."""
        print("Running backtest (stub)")
