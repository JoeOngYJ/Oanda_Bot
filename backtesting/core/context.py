"""Backtest context / state container."""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BacktestContext:
    config: Dict[str, Any] = field(default_factory=dict)
    data_store: Dict[str, Any] = field(default_factory=dict)
    portfolios: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        return self.config.get(key, default)
