"""
Abstract base class for all trading strategies.
Defines the interface that all strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from oanda_bot.utils.models import MarketTick, TradeSignal, Instrument


class BaseStrategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, config: dict):
        self.name = config['name']
        self.version = config['version']
        self.instruments = [Instrument(i) for i in config['instruments']]
        self.enabled = config.get('enabled', True)
        self.parameters = config.get('parameters', {})

    @abstractmethod
    def update(self, tick: MarketTick) -> None:
        """
        Update strategy state with new market tick.

        Args:
            tick: New market data tick
        """
        pass

    @abstractmethod
    def check_signal(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Check if current state generates a trade signal.

        Args:
            tick: Current market tick

        Returns:
            TradeSignal if conditions met, else None
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """
        Return current strategy state for debugging/auditing.

        Returns:
            Dictionary with strategy state
        """
        pass
