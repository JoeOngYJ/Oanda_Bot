"""
Signal Generator - Coordinates multiple strategies to generate trade signals.
"""

from datetime import datetime
from typing import List, Dict
from shared.models import MarketTick, TradeSignal
import uuid


class SignalGenerator:
    """Coordinates multiple strategies to generate trade signals"""

    def __init__(self, strategies: Dict):
        self.strategies = strategies

    async def generate_signals(self, tick: MarketTick) -> List[TradeSignal]:
        """
        Pass market tick to all applicable strategies.
        Return list of generated signals.

        Args:
            tick: Market data tick

        Returns:
            List of TradeSignal objects
        """
        signals = []

        for strategy_name, strategy in self.strategies.items():
            # Check if strategy trades this instrument
            if tick.instrument not in strategy.instruments:
                continue

            # Update strategy with new tick
            strategy.update(tick)

            # Check for signal
            signal = strategy.check_signal(tick)

            if signal:
                # Enrich with metadata
                signal.signal_id = str(uuid.uuid4())
                signal.timestamp = datetime.utcnow()
                signals.append(signal)

        return signals
