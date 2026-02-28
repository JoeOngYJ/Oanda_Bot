# backtesting/strategy/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from backtesting.data.models import OHLCVBar
from backtesting.strategy.signal import Signal
from backtesting.core.timeframe import Timeframe

class StrategyBase(ABC):
    """
    Abstract base class for all strategies.
    
    Design Philosophy:
    - Strategies are stateful but should not mutate external state
    - They receive bar updates and emit signals
    - They do NOT execute trades directly
    - They can subscribe to multiple timeframes
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        raw_timeframes = config.get('timeframes', ['H1'])
        self.timeframes = [
            tf if isinstance(tf, Timeframe) else Timeframe[tf]
            for tf in raw_timeframes
        ]
        
        # Internal state (strategy-specific)
        self._state: Dict = {}
    
    @abstractmethod
    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        """
        Called when a new bar closes on ANY subscribed timeframe.
        
        Args:
            bar: The newly closed bar
        
        Returns:
            Signal if conditions met, else None
        """
        pass
    
    @abstractmethod
    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        """
        Return the number of historical bars needed per timeframe
        before strategy can generate signals.
        
        Example:
            {Timeframe.H1: 200, Timeframe.D1: 50}
            (Need 200 H1 bars for SMA-200, 50 D1 bars for trend)
        """
        pass
    
    def on_backtest_start(self):
        """Called once at backtest initialization (optional override)"""
        pass
    
    def on_backtest_end(self):
        """Called once at backtest completion (optional override)"""
        pass

    def on_market_bar(self, bar: OHLCVBar):
        """
        Optional hook for strategies that need cross-instrument context.

        Called for every market bar fed into the engine. Default no-op.
        """
        return None
    
    def get_state(self) -> Dict:
        """Return internal state for debugging/logging"""
        return self._state.copy()
