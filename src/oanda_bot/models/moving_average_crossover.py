"""
Moving Average Crossover Strategy.

LONG signal: Fast MA crosses above slow MA
SHORT signal: Fast MA crosses below slow MA

All logic is deterministic and testable.
"""

from collections import deque
from decimal import Decimal
from typing import Optional, Dict
from oanda_bot.utils.models import TradeSignal, MarketTick, Side, Instrument
from .base_strategy import BaseStrategy
from oanda_bot.agents.strategy.indicators import sma


class MovingAverageCrossover(BaseStrategy):
    """
    Simple moving average crossover strategy.

    Generates signals when fast MA crosses slow MA.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        params = config['parameters']
        self.fast_period = params['fast_period']
        self.slow_period = params['slow_period']
        self.signal_threshold = params['signal_threshold']
        self.position_size = params['position_size']
        self.stop_loss_pips = params['stop_loss_pips']
        self.take_profit_pips = params['take_profit_pips']

        # Price history (store mid prices)
        max_len = self.slow_period + 2
        self.price_history: Dict[Instrument, deque] = {
            instrument: deque(maxlen=max_len)
            for instrument in self.instruments
        }

        # Track last signal to avoid duplicates
        self.last_signal_side: Dict[Instrument, Optional[Side]] = {
            instrument: None for instrument in self.instruments
        }

    def update(self, tick: MarketTick) -> None:
        """Update price history with new tick"""
        mid_price = (tick.bid + tick.ask) / 2
        self.price_history[tick.instrument].append(float(mid_price))

    def check_signal(self, tick: MarketTick) -> Optional[TradeSignal]:
        """
        Check if current state generates a signal.

        Returns TradeSignal if conditions met, else None.
        """
        prices = list(self.price_history[tick.instrument])

        # Need enough history for crossover detection
        if len(prices) < self.slow_period + 1:
            return None

        # Calculate current moving averages
        fast_ma = sma(prices, self.fast_period)
        slow_ma = sma(prices, self.slow_period)

        if fast_ma is None or slow_ma is None:
            return None

        # Calculate previous moving averages
        prev_fast_ma = sma(prices[:-1], self.fast_period)
        prev_slow_ma = sma(prices[:-1], self.slow_period)

        if prev_fast_ma is None or prev_slow_ma is None:
            return None

        # Detect crossover
        signal_side = None

        # Bullish crossover: fast crosses above slow
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            signal_side = Side.BUY

        # Bearish crossover: fast crosses below slow
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            signal_side = Side.SELL

        # No crossover
        if signal_side is None:
            return None

        # Avoid duplicate signals
        if signal_side == self.last_signal_side[tick.instrument]:
            return None

        self.last_signal_side[tick.instrument] = signal_side

        # Calculate confidence (based on MA separation)
        ma_diff_pct = abs((fast_ma - slow_ma) / slow_ma)
        confidence = min(ma_diff_pct * 10, 1.0)  # Scale to 0-1

        # Only signal if confidence exceeds threshold
        if confidence < self.signal_threshold:
            return None

        # Calculate entry price and stop/take profit
        pip_value = Decimal("0.0001")  # For most majors

        if signal_side == Side.BUY:
            entry_price = tick.ask
            stop_loss = entry_price - (pip_value * self.stop_loss_pips)
            take_profit = entry_price + (pip_value * self.take_profit_pips)
        else:
            entry_price = tick.bid
            stop_loss = entry_price + (pip_value * self.stop_loss_pips)
            take_profit = entry_price - (pip_value * self.take_profit_pips)

        # Generate signal
        return TradeSignal(
            signal_id="",  # Will be set by signal generator
            instrument=tick.instrument,
            side=signal_side,
            quantity=self.position_size,
            confidence=confidence,
            rationale=f"MA crossover: fast={fast_ma:.5f} slow={slow_ma:.5f}",
            strategy_name=self.name,
            strategy_version=self.version,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=tick.timestamp,
            metadata={
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "ma_diff_pct": ma_diff_pct
            }
        )

    def get_state(self) -> dict:
        """Return current strategy state"""
        return {
            "name": self.name,
            "version": self.version,
            "instruments": [i.value for i in self.instruments],
            "price_history_lengths": {
                i.value: len(self.price_history[i])
                for i in self.instruments
            },
            "last_signals": {
                i.value: self.last_signal_side[i].value if self.last_signal_side[i] else None
                for i in self.instruments
            }
        }
