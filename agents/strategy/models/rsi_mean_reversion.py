"""
RSI Mean Reversion Strategy.

LONG signal: RSI crosses below oversold threshold (e.g., 30)
SHORT signal: RSI crosses above overbought threshold (e.g., 70)

Mean reversion: bet on price reversing to mean.
"""

from collections import deque
from decimal import Decimal
from typing import Optional, Dict
from shared.models import TradeSignal, MarketTick, Side, Instrument
from .base_strategy import BaseStrategy
from ..indicators import rsi


class RSIMeanReversion(BaseStrategy):
    """
    RSI-based mean reversion strategy.

    Generates signals when RSI crosses overbought/oversold thresholds.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        params = config['parameters']
        self.rsi_period = params['rsi_period']
        self.oversold_threshold = params['oversold_threshold']
        self.overbought_threshold = params['overbought_threshold']
        self.position_size = params['position_size']
        self.stop_loss_pips = params['stop_loss_pips']
        self.take_profit_pips = params['take_profit_pips']

        # Price history for RSI calculation
        max_len = self.rsi_period + 3
        self.price_history: Dict[Instrument, deque] = {
            instrument: deque(maxlen=max_len)
            for instrument in self.instruments
        }

        # Track last RSI to detect crossovers
        self.last_rsi: Dict[Instrument, Optional[float]] = {
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

        # Need enough history for RSI crossover detection
        if len(prices) < self.rsi_period + 2:
            return None

        # Calculate current and previous RSI
        current_rsi = rsi(prices, self.rsi_period)
        prev_rsi = rsi(prices[:-1], self.rsi_period)

        if current_rsi is None or prev_rsi is None:
            return None

        # Update last RSI
        self.last_rsi[tick.instrument] = current_rsi

        signal_side = None

        # Detect oversold crossover (mean reversion long)
        if prev_rsi >= self.oversold_threshold and current_rsi < self.oversold_threshold:
            signal_side = Side.BUY

        # Detect overbought crossover (mean reversion short)
        elif prev_rsi <= self.overbought_threshold and current_rsi > self.overbought_threshold:
            signal_side = Side.SELL

        # No crossover
        if signal_side is None:
            return None

        # Calculate confidence (distance from threshold)
        if signal_side == Side.BUY:
            distance = self.oversold_threshold - current_rsi
            confidence = min(distance / 30, 1.0)  # Scale to 0-1
        else:
            distance = current_rsi - self.overbought_threshold
            confidence = min(distance / 30, 1.0)

        # Ensure minimum confidence
        confidence = max(confidence, 0.5)

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
            rationale=f"RSI mean reversion: RSI={current_rsi:.2f} crossed threshold",
            strategy_name=self.name,
            strategy_version=self.version,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=tick.timestamp,
            metadata={
                "current_rsi": current_rsi,
                "prev_rsi": prev_rsi,
                "threshold": self.oversold_threshold if signal_side == Side.BUY else self.overbought_threshold
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
            "last_rsi": {
                i.value: self.last_rsi[i]
                for i in self.instruments
            }
        }
