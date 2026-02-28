"""Simple breakout strategy for backtesting research."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class Breakout(StrategyBase):
    """Donchian-style breakout on a single timeframe."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.lookback = int(config.get("lookback", 20))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.008)))
        self.quantity = int(config.get("quantity", 10000))
        self.min_breakout_pct = Decimal(str(config.get("min_breakout_pct", 0.0002)))

        self._highs = deque(maxlen=self.lookback + 1)
        self._lows = deque(maxlen=self.lookback + 1)
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {self.signal_tf: self.lookback + 1}

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._highs.append(bar.high)
        self._lows.append(bar.low)
        if len(self._highs) < self.lookback + 1:
            return None

        # Use previous window to avoid lookahead.
        prev_high = max(list(self._highs)[:-1])
        prev_low = min(list(self._lows)[:-1])
        price = bar.close

        long_threshold = prev_high * (Decimal("1") + self.min_breakout_pct)
        short_threshold = prev_low * (Decimal("1") - self.min_breakout_pct)

        if price > long_threshold and self._last_signal != SignalDirection.LONG:
            self._last_signal = SignalDirection.LONG
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                entry_price=price,
                stop_loss=price * (Decimal("1") - self.stop_loss_pct),
                take_profit=price * (Decimal("1") + self.take_profit_pct),
                timeframe=self.signal_tf,
                confidence=0.7,
                quantity=self.quantity,
                metadata={"prev_high": float(prev_high), "lookback": self.lookback},
            )

        if price < short_threshold and self._last_signal != SignalDirection.SHORT:
            self._last_signal = SignalDirection.SHORT
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.SHORT,
                strategy_name=self.name,
                entry_price=price,
                stop_loss=price * (Decimal("1") + self.stop_loss_pct),
                take_profit=price * (Decimal("1") - self.take_profit_pct),
                timeframe=self.signal_tf,
                confidence=0.7,
                quantity=self.quantity,
                metadata={"prev_low": float(prev_low), "lookback": self.lookback},
            )

        return None
