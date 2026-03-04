"""ATR-normalized breakout strategy."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from statistics import mean
from typing import Deque, Dict, Optional

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection


class ATRBreakout(StrategyBase):
    """Breakout strategy that requires move size to exceed ATR threshold."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.lookback = int(config.get("lookback", 30))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_mult = float(config.get("atr_mult", 1.2))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.0045)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.009)))
        self.quantity = int(config.get("quantity", 10000))

        self._highs: Deque[Decimal] = deque(maxlen=self.lookback + 2)
        self._lows: Deque[Decimal] = deque(maxlen=self.lookback + 2)
        self._closes: Deque[Decimal] = deque(maxlen=self.atr_period + 2)
        self._true_ranges: Deque[float] = deque(maxlen=self.atr_period)
        self._prev_close: Optional[Decimal] = None
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {self.signal_tf: max(self.lookback + 1, self.atr_period + 2)}

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._closes.append(bar.close)

        if self._prev_close is not None:
            tr = max(
                float(bar.high - bar.low),
                abs(float(bar.high - self._prev_close)),
                abs(float(bar.low - self._prev_close)),
            )
            self._true_ranges.append(tr)
        self._prev_close = bar.close

        if len(self._highs) < self.lookback + 1 or len(self._true_ranges) < self.atr_period:
            return None

        prev_high = max(list(self._highs)[:-1])
        prev_low = min(list(self._lows)[:-1])
        atr = mean(self._true_ranges)
        price = float(bar.close)
        bullish_break = price > float(prev_high) + (self.atr_mult * atr)
        bearish_break = price < float(prev_low) - (self.atr_mult * atr)

        if bullish_break and self._last_signal != SignalDirection.LONG:
            self._last_signal = SignalDirection.LONG
            p = bar.close
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                entry_price=p,
                stop_loss=p * (Decimal("1") - self.stop_loss_pct),
                take_profit=p * (Decimal("1") + self.take_profit_pct),
                timeframe=self.signal_tf,
                confidence=0.72,
                quantity=self.quantity,
                metadata={"atr": atr, "atr_mult": self.atr_mult},
            )

        if bearish_break and self._last_signal != SignalDirection.SHORT:
            self._last_signal = SignalDirection.SHORT
            p = bar.close
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.SHORT,
                strategy_name=self.name,
                entry_price=p,
                stop_loss=p * (Decimal("1") + self.stop_loss_pct),
                take_profit=p * (Decimal("1") - self.take_profit_pct),
                timeframe=self.signal_tf,
                confidence=0.72,
                quantity=self.quantity,
                metadata={"atr": atr, "atr_mult": self.atr_mult},
            )

        return None
