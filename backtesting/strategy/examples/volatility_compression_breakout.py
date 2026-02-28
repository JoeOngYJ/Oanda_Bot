"""Volatility compression then expansion breakout strategy."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from statistics import mean
from typing import Deque, Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class VolatilityCompressionBreakout(StrategyBase):
    """Triggers when ATR is compressed and price breaks recent range."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.range_lookback = int(config.get("range_lookback", 20))
        self.atr_period = int(config.get("atr_period", 14))
        self.compression_window = int(config.get("compression_window", 40))
        self.compression_ratio = float(config.get("compression_ratio", 0.75))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.01)))
        self.quantity = int(config.get("quantity", 10000))

        self._highs: Deque[Decimal] = deque(maxlen=self.range_lookback + 1)
        self._lows: Deque[Decimal] = deque(maxlen=self.range_lookback + 1)
        self._true_ranges: Deque[float] = deque(maxlen=max(self.compression_window, self.atr_period))
        self._prev_close: Optional[Decimal] = None
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {
            self.signal_tf: max(self.range_lookback + 1, self.compression_window, self.atr_period + 2)
        }

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if self._prev_close is not None:
            tr = max(
                float(bar.high - bar.low),
                abs(float(bar.high - self._prev_close)),
                abs(float(bar.low - self._prev_close)),
            )
            self._true_ranges.append(tr)
        self._prev_close = bar.close

        if len(self._highs) < self.range_lookback + 1 or len(self._true_ranges) < self.compression_window:
            return None

        atr_recent = mean(list(self._true_ranges)[-self.atr_period :])
        atr_long = mean(self._true_ranges)
        compressed = atr_recent <= (self.compression_ratio * atr_long)

        if not compressed:
            return None

        prev_high = max(list(self._highs)[:-1])
        prev_low = min(list(self._lows)[:-1])
        price = bar.close

        if price > prev_high and self._last_signal != SignalDirection.LONG:
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
                confidence=0.74,
                quantity=self.quantity,
                metadata={"atr_recent": atr_recent, "atr_long": atr_long},
            )

        if price < prev_low and self._last_signal != SignalDirection.SHORT:
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
                confidence=0.74,
                quantity=self.quantity,
                metadata={"atr_recent": atr_recent, "atr_long": atr_long},
            )

        return None
