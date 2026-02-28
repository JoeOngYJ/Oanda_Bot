"""EMA trend pullback strategy."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Deque, Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class EMATrendPullback(StrategyBase):
    """Trend-following strategy using EMA alignment + pullback entry."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.fast_period = int(config.get("fast_period", 50))
        self.slow_period = int(config.get("slow_period", 200))
        self.pullback_pct = Decimal(str(config.get("pullback_pct", 0.0008)))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.008)))
        self.quantity = int(config.get("quantity", 10000))

        self._closes: Deque[Decimal] = deque(maxlen=self.slow_period + 5)
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {self.signal_tf: self.slow_period}

    def _ema(self, values: Deque[Decimal], period: int) -> Decimal:
        alpha = Decimal("2") / Decimal(period + 1)
        ema = values[0]
        for v in list(values)[1:]:
            ema = (alpha * v) + ((Decimal("1") - alpha) * ema)
        return ema

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._closes.append(bar.close)
        if len(self._closes) < self.slow_period:
            return None

        ema_fast = self._ema(self._closes, self.fast_period)
        ema_slow = self._ema(self._closes, self.slow_period)
        price = bar.close

        bullish = ema_fast > ema_slow
        bearish = ema_fast < ema_slow

        # Enter after shallow pullback toward fast EMA.
        long_trigger = ema_fast * (Decimal("1") - self.pullback_pct)
        short_trigger = ema_fast * (Decimal("1") + self.pullback_pct)

        if bullish and price <= long_trigger and self._last_signal != SignalDirection.LONG:
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
                metadata={
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "pullback_pct": float(self.pullback_pct),
                },
            )

        if bearish and price >= short_trigger and self._last_signal != SignalDirection.SHORT:
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
                metadata={
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "pullback_pct": float(self.pullback_pct),
                },
            )

        return None
