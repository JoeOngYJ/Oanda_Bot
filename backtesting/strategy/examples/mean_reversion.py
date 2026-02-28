"""Simple mean-reversion strategy for backtesting research."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class MeanReversion(StrategyBase):
    """SMA deviation mean-reversion strategy on one timeframe."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.sma_period = int(config.get("sma_period", 30))
        self.deviation_pct = Decimal(str(config.get("deviation_pct", 0.0025)))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.003)))
        self.quantity = int(config.get("quantity", 10000))

        self._closes = deque(maxlen=self.sma_period)
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {self.signal_tf: self.sma_period}

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._closes.append(bar.close)
        if len(self._closes) < self.sma_period:
            return None

        sma = sum(self._closes) / Decimal(len(self._closes))
        upper = sma * (Decimal("1") + self.deviation_pct)
        lower = sma * (Decimal("1") - self.deviation_pct)
        price = bar.close

        if price < lower and self._last_signal != SignalDirection.LONG:
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
                confidence=0.65,
                quantity=self.quantity,
                metadata={"sma": float(sma), "sma_period": self.sma_period},
            )

        if price > upper and self._last_signal != SignalDirection.SHORT:
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
                confidence=0.65,
                quantity=self.quantity,
                metadata={"sma": float(sma), "sma_period": self.sma_period},
            )

        return None
