"""RSI + Bollinger mean-reversion strategy."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from statistics import pstdev
from typing import Deque, Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class RSIBollingerReversion(StrategyBase):
    """Reversion strategy requiring both band extension and RSI extreme."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.window = int(config.get("window", 20))
        self.std_mult = float(config.get("std_mult", 2.0))
        self.rsi_period = int(config.get("rsi_period", 14))
        self.rsi_overbought = float(config.get("rsi_overbought", 70))
        self.rsi_oversold = float(config.get("rsi_oversold", 30))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.003)))
        self.quantity = int(config.get("quantity", 10000))

        self._closes: Deque[Decimal] = deque(maxlen=max(self.window + 2, self.rsi_period + 2))
        self._last_signal: Optional[SignalDirection] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {self.signal_tf: max(self.window, self.rsi_period) + 1}

    def _rsi(self) -> float:
        closes = [float(v) for v in self._closes]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        period = self.rsi_period
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            return None

        self._closes.append(bar.close)
        if len(self._closes) < max(self.window, self.rsi_period) + 1:
            return None

        win = [float(v) for v in list(self._closes)[-self.window:]]
        mean = sum(win) / len(win)
        std = pstdev(win)
        upper = mean + self.std_mult * std
        lower = mean - self.std_mult * std
        price = float(bar.close)
        rsi = self._rsi()

        if price <= lower and rsi <= self.rsi_oversold and self._last_signal != SignalDirection.LONG:
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
                confidence=0.68,
                quantity=self.quantity,
                metadata={"rsi": rsi, "bb_lower": lower, "bb_upper": upper},
            )

        if price >= upper and rsi >= self.rsi_overbought and self._last_signal != SignalDirection.SHORT:
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
                confidence=0.68,
                quantity=self.quantity,
                metadata={"rsi": rsi, "bb_lower": lower, "bb_upper": upper},
            )

        return None
