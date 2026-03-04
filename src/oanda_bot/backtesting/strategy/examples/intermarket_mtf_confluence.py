"""Intermarket + multi-timeframe confluence strategy."""

from __future__ import annotations

from collections import defaultdict, deque
from decimal import Decimal
from typing import Deque, Dict, List, Optional

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection


class IntermarketMTFConfluence(StrategyBase):
    """
    Strategy design:
    - Primary pair trend must align on higher timeframes.
    - Reference pairs must show majority trend agreement.
    - Entry happens on smallest configured timeframe of the primary pair.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.primary_instrument = str(config["primary_instrument"])
        self.reference_instruments: List[str] = [str(x) for x in config.get("reference_instruments", [])]
        self.ema_fast = int(config.get("ema_fast", 50))
        self.ema_slow = int(config.get("ema_slow", 200))
        self.min_ref_alignment = float(config.get("min_ref_alignment", 0.5))
        self.relative_strength_lookback = int(config.get("relative_strength_lookback", 12))
        self.relative_strength_min = float(config.get("relative_strength_min", 0.0002))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.008)))
        self.quantity = int(config.get("quantity", 10000))
        self.cooldown_bars = int(config.get("cooldown_bars", 8))

        self.entry_timeframe = min(self.timeframes, key=lambda tf: tf.seconds)
        self.confirm_timeframes = [tf for tf in self.timeframes if tf != self.entry_timeframe]
        if not self.confirm_timeframes:
            self.confirm_timeframes = [self.entry_timeframe]

        self._closes: Dict[str, Dict[Timeframe, Deque[Decimal]]] = defaultdict(
            lambda: {tf: deque(maxlen=max(self.ema_slow + 10, self.relative_strength_lookback + 10)) for tf in self.timeframes}
        )
        self._last_signal: Optional[SignalDirection] = None
        self._bars_since_signal = self.cooldown_bars

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        return {tf: max(self.ema_slow, self.relative_strength_lookback + 2) for tf in self.timeframes}

    @staticmethod
    def _ema(values: Deque[Decimal], period: int) -> Decimal:
        alpha = Decimal("2") / Decimal(period + 1)
        ema = values[0]
        for v in list(values)[1:]:
            ema = (alpha * v) + ((Decimal("1") - alpha) * ema)
        return ema

    def _trend_direction(self, instrument: str, timeframe: Timeframe) -> Optional[int]:
        closes = self._closes[instrument][timeframe]
        if len(closes) < self.ema_slow:
            return None
        fast = self._ema(closes, self.ema_fast)
        slow = self._ema(closes, self.ema_slow)
        if fast > slow:
            return 1
        if fast < slow:
            return -1
        return 0

    def _relative_strength(self) -> Optional[float]:
        primary = self._closes[self.primary_instrument][self.entry_timeframe]
        if len(primary) < self.relative_strength_lookback + 1:
            return None
        p_now = float(primary[-1])
        p_prev = float(primary[-(self.relative_strength_lookback + 1)])
        if p_prev == 0:
            return None
        p_ret = (p_now / p_prev) - 1.0

        ref_rets = []
        for ref in self.reference_instruments:
            closes = self._closes[ref][self.entry_timeframe]
            if len(closes) < self.relative_strength_lookback + 1:
                continue
            r_now = float(closes[-1])
            r_prev = float(closes[-(self.relative_strength_lookback + 1)])
            if r_prev == 0:
                continue
            ref_rets.append((r_now / r_prev) - 1.0)
        if not ref_rets:
            return None
        return p_ret - (sum(ref_rets) / len(ref_rets))

    def on_market_bar(self, bar: OHLCVBar):
        if bar.timeframe not in self.timeframes:
            return None
        self._closes[str(bar.instrument)][bar.timeframe].append(bar.close)
        return None

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if str(bar.instrument) != self.primary_instrument:
            return None
        if bar.timeframe != self.entry_timeframe:
            return None

        self._bars_since_signal += 1
        if self._bars_since_signal < self.cooldown_bars:
            return None

        primary_trends = []
        for tf in self.confirm_timeframes:
            d = self._trend_direction(self.primary_instrument, tf)
            if d is None:
                return None
            primary_trends.append(d)
        if not primary_trends:
            return None
        if not all(x == primary_trends[0] for x in primary_trends):
            return None
        primary_dir = primary_trends[0]
        if primary_dir == 0:
            return None

        # Reference alignment on same confirm timeframes.
        agree = 0
        total = 0
        for ref in self.reference_instruments:
            ref_votes = []
            for tf in self.confirm_timeframes:
                d = self._trend_direction(ref, tf)
                if d is None:
                    continue
                ref_votes.append(d)
            if not ref_votes:
                continue
            ref_dir = 1 if sum(ref_votes) > 0 else -1 if sum(ref_votes) < 0 else 0
            if ref_dir == 0:
                continue
            total += 1
            if ref_dir == primary_dir:
                agree += 1
        if total == 0:
            return None
        alignment = agree / total
        if alignment < self.min_ref_alignment:
            return None

        rel = self._relative_strength()
        if rel is None:
            return None
        if primary_dir == 1 and rel < self.relative_strength_min:
            return None
        if primary_dir == -1 and rel > -self.relative_strength_min:
            return None

        direction = SignalDirection.LONG if primary_dir == 1 else SignalDirection.SHORT
        if direction == self._last_signal:
            return None
        self._last_signal = direction
        self._bars_since_signal = 0

        price = bar.close
        if direction == SignalDirection.LONG:
            stop_loss = price * (Decimal("1") - self.stop_loss_pct)
            take_profit = price * (Decimal("1") + self.take_profit_pct)
        else:
            stop_loss = price * (Decimal("1") + self.stop_loss_pct)
            take_profit = price * (Decimal("1") - self.take_profit_pct)

        return Signal(
            timestamp=bar.timestamp,
            instrument=bar.instrument,
            direction=direction,
            strategy_name=self.name,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=self.entry_timeframe,
            confidence=min(1.0, 0.6 + (0.4 * alignment)),
            quantity=self.quantity,
            metadata={
                "primary_dir": primary_dir,
                "ref_alignment": alignment,
                "relative_strength": rel,
                "reference_instruments": self.reference_instruments,
            },
        )
