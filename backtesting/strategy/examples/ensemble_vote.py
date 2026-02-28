"""Ensemble strategy that votes across component strategies."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class EnsembleVoteStrategy(StrategyBase):
    """Emit signal only when enough component strategies align direction."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.signal_tf = self.timeframes[0]
        self.min_votes = int(config.get("min_votes", 2))
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.004)))
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.007)))
        self.quantity = int(config.get("quantity", 10000))
        self.cooldown_bars = int(config.get("cooldown_bars", 5))
        self._bars_since_signal = self.cooldown_bars
        self._last_signal: Optional[SignalDirection] = None

        component_cfgs = list(config.get("components", []))
        if not component_cfgs:
            raise ValueError("EnsembleVoteStrategy requires non-empty `components`.")

        self.components: List[StrategyBase] = []
        for idx, comp in enumerate(component_cfgs):
            comp_cls = comp["class"]
            comp_cfg = dict(comp)
            comp_cfg.pop("class", None)
            comp_cfg["name"] = comp_cfg.get("name", f"{self.name}_C{idx + 1}_{comp_cls.__name__}")
            comp_cfg["timeframes"] = comp_cfg.get("timeframes", [self.signal_tf])
            self.components.append(comp_cls(comp_cfg))

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        warmups = [c.get_required_warmup_bars().get(self.signal_tf, 0) for c in self.components]
        return {self.signal_tf: max(warmups) if warmups else 0}

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.signal_tf:
            for comp in self.components:
                comp.on_bar(bar)
            return None

        self._bars_since_signal += 1
        long_votes = 0
        short_votes = 0
        voters: List[str] = []

        for comp in self.components:
            signal = comp.on_bar(bar)
            if signal is None:
                continue
            voters.append(comp.name)
            if signal.direction == SignalDirection.LONG:
                long_votes += 1
            elif signal.direction == SignalDirection.SHORT:
                short_votes += 1

        if self._bars_since_signal < self.cooldown_bars:
            return None

        direction: Optional[SignalDirection] = None
        confidence = 0.0
        if long_votes >= self.min_votes and long_votes > short_votes:
            direction = SignalDirection.LONG
            confidence = min(1.0, long_votes / len(self.components))
        elif short_votes >= self.min_votes and short_votes > long_votes:
            direction = SignalDirection.SHORT
            confidence = min(1.0, short_votes / len(self.components))

        if direction is None or direction == self._last_signal:
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
            timeframe=self.signal_tf,
            confidence=confidence,
            quantity=self.quantity,
            metadata={
                "long_votes": long_votes,
                "short_votes": short_votes,
                "voters": voters,
            },
        )
