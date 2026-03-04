"""Regime-based strategy router."""

from __future__ import annotations

from typing import Dict, Optional

from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal


class RegimeSwitchRouter(StrategyBase):
    """
    Wrap multiple strategies and route signal generation by current regime.
    """

    def __init__(self, config):
        super().__init__(config)
        self.regime_to_strategy: Dict[str, str] = {
            str(k): str(v) for k, v in config.get("regime_to_strategy", {}).items()
        }
        self.default_strategy_name: Optional[str] = config.get("default_strategy")
        self.active_regime: Optional[str] = None

        strategies_cfg = config.get("strategies", {})
        self.strategies: Dict[str, StrategyBase] = {}
        for name, scfg in strategies_cfg.items():
            cls = scfg["class"]
            self.strategies[name] = cls(scfg)
        if not self.strategies:
            raise ValueError("RegimeSwitchRouter requires non-empty strategies config")

    def set_regime(self, regime: Optional[str]) -> None:
        self.active_regime = None if regime is None else str(regime)

    def _active_strategy_name(self) -> str:
        if self.active_regime is not None:
            selected = self.regime_to_strategy.get(self.active_regime)
            if selected in self.strategies:
                return selected
        if self.default_strategy_name in self.strategies:
            return str(self.default_strategy_name)
        return next(iter(self.strategies.keys()))

    def on_market_bar(self, bar: OHLCVBar):
        for strategy in self.strategies.values():
            strategy.on_market_bar(bar)
        return None

    def on_bar(self, bar: OHLCVBar):
        strategy_name = self._active_strategy_name()
        signal: Optional[Signal] = self.strategies[strategy_name].on_bar(bar)
        if signal is None:
            return None
        signal.strategy_name = strategy_name
        meta = dict(signal.metadata or {})
        meta["regime"] = self.active_regime
        meta["selected_strategy"] = strategy_name
        signal.metadata = meta
        return signal

    def get_required_warmup_bars(self):
        merged = {}
        for strategy in self.strategies.values():
            req = strategy.get_required_warmup_bars()
            for tf, bars in req.items():
                merged[tf] = max(merged.get(tf, 0), bars)
        return merged
