"""Ensemble decision strategy with regime-aware weighted voting."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional

from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection


class RegimeEnsembleDecisionStrategy(StrategyBase):
    """
    Runs multiple strategy modules in parallel and decides with regime-aware voting.
    """

    def __init__(self, config):
        super().__init__(config)
        modules_cfg = config.get("modules", {})
        if not modules_cfg:
            raise ValueError("RegimeEnsembleDecisionStrategy requires non-empty modules.")

        self.modules: Dict[str, StrategyBase] = {}
        self.module_style: Dict[str, str] = {}
        self.module_weight: Dict[str, float] = {}
        for name, mcfg in modules_cfg.items():
            cls = mcfg["class"]
            self.modules[name] = cls(mcfg)
            self.module_style[name] = str(mcfg.get("style", "generic"))
            self.module_weight[name] = float(mcfg.get("weight", 1.0))

        self.regime_style_weights: Dict[str, Dict[str, float]] = {
            str(k): {str(sk): float(sv) for sk, sv in v.items()}
            for k, v in config.get("regime_style_weights", {}).items()
        }
        self.default_style_weights: Dict[str, float] = {
            str(k): float(v) for k, v in config.get("default_style_weights", {}).items()
        } or {
            "trend": 1.0,
            "mean_reversion": 1.0,
            "breakout": 1.0,
            "stat_arb": 1.0,
        }
        self.decision_threshold = float(config.get("decision_threshold", 0.25))
        self.min_votes = int(config.get("min_votes", 1))
        self.active_regime: Optional[str] = None
        self.regime_probabilities: Dict[str, float] = {}

    def set_regime(self, regime: Optional[str]) -> None:
        self.active_regime = None if regime is None else str(regime)

    def set_regime_probabilities(self, probs: Dict[str, float]) -> None:
        self.regime_probabilities = dict(probs or {})

    def on_market_bar(self, bar: OHLCVBar):
        for module in self.modules.values():
            module.on_market_bar(bar)
        return None

    def on_bar(self, bar: OHLCVBar):
        votes: List[Signal] = []
        score = 0.0

        for module_name, module in self.modules.items():
            sig = module.on_bar(bar)
            if sig is None:
                continue
            style = self.module_style.get(module_name, "generic")
            module_w = self.module_weight.get(module_name, 1.0)
            regime_w = self._style_weight_for_active_regime(style)
            direction = 1.0 if sig.direction == SignalDirection.LONG else -1.0
            conf = max(0.0, min(1.0, float(sig.confidence)))
            score += direction * conf * module_w * regime_w
            votes.append(sig)

        if len(votes) < self.min_votes:
            return None
        if abs(score) < self.decision_threshold:
            return None

        direction = SignalDirection.LONG if score > 0 else SignalDirection.SHORT
        winner = max(votes, key=lambda s: float(s.confidence))
        metadata = {
            "regime": self.active_regime,
            "regime_probabilities": self.regime_probabilities,
            "ensemble_score": score,
            "votes": [
                {
                    "strategy": v.strategy_name,
                    "direction": v.direction.value,
                    "confidence": float(v.confidence),
                }
                for v in votes
            ],
        }
        return Signal(
            timestamp=bar.timestamp,
            instrument=bar.instrument,
            direction=direction,
            strategy_name=self.name,
            entry_price=winner.entry_price if winner.entry_price else Decimal(str(bar.close)),
            stop_loss=winner.stop_loss,
            take_profit=winner.take_profit,
            timeframe=bar.timeframe,
            confidence=min(1.0, abs(score)),
            quantity=int(winner.quantity),
            metadata=metadata,
        )

    def _style_weight_for_active_regime(self, style: str) -> float:
        if self.active_regime is None:
            return float(self.default_style_weights.get(style, 1.0))
        per_regime = self.regime_style_weights.get(self.active_regime, {})
        return float(per_regime.get(style, self.default_style_weights.get(style, 1.0)))

    def get_required_warmup_bars(self):
        merged = {}
        for module in self.modules.values():
            req = module.get_required_warmup_bars()
            for tf, bars in req.items():
                merged[tf] = max(merged.get(tf, 0), bars)
        return merged
