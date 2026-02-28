from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

from backtesting.core.regime_runtime import KMeansRegimePredictor, RegimeModel
from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol
from backtesting.data.models import OHLCVBar
from backtesting.strategy.examples.regime_ensemble_decision import RegimeEnsembleDecisionStrategy
from backtesting.strategy.examples.regime_switch_router import RegimeSwitchRouter
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class AlwaysSignalStrategy(StrategyBase):
    def on_bar(self, bar: OHLCVBar):
        return Signal(
            timestamp=bar.timestamp,
            instrument=bar.instrument,
            direction=SignalDirection.LONG,
            strategy_name=self.name,
            entry_price=bar.close,
            stop_loss=None,
            take_profit=None,
            timeframe=bar.timeframe,
            confidence=1.0,
            quantity=1000,
        )

    def get_required_warmup_bars(self):
        return {Timeframe.H1: 1}


def test_kmeans_regime_predictor_nearest_center():
    model = RegimeModel(
        feature_columns=["x", "y"],
        train_mean={"x": 0.0, "y": 0.0},
        train_std={"x": 1.0, "y": 1.0},
        centers=np.array([[0.0, 0.0], [10.0, 10.0]]),
        regime_to_strategy={"0": "A", "1": "B"},
    )
    predictor = KMeansRegimePredictor(model)
    bar = OHLCVBar(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
        volume=1,
    )
    reg = predictor.predict(bar, {"x": 0.1, "y": 0.2}, {})
    assert reg == "0"
    assert set(predictor.last_probabilities.keys()) == {"0", "1"}
    assert abs(sum(predictor.last_probabilities.values()) - 1.0) < 1e-9


def test_regime_switch_router_routes_to_active_strategy():
    cfg = {
        "name": "router",
        "timeframes": [Timeframe.H1],
        "regime_to_strategy": {"0": "A", "1": "B"},
        "default_strategy": "A",
        "strategies": {
            "A": {"name": "A", "class": AlwaysSignalStrategy, "timeframes": [Timeframe.H1]},
            "B": {"name": "B", "class": AlwaysSignalStrategy, "timeframes": [Timeframe.H1]},
        },
    }
    router = RegimeSwitchRouter(cfg)
    router.set_regime("1")

    bar = OHLCVBar(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1"),
        high=Decimal("1.2"),
        low=Decimal("1.0"),
        close=Decimal("1.15"),
        volume=10,
    )
    signal = router.on_bar(bar)
    assert signal is not None
    assert signal.strategy_name == "B"
    assert signal.metadata["selected_strategy"] == "B"
    assert signal.metadata["regime"] == "1"


def test_regime_ensemble_decision_emits_weighted_vote_signal():
    cfg = {
        "name": "ensemble",
        "timeframes": [Timeframe.H1],
        "decision_threshold": 0.1,
        "min_votes": 1,
        "regime_style_weights": {"0": {"trend": 2.0, "mean_reversion": 0.3, "breakout": 1.0}},
        "modules": {
            "trend_a": {
                "name": "trend_a",
                "class": AlwaysSignalStrategy,
                "timeframes": [Timeframe.H1],
                "style": "trend",
                "weight": 1.0,
            },
            "trend_b": {
                "name": "trend_b",
                "class": AlwaysSignalStrategy,
                "timeframes": [Timeframe.H1],
                "style": "trend",
                "weight": 0.8,
            },
        },
    }
    s = RegimeEnsembleDecisionStrategy(cfg)
    s.set_regime("0")
    s.set_regime_probabilities({"0": 0.9, "1": 0.1})
    bar = OHLCVBar(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1"),
        high=Decimal("1.2"),
        low=Decimal("1.0"),
        close=Decimal("1.15"),
        volume=10,
    )
    sig = s.on_bar(bar)
    assert sig is not None
    assert sig.direction == SignalDirection.LONG
    assert "ensemble_score" in sig.metadata
    assert sig.metadata["regime"] == "0"
