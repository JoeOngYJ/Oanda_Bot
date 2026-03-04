from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from oanda_bot.backtesting.core.backtester import Backtester, FeatureEngineer, RegimePredictor, RiskManager
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.core.types import InstrumentSymbol
from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.execution.commission import OandaSpreadOnlyCommissionModel
from oanda_bot.backtesting.execution.simulator import ExecutionSimulator
from oanda_bot.backtesting.execution.slippage import SlippageModel
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection


class CountingFeatureEngineer(FeatureEngineer):
    def __init__(self):
        self.calls = 0

    def compute(self, bar: OHLCVBar, state):
        self.calls += 1
        return {"close": float(bar.close)}


class ConstantRegimePredictor(RegimePredictor):
    def __init__(self):
        self.calls = 0

    def predict(self, bar: OHLCVBar, features, state):
        self.calls += 1
        return "trend"


class PassThroughRiskManager(RiskManager):
    def __init__(self):
        self.calls = 0
        self.regimes = []

    def assess(self, signal: Signal, bar: OHLCVBar, portfolio, state):
        self.calls += 1
        self.regimes.append(signal.metadata.get("regime"))
        return signal


class FlipOnBarsStrategy(StrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self._count = 0

    def on_bar(self, bar: OHLCVBar):
        if bar.timeframe != Timeframe.H1:
            return None
        self._count += 1
        if self._count == 1:
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=None,
                take_profit=None,
                timeframe=Timeframe.H1,
                confidence=1.0,
                quantity=1000,
            )
        if self._count == 3:
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.SHORT,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=None,
                take_profit=None,
                timeframe=Timeframe.H1,
                confidence=1.0,
                quantity=1000,
            )
        return None

    def get_required_warmup_bars(self):
        return {Timeframe.H1: 1}


def test_execution_simulator_next_open_fill_mode():
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=SlippageModel(
            spread_pips_by_instrument={"EUR_USD": Decimal("0")},
            default_spread_pips=Decimal("0"),
            slippage_pips=Decimal("0"),
        ),
        commission_model=OandaSpreadOnlyCommissionModel(),
        fill_mode="next_open",
    )
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)

    signal = Signal(
        timestamp=t0,
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.LONG,
        strategy_name="s",
        entry_price=Decimal("1.1000"),
        stop_loss=None,
        take_profit=None,
        timeframe=Timeframe.H1,
        confidence=1.0,
        quantity=1000,
    )
    sim.process_signal(signal)

    bar0 = OHLCVBar(
        timestamp=t0,
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1000"),
        high=Decimal("1.1010"),
        low=Decimal("1.0990"),
        close=Decimal("1.1005"),
        volume=100,
    )
    sim.process_bar(bar0)
    assert len(sim.filled_orders) == 0

    bar1 = OHLCVBar(
        timestamp=t1,
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1010"),
        high=Decimal("1.1020"),
        low=Decimal("1.1000"),
        close=Decimal("1.1015"),
        volume=100,
    )
    sim.process_bar(bar1)
    assert len(sim.filled_orders) == 1
    assert sim.filled_orders[0]["fill_price"] == Decimal("1.1010")
    assert sim.filled_orders[0]["timestamp"] == t1


def test_backtester_runs_explicit_pipeline_stages():
    idx = pd.to_datetime(
        [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc),
        ]
    )
    df = pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.1020, 1.1030],
            "high": [1.1010, 1.1020, 1.1030, 1.1040],
            "low": [1.0990, 1.1000, 1.1010, 1.1020],
            "close": [1.1005, 1.1015, 1.1025, 1.1035],
            "volume": [100, 100, 100, 100],
        },
        index=idx,
    )

    feature_engineer = CountingFeatureEngineer()
    regime_predictor = ConstantRegimePredictor()
    risk_manager = PassThroughRiskManager()

    cfg = {
        "data": {
            "instrument": "EUR_USD",
            "base_timeframe": Timeframe.H1,
            "start_date": idx.min().to_pydatetime(),
            "end_date": idx.max().to_pydatetime(),
        },
        "strategy": {
            "name": "flip",
            "class": FlipOnBarsStrategy,
            "timeframes": [Timeframe.H1],
        },
        "execution": {
            "initial_capital": 10000,
            "slippage_pips": 0,
            "spreads_pips": {"EUR_USD": 0.0},
            "pricing_model": "spread_only",
            "fill_mode": "next_open",
        },
        "data_dict": {Timeframe.H1: df},
    }

    result = Backtester(
        context=cfg,
        feature_engineer=feature_engineer,
        regime_predictor=regime_predictor,
        risk_manager=risk_manager,
    ).run()

    assert feature_engineer.calls == 4
    assert regime_predictor.calls == 4
    assert risk_manager.calls == 2
    assert risk_manager.regimes == ["trend", "trend"]
    assert result.total_trades == 1
