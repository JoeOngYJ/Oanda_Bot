from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from oanda_bot.backtesting.core.engine import BacktestEngine
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection


class FlipStrategy(StrategyBase):
    """Emit one long then one short signal to force a close."""

    def __init__(self, config):
        super().__init__(config)
        self._seen = 0

    def on_bar(self, bar: OHLCVBar):
        if bar.timeframe != Timeframe.H1:
            return None
        self._seen += 1
        if self._seen == 1:
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
                quantity=10000,
            )
        if self._seen == 2:
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
                quantity=10000,
            )
        return None

    def get_required_warmup_bars(self):
        return {Timeframe.H1: 1}


def test_engine_applies_spread_and_commission_in_realized_trade():
    idx = pd.to_datetime(
        [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
        ]
    )
    df = pd.DataFrame(
        {
            "open": [1.1000, 1.1010, 1.1000],
            "high": [1.1015, 1.1020, 1.1010],
            "low": [1.0990, 1.1000, 1.0995],
            "close": [1.1000, 1.1010, 1.1005],
            "volume": [100, 100, 100],
        },
        index=idx,
    )

    cfg = {
        "data": {
            "instrument": "EUR_USD",
            "base_timeframe": Timeframe.H1,
            "start_date": idx.min().to_pydatetime(),
            "end_date": idx.max().to_pydatetime(),
        },
        "strategy": {
            "name": "flip",
            "class": FlipStrategy,
            "timeframes": [Timeframe.H1],
        },
        "execution": {
            "initial_capital": 10000,
            "slippage_pips": 0,
            "spreads_pips": {"EUR_USD": 1.4},
            "pricing_model": "oanda_core",
            "core_commission_per_10k_units": 1.0,
        },
        "data_dict": {Timeframe.H1: df},
    }

    result = BacktestEngine(cfg).run()

    assert result.total_trades == 1
    # Entry ask at 1.10007, exit bid at 1.10093 -> 8.6 gross for 10k units.
    # Commissions: 1.0 + 1.0 => net 6.6
    assert round(result.trades[0]["pnl"], 6) == 6.6
    assert round(result.total_fees_paid, 6) == 2.0
