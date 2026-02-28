from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtesting.core.backtester import Backtester
from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


class OneTradeStrategy(StrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self._count = 0

    def on_bar(self, bar: OHLCVBar):
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
                timeframe=bar.timeframe,
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
                timeframe=bar.timeframe,
                confidence=1.0,
                quantity=1000,
            )
        return None

    def get_required_warmup_bars(self):
        return {Timeframe.H1: 1}


def test_system_state_tracks_positions_and_buffers(tmp_path: Path):
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
    snap_path = tmp_path / "state_snapshots.jsonl"

    cfg = {
        "data": {
            "instrument": "EUR_USD",
            "base_timeframe": Timeframe.H1,
            "start_date": idx.min().to_pydatetime(),
            "end_date": idx.max().to_pydatetime(),
        },
        "strategy": {
            "name": "one_trade",
            "class": OneTradeStrategy,
            "timeframes": [Timeframe.H1],
        },
        "execution": {
            "initial_capital": 10000,
            "slippage_pips": 0,
            "spreads_pips": {"EUR_USD": 0.0},
            "pricing_model": "spread_only",
            "fill_mode": "next_open",
        },
        "state": {
            "snapshot_path": str(snap_path),
            "snapshot_every_bars": 1,
        },
        "data_dict": {Timeframe.H1: df},
    }

    bt = Backtester(context=cfg)
    result = bt.run()
    state = bt.latest_state
    assert state is not None
    assert state.bar_count == len(df)
    assert len(state.market_data_buffer) == len(df)
    assert len(state.feature_buffer) == len(df)
    assert len(state.equity_curve) >= len(df)
    assert len(state.trade_history) >= 2  # one entry + one exit fill
    assert abs(float(state.total_equity) - result.final_equity) < 1e-9
    assert snap_path.exists()
    lines = snap_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(df)
