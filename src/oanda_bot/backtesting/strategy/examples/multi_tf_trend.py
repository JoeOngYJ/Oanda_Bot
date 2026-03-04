# backtesting/strategy/examples/multi_tf_trend.py

from typing import Dict, Optional

import pandas as pd

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.features.compute.cpu_engine import IndicatorEngine
from oanda_bot.backtesting.strategy.base import StrategyBase
from oanda_bot.backtesting.strategy.signal import Signal, SignalDirection

class MultiTimeframeTrendStrategy(StrategyBase):
    """
    Strategy Logic:
    - D1: Identify major trend (50 EMA vs 200 EMA)
    - H4: Confirm trend alignment
    - H1: Entry structure (pullback to EMA)
    - M15: Precise entry on break of structure
    
    Only enter when all timeframes aligned.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Strategy parameters
        self.ema_fast_period = config.get('ema_fast', 50)
        self.ema_slow_period = config.get('ema_slow', 200)
        
        # Data buffers (store recent bars per timeframe)
        self._buffers: Dict[Timeframe, pd.DataFrame] = {
            tf: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            for tf in self.timeframes
        }
        
        # Indicator engine
        self.engine = IndicatorEngine()
        self.entry_timeframe = min(self.timeframes, key=lambda tf: tf.seconds)
    
    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        """Need 200 bars for slow EMA on all timeframes"""
        return {tf: self.ema_slow_period for tf in self.timeframes}
    
    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        """Process new bar and generate signal if conditions met"""
        
        # Update buffer for this timeframe
        tf = bar.timeframe
        new_row = pd.DataFrame([{
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': bar.volume
        }], index=[bar.timestamp])
        
        buf = self._buffers[tf]
        if buf.empty:
            self._buffers[tf] = new_row
        else:
            merged = pd.concat([buf, new_row], axis=0)
            # Keep only latest rows and de-duplicate timestamps to avoid pandas concat warnings.
            self._buffers[tf] = merged[~merged.index.duplicated(keep="last")].tail(self.ema_slow_period + 50)
        
        # Only generate signals on the execution timeframe (smallest configured timeframe).
        if tf != self.entry_timeframe:
            return None
        
        # Check if we have enough data
        if not self._has_sufficient_data():
            return None
        
        # Calculate indicators for all timeframes
        trends = {}
        for timeframe in self.timeframes:
            df = self._buffers[timeframe]
            close = df['close']
            
            ema_fast = self.engine.ema(close, self.ema_fast_period)
            ema_slow = self.engine.ema(close, self.ema_slow_period)
            
            trends[timeframe] = {
                'bullish': ema_fast.iloc[-1] > ema_slow.iloc[-1],
                'ema_fast': ema_fast.iloc[-1],
                'ema_slow': ema_slow.iloc[-1]
            }
        
        # Check alignment
        all_bullish = all(t['bullish'] for t in trends.values())
        all_bearish = all(not t['bullish'] for t in trends.values())
        
        if not (all_bullish or all_bearish):
            return None  # No alignment
        
        # Check entry conditions on execution timeframe.
        entry_close = self._buffers[self.entry_timeframe]["close"].iloc[-1]
        entry_ema_fast = trends[self.entry_timeframe]["ema_fast"]
        
        if all_bullish and entry_close > entry_ema_fast:
            # Price above fast EMA in uptrend - LONG signal
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=entry_ema_fast * 0.998,  # 0.2% below EMA
                take_profit=bar.close * 1.015,  # 1.5% target
                timeframe=self.entry_timeframe,
                confidence=0.8,
                metadata=trends
            )

        elif all_bearish and entry_close < entry_ema_fast:
            # Price below fast EMA in downtrend - SHORT signal
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.SHORT,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=entry_ema_fast * 1.002,
                take_profit=bar.close * 0.985,
                timeframe=self.entry_timeframe,
                confidence=0.8,
                metadata=trends
            )
        
        return None
    
    def _has_sufficient_data(self) -> bool:
        """Check if all buffers have enough bars"""
        return all(
            len(self._buffers[tf]) >= self.ema_slow_period
            for tf in self.timeframes
        )
