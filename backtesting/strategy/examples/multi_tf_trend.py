# backtesting/strategy/examples/multi_tf_trend.py

from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection
from backtesting.data.models import OHLCVBar
from backtesting.core.timeframe import Timeframe
from backtesting.features.compute.cpu_engine import IndicatorEngine
from typing import Optional, Dict
import pandas as pd

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
        
        self._buffers[tf] = pd.concat([self._buffers[tf], new_row])
        self._buffers[tf] = self._buffers[tf].tail(self.ema_slow_period + 50)  # Keep extra
        
        # Only generate signals on base timeframe (M15)
        if tf != Timeframe.M15:
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
        
        # Check entry conditions on M15
        m15_close = self._buffers[Timeframe.M15]['close'].iloc[-1]
        m15_ema_fast = trends[Timeframe.M15]['ema_fast']
        
        if all_bullish and m15_close > m15_ema_fast:
            # Price above fast EMA in uptrend - LONG signal
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.LONG,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=m15_ema_fast * 0.998,  # 0.2% below EMA
                take_profit=bar.close * 1.015,  # 1.5% target
                timeframe=Timeframe.M15,
                confidence=0.8,
                metadata=trends
            )
        
        elif all_bearish and m15_close < m15_ema_fast:
            # Price below fast EMA in downtrend - SHORT signal
            return Signal(
                timestamp=bar.timestamp,
                instrument=bar.instrument,
                direction=SignalDirection.SHORT,
                strategy_name=self.name,
                entry_price=bar.close,
                stop_loss=m15_ema_fast * 1.002,
                take_profit=bar.close * 0.985,
                timeframe=Timeframe.M15,
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