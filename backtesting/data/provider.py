# backtesting/data/provider.py

from typing import Dict, Iterator
import pandas as pd
from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar

class MultiTimeframeDataProvider:
    """
    Provides aligned multi-timeframe data during backtest.
    
    Critical Feature: LOOKAHEAD BIAS PREVENTION
    - Higher timeframe bars only update when they close
    - Lower timeframe gets all bars
    - Uses "forward-fill" for higher TF until new bar closes
    """
    
    def __init__(self, data_dict: Dict[Timeframe, pd.DataFrame]):
        """
        Args:
            data_dict: {Timeframe: DataFrame} all pre-loaded
        """
        self.data = data_dict
        self.timeframes = sorted(data_dict.keys(), key=lambda tf: tf.seconds)
        self.base_timeframe = self.timeframes[0]  # Lowest TF
        
        # Align all dataframes to base timeframe index
        self._align_timeframes()
    
    def _align_timeframes(self):
        """
        Align higher timeframes to base timeframe using forward-fill.
        This prevents lookahead bias.
        """
        base_index = self.data[self.base_timeframe].index
        
        for tf in self.timeframes[1:]:  # Skip base TF
            # Reindex to base TF, forward-fill
            self.data[tf] = self.data[tf].reindex(base_index, method='ffill')
    
    def iterate_bars(self) -> Iterator[Dict[Timeframe, OHLCVBar]]:
        """
        Iterate through time, yielding dict of bars for each timestamp.
        
        Yields:
            {Timeframe: OHLCVBar} at each base timeframe timestamp
        """
        base_df = self.data[self.base_timeframe]
        
        for timestamp, row in base_df.iterrows():
            bars = {}
            
            for tf in self.timeframes:
                tf_row = self.data[tf].loc[timestamp]
                
                bars[tf] = OHLCVBar(
                    timestamp=timestamp,
                    timeframe=tf,
                    instrument=self.instrument,  # From config
                    open=Decimal(str(tf_row['open'])),
                    high=Decimal(str(tf_row['high'])),
                    low=Decimal(str(tf_row['low'])),
                    close=Decimal(str(tf_row['close'])),
                    volume=int(tf_row['volume'])
                )
            
            yield bars