# backtesting/data/manager.py

from pathlib import Path
from typing import List, Dict, Optional
import datetime as dt
import pandas as pd
from oanda_bot.backtesting.core.timeframe import Timeframe
from .downloader import OandaDownloader
from .warehouse import DataWarehouse

class DataManager:
    """
    High-level interface for data acquisition and retrieval.
    
    Responsibilities:
    - Download data if missing
    - Resample to multiple timeframes
    - Cache aggressively
    - Validate data integrity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.data_dir = Path(self.config.get("data_dir", "data/backtesting"))
        self.downloader = OandaDownloader(self.config.get("oanda", {}))
        self.warehouse = DataWarehouse(self.data_dir)
    
    def ensure_data(
        self,
        instrument: str,
        base_timeframe: Timeframe,
        start_date: Optional[dt.datetime],
        end_date: Optional[dt.datetime],
        timeframes: List[Timeframe],
        force_download: bool = False,
        *,
        price: str = "M",
        count: Optional[int] = None,
        store_bid_ask: bool = False,
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Ensure data exists for all requested timeframes.
        Downloads base timeframe, resamples to higher TFs.
        
        Returns:
            {Timeframe: DataFrame} with OHLCV columns
        """
        # Check cache
        cached = None
        if not force_download:
            # load without slicing so we can inspect cached bounds
            cached = self.warehouse.load(instrument, base_timeframe, None, None)

            if cached is not None:
                # if the cached data fully covers the requested range, return it
                if start_date is None and end_date is None:
                    return self._resample_to_timeframes(cached, timeframes)

                cached_min = cached.index.min()
                cached_max = cached.index.max()

                needs_left = start_date is not None and start_date < cached_min
                needs_right = end_date is not None and end_date > cached_max

                if not needs_left and not needs_right:
                    # cached covers the requested span (or request was within cache)
                    # slice to the requested window before resampling
                    slice_start = start_date or cached_min
                    slice_end = end_date or cached_max
                    sliced = cached.loc[slice_start:slice_end]
                    return self._resample_to_timeframes(sliced, timeframes)

                # We'll fill missing ranges non-destructively
                parts = [cached]
                gran = base_timeframe.to_oanda_granularity() if isinstance(base_timeframe, Timeframe) else str(base_timeframe)

                # download left side (older data)
                if needs_left:
                    left_end = cached_min - dt.timedelta(microseconds=1)
                    left_df = self.downloader.download(
                                    instrument=instrument, granularity=gran,
                                    start=start_date, end=left_end,
                                    price=price, count=count, store_bid_ask=store_bid_ask)
                    self._validate_data(left_df)
                    parts.insert(0, left_df)

                # download right side (newer data)
                if needs_right:
                    right_start = cached_max + dt.timedelta(microseconds=1)
                    right_df = self.downloader.download(
                        instrument=instrument, granularity=gran, start=right_start, end=end_date,
                        price=price, count=count, store_bid_ask=store_bid_ask
                    )
                    self._validate_data(right_df)
                    parts.append(right_df)

                # concatenate and deduplicate
                merged = pd.concat(parts).sort_index()
                merged = merged[~merged.index.duplicated(keep="first")]

                # store merged dataset
                self.warehouse.save(merged, instrument, base_timeframe)

                # slice to requested window and resample
                slice_start = start_date or merged.index.min()
                slice_end = end_date or merged.index.max()
                sliced = merged.loc[slice_start:slice_end]
                return self._resample_to_timeframes(sliced, timeframes)

        # If we reach here we need to download fresh (no cache or force)
        gran = base_timeframe.to_oanda_granularity() if isinstance(base_timeframe, Timeframe) else str(base_timeframe)
        data = self.downloader.download(
            instrument=instrument, granularity=gran, start=start_date, end=end_date,
            price=price, count=count, store_bid_ask=store_bid_ask
        )

        # Validate
        self._validate_data(data)

        # Store
        self.warehouse.save(data, instrument, base_timeframe)

        # Resample
        return self._resample_to_timeframes(data, timeframes)
    
    def _resample_to_timeframes(self, base_data, timeframes):
        result = {}
        has_mid = all(c in base_data.columns for c in ("mid_o","mid_h","mid_l","mid_c"))

        for tf in timeframes:
            if has_mid:
                agg = {
                    "mid_o": "first",
                    "mid_h": "max",
                    "mid_l": "min",
                    "mid_c": "last",
                    "volume": "sum",
                }
                if "spread_c" in base_data.columns:
                    agg["spread_c"] = "mean"
                res = base_data.resample(tf.to_pandas_freq()).agg(agg).dropna()
                res = res.rename(columns={"mid_o":"open","mid_h":"high","mid_l":"low","mid_c":"close"})
            else:
                agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                res = base_data.resample(tf.to_pandas_freq()).agg(agg).dropna()

            result[tf] = res
        return result

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Basic validation on returned dataframe."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Downloader did not return a pandas DataFrame")
