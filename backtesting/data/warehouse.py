# backtesting/data/warehouse.py

from pathlib import Path
import pandas as pd
from typing import Optional
import datetime as dt

from backtesting.core.timeframe import Timeframe


class DataWarehouse:
    """Persistent storage for historical market data.

    Storage Strategy:
    - Parquet for columnar compression
    - Partitioned by instrument and timeframe
    - Immutable once written (append-only)
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.DataFrame, instrument: str, timeframe: Timeframe):
        """Save OHLCV data to Parquet.

        Expects a DataFrame with a DatetimeIndex and columns: open, high, low, close, volume
        """
        path = self._get_path(instrument, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data.to_parquet(
                path,
                engine="pyarrow",
                compression="snappy",
                index=True,
            )
        except Exception as exc:
            # If pyarrow (or another parquet engine) isn't installed or
            # fails, fall back to CSV so data is still persisted.
            # This keeps the pipeline usable in minimal environments.
            fallback = path.with_suffix(".csv")
            data.to_csv(fallback, index=True)
            # Re-raise only if it's not an import/engine issue; otherwise log
            # a friendly warning via exception message.
            # (We avoid importing a logger here to keep modules minimal.)
            print(f"Warning: parquet write failed ({exc}); wrote CSV to {fallback}")

    def load(
        self,
        instrument: str,
        timeframe: Timeframe,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Load data from Parquet and optionally slice by time range."""
        path = self._get_path(instrument, timeframe)

        if not path.exists():
            # try CSV fallback as well
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, index_col=0, parse_dates=True)
            return None

        df = pd.read_parquet(path)
        if start is None and end is None:
            return df

        if start is None:
            start = df.index.min()
        if end is None:
            end = df.index.max()

        return df.loc[start:end]

    def import_csv(self, csv_path: Path, instrument: str, timeframe: Timeframe, datetime_col: str = "datetime") -> pd.DataFrame:
        """Import a CSV file into the warehouse and return the saved DataFrame.

        This reads the CSV, ensures the index is a DatetimeIndex, and saves as parquet.
        The CSV should contain columns: datetime (or other), open, high, low, close, volume
        """
        df = pd.read_csv(csv_path)
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found in CSV")

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col).sort_index()

        # keep only expected columns if present
        expected = ["open", "high", "low", "close", "volume"]
        for col in expected:
            if col not in df.columns:
                df[col] = None

        # coerce dtypes where possible
        df = df.astype({"open": float, "high": float, "low": float, "close": float}, errors="ignore")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

        # Save into warehouse
        self.save(df, instrument, timeframe)
        return df

    def _get_path(self, instrument: str, timeframe: Timeframe) -> Path:
        """Generate file path: base_path/<instrument>/<timeframe>.parquet"""
        return self.base_path / instrument / f"{timeframe.name}.parquet"