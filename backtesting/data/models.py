# backtesting/data/models.py

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Dict, Any
from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol


@dataclass(frozen=True)
class OHLCVBar:
    """
    Immutable OHLCV bar representation with validation and computed properties.
    """
    timestamp: datetime
    timeframe: Timeframe
    instrument: InstrumentSymbol
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def __post_init__(self):
        """
        Normalize and validate bar data.
        - Ensures timestamp is timezone-aware UTC
        - Converts prices to Decimal
        - Converts volume to int
        - Validates OHLC relationships and volume
        """
        # Normalize timestamp to UTC
        if isinstance(self.timestamp, str):
            # Parse ISO format string
            try:
                ts = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid timestamp string: {self.timestamp}") from e
            object.__setattr__(self, 'timestamp', ts)

        # Ensure timezone-aware (treat naive as UTC)
        if self.timestamp.tzinfo is None:
            ts_utc = self.timestamp.replace(tzinfo=timezone.utc)
            object.__setattr__(self, 'timestamp', ts_utc)

        # Normalize prices to Decimal
        for field in ['open', 'high', 'low', 'close']:
            value = getattr(self, field)
            if not isinstance(value, Decimal):
                try:
                    decimal_value = Decimal(str(value))
                    object.__setattr__(self, field, decimal_value)
                except (InvalidOperation, ValueError) as e:
                    raise ValueError(f"Invalid {field} value: {value}") from e

        # Normalize volume to int
        if not isinstance(self.volume, int):
            try:
                int_volume = int(self.volume)
                object.__setattr__(self, 'volume', int_volume)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid volume value: {self.volume}") from e

        # Validate OHLC relationships
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) must be >= Low ({self.low})")
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Close ({self.close})")
        if self.volume < 0:
            raise ValueError(f"Volume ({self.volume}) cannot be negative")

    @property
    def typical_price(self) -> Decimal:
        """(High + Low + Close) / 3"""
        return (self.high + self.low + self.close) / Decimal('3')

    @property
    def body_size(self) -> Decimal:
        """Absolute candle body size"""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> Decimal:
        """Upper shadow length"""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        """Lower shadow length"""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """True if close > open"""
        return self.close > self.open

    @property
    def range_size(self) -> Decimal:
        """Total range from high to low"""
        return self.high - self.low

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCVBar":
        """
        Create OHLCVBar from dictionary with flexible key names.

        Accepts flexible keys:
        - timestamp/time
        - timeframe/tf/granularity
        - instrument/symbol
        - open/high/low/close
        - volume/vol

        Timeframe can be a string ("1H", "H1") or Timeframe enum.

        Args:
            data: Dictionary with bar data

        Returns:
            OHLCVBar instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Extract timestamp
        timestamp = data.get('timestamp') or data.get('time')
        if timestamp is None:
            raise ValueError("Missing timestamp field (timestamp or time)")

        # Extract timeframe
        tf_value = data.get('timeframe') or data.get('tf') or data.get('granularity')
        if tf_value is None:
            raise ValueError("Missing timeframe field (timeframe, tf, or granularity)")

        # Convert timeframe string to Timeframe enum if needed
        if isinstance(tf_value, str):
            # Try OANDA granularity first, then pandas freq
            try:
                timeframe = Timeframe.from_oanda_granularity(tf_value)
            except ValueError:
                try:
                    timeframe = Timeframe.from_pandas_freq(tf_value)
                except ValueError:
                    raise ValueError(f"Invalid timeframe string: {tf_value}")
        elif isinstance(tf_value, Timeframe):
            timeframe = tf_value
        else:
            raise ValueError(f"Invalid timeframe type: {type(tf_value)}")

        # Extract instrument
        instrument = data.get('instrument') or data.get('symbol')
        if instrument is None:
            raise ValueError("Missing instrument field (instrument or symbol)")
        instrument = InstrumentSymbol(str(instrument))

        # Extract OHLC
        open_price = data.get('open')
        high_price = data.get('high')
        low_price = data.get('low')
        close_price = data.get('close')

        if any(x is None for x in [open_price, high_price, low_price, close_price]):
            raise ValueError("Missing OHLC fields (open, high, low, close)")

        # Extract volume
        volume = data.get('volume') or data.get('vol')
        if volume is None:
            raise ValueError("Missing volume field (volume or vol)")

        return cls(
            timestamp=timestamp,
            timeframe=timeframe,
            instrument=instrument,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize OHLCVBar to dictionary.

        Returns:
            Dictionary with serialized bar data
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe.name,
            'instrument': str(self.instrument),
            'open': str(self.open),
            'high': str(self.high),
            'low': str(self.low),
            'close': str(self.close),
            'volume': self.volume
        }

    def __repr__(self) -> str:
        """Friendly string representation"""
        return (
            f"OHLCVBar({self.instrument} {self.timeframe.name} "
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"O:{self.open} H:{self.high} L:{self.low} C:{self.close} V:{self.volume})"
        )
