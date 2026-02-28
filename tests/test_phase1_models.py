# tests/test_phase1_models.py

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol, OrderSide, OrderStatus
from backtesting.data.models import OHLCVBar


class TestTimeframe:
    """Test Timeframe enum functionality"""

    def test_timeframe_members(self):
        """Test that all expected timeframe members exist"""
        assert Timeframe.M15.name == "M15"
        assert Timeframe.M30.name == "M30"
        assert Timeframe.H1.name == "H1"
        assert Timeframe.H4.name == "H4"
        assert Timeframe.D1.name == "D1"

    def test_timeframe_seconds(self):
        """Test seconds property returns correct values"""
        assert Timeframe.M15.seconds == 900
        assert Timeframe.M30.seconds == 1800
        assert Timeframe.H1.seconds == 3600
        assert Timeframe.H4.seconds == 14400
        assert Timeframe.D1.seconds == 86400

    def test_to_pandas_freq(self):
        """Test conversion to pandas frequency strings"""
        assert Timeframe.M15.to_pandas_freq() == "15T"
        assert Timeframe.M30.to_pandas_freq() == "30T"
        assert Timeframe.H1.to_pandas_freq() == "1H"
        assert Timeframe.H4.to_pandas_freq() == "4H"
        assert Timeframe.D1.to_pandas_freq() == "1D"

    def test_to_oanda_granularity(self):
        """Test conversion to OANDA granularity strings"""
        assert Timeframe.M15.to_oanda_granularity() == "M15"
        assert Timeframe.M30.to_oanda_granularity() == "M30"
        assert Timeframe.H1.to_oanda_granularity() == "H1"
        assert Timeframe.H4.to_oanda_granularity() == "H4"
        assert Timeframe.D1.to_oanda_granularity() == "D1"

    def test_from_pandas_freq_exact(self):
        """Test creating Timeframe from exact pandas frequency strings"""
        assert Timeframe.from_pandas_freq("15T") == Timeframe.M15
        assert Timeframe.from_pandas_freq("30T") == Timeframe.M30
        assert Timeframe.from_pandas_freq("1H") == Timeframe.H1
        assert Timeframe.from_pandas_freq("4H") == Timeframe.H4
        assert Timeframe.from_pandas_freq("1D") == Timeframe.D1

    def test_from_pandas_freq_variants(self):
        """Test creating Timeframe from variant pandas frequency strings"""
        # 60 minutes = 1 hour
        assert Timeframe.from_pandas_freq("60T") == Timeframe.H1
        assert Timeframe.from_pandas_freq("60min") == Timeframe.H1

        # Case insensitive
        assert Timeframe.from_pandas_freq("1h") == Timeframe.H1
        assert Timeframe.from_pandas_freq("1d") == Timeframe.D1

    def test_from_pandas_freq_invalid(self):
        """Test that invalid pandas frequency strings raise ValueError"""
        with pytest.raises(ValueError, match="Unknown pandas frequency"):
            Timeframe.from_pandas_freq("5T")

        with pytest.raises(ValueError, match="Unknown pandas frequency"):
            Timeframe.from_pandas_freq("2H")

    def test_from_oanda_granularity_exact(self):
        """Test creating Timeframe from OANDA granularity strings"""
        assert Timeframe.from_oanda_granularity("M15") == Timeframe.M15
        assert Timeframe.from_oanda_granularity("M30") == Timeframe.M30
        assert Timeframe.from_oanda_granularity("H1") == Timeframe.H1
        assert Timeframe.from_oanda_granularity("H4") == Timeframe.H4
        assert Timeframe.from_oanda_granularity("D1") == Timeframe.D1

    def test_from_oanda_granularity_case_insensitive(self):
        """Test that OANDA granularity parsing is case-insensitive"""
        assert Timeframe.from_oanda_granularity("h1") == Timeframe.H1
        assert Timeframe.from_oanda_granularity("m15") == Timeframe.M15
        assert Timeframe.from_oanda_granularity("d1") == Timeframe.D1

    def test_from_oanda_granularity_invalid(self):
        """Test that invalid OANDA granularity strings raise ValueError"""
        with pytest.raises(ValueError, match="Unknown OANDA granularity"):
            Timeframe.from_oanda_granularity("M5")

        with pytest.raises(ValueError, match="Unknown OANDA granularity"):
            Timeframe.from_oanda_granularity("H2")

    def test_get_hierarchy(self):
        """Test that hierarchy returns timeframes ordered by duration"""
        hierarchy = Timeframe.get_hierarchy()

        # Should be a list
        assert isinstance(hierarchy, list)

        # Should contain all timeframes
        assert len(hierarchy) == 5

        # Should be ordered by seconds (lowest first)
        assert hierarchy[0] == Timeframe.M15
        assert hierarchy[1] == Timeframe.M30
        assert hierarchy[2] == Timeframe.H1
        assert hierarchy[3] == Timeframe.H4
        assert hierarchy[4] == Timeframe.D1

        # Verify ordering by seconds
        for i in range(len(hierarchy) - 1):
            assert hierarchy[i].seconds < hierarchy[i + 1].seconds

    def test_pandas_roundtrip(self):
        """Test that pandas frequency round-trips correctly"""
        for tf in Timeframe:
            pandas_freq = tf.to_pandas_freq()
            reconstructed = Timeframe.from_pandas_freq(pandas_freq)
            assert reconstructed == tf

    def test_oanda_roundtrip(self):
        """Test that OANDA granularity round-trips correctly"""
        for tf in Timeframe:
            oanda_gran = tf.to_oanda_granularity()
            reconstructed = Timeframe.from_oanda_granularity(oanda_gran)
            assert reconstructed == tf


class TestOHLCVBar:
    """Test OHLCVBar dataclass functionality"""

    def test_create_bar_from_dict_with_strings(self):
        """Test creating OHLCVBar from dict with string numbers and ISO timestamp"""
        data = {
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0850',
            'high': '1.0875',
            'low': '1.0840',
            'close': '1.0860',
            'volume': '1000'
        }

        bar = OHLCVBar.from_dict(data)

        # Check types
        assert isinstance(bar.timestamp, datetime)
        assert isinstance(bar.timeframe, Timeframe)
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)
        assert isinstance(bar.volume, int)

        # Check values
        assert bar.timeframe == Timeframe.H1
        assert bar.instrument == InstrumentSymbol('EUR_USD')
        assert bar.open == Decimal('1.0850')
        assert bar.high == Decimal('1.0875')
        assert bar.low == Decimal('1.0840')
        assert bar.close == Decimal('1.0860')
        assert bar.volume == 1000

        # Check timestamp is timezone-aware UTC
        assert bar.timestamp.tzinfo is not None
        assert bar.timestamp.year == 2024
        assert bar.timestamp.month == 1
        assert bar.timestamp.day == 15

    def test_computed_properties(self):
        """Test computed properties of OHLCVBar"""
        bar = OHLCVBar.from_dict({
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0850',
            'high': '1.0900',
            'low': '1.0800',
            'close': '1.0880',
            'volume': '1000'
        })

        # Test typical_price: (H + L + C) / 3
        expected_typical = (Decimal('1.0900') + Decimal('1.0800') + Decimal('1.0880')) / Decimal('3')
        assert bar.typical_price == expected_typical

        # Test body_size: |C - O|
        assert bar.body_size == Decimal('0.0030')

        # Test upper_wick: H - max(O, C)
        assert bar.upper_wick == Decimal('0.0020')

        # Test lower_wick: min(O, C) - L
        assert bar.lower_wick == Decimal('0.0050')

        # Test is_bullish: C > O
        assert bar.is_bullish is True

        # Test range_size: H - L
        assert bar.range_size == Decimal('0.0100')

    def test_bearish_candle(self):
        """Test is_bullish property for bearish candle"""
        bar = OHLCVBar.from_dict({
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0880',
            'high': '1.0900',
            'low': '1.0800',
            'close': '1.0850',
            'volume': '1000'
        })

        assert bar.is_bullish is False
        assert bar.body_size == Decimal('0.0030')

    def test_validation_high_less_than_low(self):
        """Test that high < low raises ValueError"""
        with pytest.raises(ValueError, match="High .* must be >= Low"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0850',
                'high': '1.0800',  # High < Low
                'low': '1.0900',
                'close': '1.0860',
                'volume': '1000'
            })

    def test_validation_high_less_than_open(self):
        """Test that high < open raises ValueError"""
        with pytest.raises(ValueError, match="High .* must be >= Open"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0900',
                'high': '1.0850',  # High < Open
                'low': '1.0800',
                'close': '1.0860',
                'volume': '1000'
            })

    def test_validation_high_less_than_close(self):
        """Test that high < close raises ValueError"""
        with pytest.raises(ValueError, match="High .* must be >= Close"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0850',
                'high': '1.0860',
                'low': '1.0800',
                'close': '1.0900',  # Close > High
                'volume': '1000'
            })

    def test_validation_low_greater_than_open(self):
        """Test that low > open raises ValueError"""
        with pytest.raises(ValueError, match="Low .* must be <= Open"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0800',
                'high': '1.0900',
                'low': '1.0850',  # Low > Open
                'close': '1.0860',
                'volume': '1000'
            })

    def test_validation_low_greater_than_close(self):
        """Test that low > close raises ValueError"""
        with pytest.raises(ValueError, match="Low .* must be <= Close"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0900',
                'high': '1.0900',
                'low': '1.0870',  # Low > Close but Low < Open
                'close': '1.0860',
                'volume': '1000'
            })

    def test_validation_negative_volume(self):
        """Test that negative volume raises ValueError"""
        with pytest.raises(ValueError, match="Volume .* cannot be negative"):
            OHLCVBar.from_dict({
                'timestamp': '2024-01-15T10:30:00Z',
                'timeframe': 'H1',
                'instrument': 'EUR_USD',
                'open': '1.0850',
                'high': '1.0900',
                'low': '1.0800',
                'close': '1.0860',
                'volume': '-100'
            })

    def test_flexible_dict_keys(self):
        """Test that from_dict accepts flexible key names"""
        # Test with alternative key names
        data = {
            'time': '2024-01-15T10:30:00Z',  # 'time' instead of 'timestamp'
            'tf': 'H1',  # 'tf' instead of 'timeframe'
            'symbol': 'EUR_USD',  # 'symbol' instead of 'instrument'
            'open': '1.0850',
            'high': '1.0875',
            'low': '1.0840',
            'close': '1.0860',
            'vol': '1000'  # 'vol' instead of 'volume'
        }

        bar = OHLCVBar.from_dict(data)
        assert bar.instrument == InstrumentSymbol('EUR_USD')
        assert bar.volume == 1000

    def test_to_dict_serialization(self):
        """Test that to_dict serializes correctly"""
        bar = OHLCVBar.from_dict({
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0850',
            'high': '1.0875',
            'low': '1.0840',
            'close': '1.0860',
            'volume': '1000'
        })

        data = bar.to_dict()

        # Check that all fields are present
        assert 'timestamp' in data
        assert 'timeframe' in data
        assert 'instrument' in data
        assert 'open' in data
        assert 'high' in data
        assert 'low' in data
        assert 'close' in data
        assert 'volume' in data

        # Check that timestamp is ISO string
        assert isinstance(data['timestamp'], str)
        assert 'T' in data['timestamp']

        # Check that prices are strings
        assert isinstance(data['open'], str)
        assert isinstance(data['high'], str)
        assert isinstance(data['low'], str)
        assert isinstance(data['close'], str)

        # Check that volume is int
        assert isinstance(data['volume'], int)

    def test_repr(self):
        """Test that __repr__ returns a friendly string"""
        bar = OHLCVBar.from_dict({
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0850',
            'high': '1.0875',
            'low': '1.0840',
            'close': '1.0860',
            'volume': '1000'
        })

        repr_str = repr(bar)

        # Check that repr contains key information
        assert 'EUR_USD' in repr_str
        assert 'H1' in repr_str
        assert '2024-01-15' in repr_str
        assert '1.0850' in repr_str
        assert '1000' in repr_str

    def test_immutability(self):
        """Test that OHLCVBar is immutable (frozen dataclass)"""
        bar = OHLCVBar.from_dict({
            'timestamp': '2024-01-15T10:30:00Z',
            'timeframe': 'H1',
            'instrument': 'EUR_USD',
            'open': '1.0850',
            'high': '1.0875',
            'low': '1.0840',
            'close': '1.0860',
            'volume': '1000'
        })

        # Attempt to modify should raise an error
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            bar.open = Decimal('1.0900')


class TestTypes:
    """Test type aliases and enums"""

    def test_order_side_enum(self):
        """Test OrderSide enum members"""
        assert OrderSide.LONG.value == "LONG"
        assert OrderSide.SHORT.value == "SHORT"

    def test_order_status_enum(self):
        """Test OrderStatus enum members"""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"

    def test_instrument_symbol_type(self):
        """Test InstrumentSymbol type alias"""
        symbol = InstrumentSymbol("EUR_USD")
        assert isinstance(symbol, str)
        assert symbol == "EUR_USD"
