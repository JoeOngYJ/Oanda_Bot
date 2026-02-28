"""
Unit tests for data normalizer.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from agents.market_data.data_normalizer import DataNormalizer
from shared.models import Instrument


class TestDataNormalizer:
    """Tests for DataNormalizer class"""

    def test_normalize_valid_tick(self):
        """Test normalizing a valid Oanda tick"""
        normalizer = DataNormalizer()

        raw_tick = {
            "instrument": "EUR_USD",
            "time": "1234567890.123",
            "bids": [{"price": "1.08500", "liquidity": 10000000}],
            "asks": [{"price": "1.08505", "liquidity": 10000000}],
            "closeoutBid": "1.08500",
            "closeoutAsk": "1.08505"
        }

        tick = normalizer.normalize(raw_tick, Instrument.EUR_USD)

        assert tick.instrument == Instrument.EUR_USD
        assert tick.bid == Decimal("1.08500")
        assert tick.ask == Decimal("1.08505")
        assert tick.spread == Decimal("0.00005")
        assert tick.bid_volume == 10000000
        assert tick.ask_volume == 10000000
        assert tick.source == "oanda"
        assert tick.data_version == "1.0.0"

    def test_normalize_without_liquidity(self):
        """Test normalizing tick without liquidity data"""
        normalizer = DataNormalizer()

        raw_tick = {
            "instrument": "GBP_USD",
            "time": "1234567890.456",
            "bids": [{"price": "1.25000"}],
            "asks": [{"price": "1.25010"}],
            "closeoutBid": "1.25000",
            "closeoutAsk": "1.25010"
        }

        tick = normalizer.normalize(raw_tick, Instrument.GBP_USD)

        assert tick.instrument == Instrument.GBP_USD
        assert tick.bid == Decimal("1.25000")
        assert tick.ask == Decimal("1.25010")
        assert tick.spread == Decimal("0.00010")
        assert tick.bid_volume is None
        assert tick.ask_volume is None

    def test_timestamp_parsing(self):
        """Test Unix timestamp parsing"""
        normalizer = DataNormalizer()

        raw_tick = {
            "instrument": "USD_JPY",
            "time": "1609459200.0",  # 2021-01-01 00:00:00 UTC
            "bids": [{"price": "103.500"}],
            "asks": [{"price": "103.505"}]
        }

        tick = normalizer.normalize(raw_tick, Instrument.USD_JPY)

        expected_time = datetime(2021, 1, 1, 0, 0, 0)
        assert tick.timestamp == expected_time

    def test_spread_calculation(self):
        """Test spread is calculated correctly"""
        normalizer = DataNormalizer()

        raw_tick = {
            "instrument": "EUR_USD",
            "time": "1234567890.0",
            "bids": [{"price": "1.10000"}],
            "asks": [{"price": "1.10020"}]
        }

        tick = normalizer.normalize(raw_tick, Instrument.EUR_USD)

        assert tick.spread == Decimal("0.00020")
        assert tick.spread == tick.ask - tick.bid
