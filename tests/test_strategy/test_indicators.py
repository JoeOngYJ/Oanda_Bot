"""
Unit tests for technical indicators.
"""

import pytest
from agents.strategy.indicators import sma, ema, rsi


class TestSMA:
    """Tests for Simple Moving Average"""

    def test_sma_calculation(self):
        """Test basic SMA calculation"""
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sma(prices, 3) == 4.0  # (3+4+5)/3

    def test_sma_insufficient_data(self):
        """Test SMA returns None with insufficient data"""
        prices = [1.0, 2.0]
        assert sma(prices, 5) is None

    def test_sma_exact_period(self):
        """Test SMA with exact period length"""
        prices = [10.0, 20.0, 30.0]
        assert sma(prices, 3) == 20.0


class TestRSI:
    """Tests for Relative Strength Index"""

    def test_rsi_calculation(self):
        """Test basic RSI calculation"""
        # Prices with clear uptrend
        prices = [44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0,
                  48.5, 49.0, 49.5, 50.0, 50.5, 51.0]
        result = rsi(prices, 14)
        assert result is not None
        assert 50 < result <= 100  # Should be high RSI for uptrend (100 is valid)

    def test_rsi_insufficient_data(self):
        """Test RSI returns None with insufficient data"""
        prices = [44.0, 44.5, 45.0]
        assert rsi(prices, 14) is None

    def test_rsi_no_losses(self):
        """Test RSI with all gains (no losses)"""
        prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                  10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        result = rsi(prices, 14)
        assert result == 100.0  # All gains = RSI 100


class TestEMA:
    """Tests for Exponential Moving Average"""

    def test_ema_calculation(self):
        """Test basic EMA calculation"""
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ema(prices, 3)
        assert result is not None
        assert 3.0 < result < 5.0  # Should be between SMA and last price

    def test_ema_insufficient_data(self):
        """Test EMA returns None with insufficient data"""
        prices = [1.0, 2.0]
        assert ema(prices, 5) is None
