"""
Unit tests for data validator.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from oanda_bot.agents.market_data.data_validator import DataValidator
from oanda_bot.utils.models import MarketTick, Instrument
from unittest.mock import MagicMock


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = MagicMock()
    return config


@pytest.fixture
def validator(mock_config):
    """Create validator instance"""
    return DataValidator(mock_config)


class TestDataValidator:
    """Tests for DataValidator class"""

    def test_valid_tick(self, validator):
        """Test validating a valid tick"""
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.08500"),
            ask=Decimal("1.08505"),
            spread=Decimal("0.00005")
        )

        is_valid, issues = validator.validate(tick)

        assert is_valid is True
        assert len(issues) == 0

    def test_invalid_bid_ask(self, validator):
        """Test that bid >= ask is rejected at model level"""
        from pydantic import ValidationError

        # MarketTick model should reject invalid bid/ask at creation
        with pytest.raises(ValidationError) as exc_info:
            tick = MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=datetime.utcnow(),
                bid=Decimal("1.08505"),
                ask=Decimal("1.08500"),
                spread=Decimal("0.00005")
            )

        assert "Ask price" in str(exc_info.value)

    def test_excessive_spread(self, validator):
        """Test that excessive spread is flagged"""
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.08000"),
            ask=Decimal("1.09000"),  # 1% spread
            spread=Decimal("0.01000")
        )

        is_valid, issues = validator.validate(tick)

        assert is_valid is False
        assert any("Excessive spread" in issue for issue in issues)

    def test_large_price_jump(self, validator):
        """Test that large price jumps are detected"""
        # First tick
        tick1 = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.08000"),
            ask=Decimal("1.08005"),
            spread=Decimal("0.00005")
        )
        validator.validate(tick1)

        # Second tick with large jump
        tick2 = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            bid=Decimal("1.12000"),  # 3.7% jump
            ask=Decimal("1.12005"),
            spread=Decimal("0.00005")
        )

        is_valid, issues = validator.validate(tick2)

        assert is_valid is False
        assert any("Large price jump" in issue for issue in issues)

    def test_out_of_order_tick(self, validator):
        """Test that out-of-order ticks are rejected"""
        # First tick
        tick1 = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.08000"),
            ask=Decimal("1.08005"),
            spread=Decimal("0.00005")
        )
        validator.validate(tick1)

        # Second tick with earlier timestamp
        tick2 = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow() - timedelta(seconds=10),
            bid=Decimal("1.08010"),
            ask=Decimal("1.08015"),
            spread=Decimal("0.00005")
        )

        is_valid, issues = validator.validate(tick2)

        assert is_valid is False
        assert any("Out-of-order" in issue for issue in issues)

    def test_stale_tick(self, validator):
        """Test that stale ticks are flagged"""
        # Tick from 20 seconds ago
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow() - timedelta(seconds=20),
            bid=Decimal("1.08000"),
            ask=Decimal("1.08005"),
            spread=Decimal("0.00005")
        )

        is_valid, issues = validator.validate(tick)

        assert is_valid is False
        assert any("Stale tick" in issue for issue in issues)

    def test_multiple_instruments_tracked_separately(self, validator):
        """Test that different instruments are tracked independently"""
        # EUR_USD tick
        tick1 = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.08000"),
            ask=Decimal("1.08005"),
            spread=Decimal("0.00005")
        )
        validator.validate(tick1)

        # GBP_USD tick (different instrument)
        tick2 = MarketTick(
            instrument=Instrument.GBP_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.25000"),
            ask=Decimal("1.25005"),
            spread=Decimal("0.00005")
        )

        is_valid, issues = validator.validate(tick2)

        # Should be valid (no previous tick for GBP_USD)
        assert is_valid is True
