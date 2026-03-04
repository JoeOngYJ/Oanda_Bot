"""
Unit tests for pre-trade checks.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from oanda_bot.utils.config import RiskLimitsConfig, PerInstrumentLimits, CircuitBreakerConfig
from oanda_bot.utils.models import TradeSignal, Instrument, Side
from oanda_bot.agents.risk.limits import RiskLimits
from oanda_bot.agents.risk.pre_trade_checks import PreTradeChecker


@pytest.fixture
def risk_config():
    """Create test risk configuration"""
    return RiskLimitsConfig(
        max_daily_loss=1000.0,
        max_drawdown=0.15,
        max_total_exposure=50000.0,
        per_instrument=PerInstrumentLimits(
            max_position_size=10000,
            max_order_size=5000
        ),
        max_leverage=10.0,
        min_account_balance=5000.0,
        require_stop_loss=True,
        max_stop_loss_distance=0.02,
        max_correlated_exposure=30000.0,
        circuit_breaker=CircuitBreakerConfig(
            consecutive_losses=5,
            loss_velocity_1h=500.0,
            volatility_spike_threshold=3.0
        ),
        max_orders_per_minute=10,
        max_open_positions=5
    )


@pytest.fixture
def risk_limits(risk_config):
    """Create RiskLimits instance"""
    return RiskLimits(risk_config)


@pytest.fixture
def pre_trade_checker(risk_limits):
    """Create PreTradeChecker instance"""
    return PreTradeChecker(risk_limits, Decimal("10000.0"))


@pytest.fixture
def valid_signal():
    """Create a valid trade signal"""
    return TradeSignal(
        signal_id="test-signal-1",
        timestamp=datetime.utcnow(),
        strategy_name="TestStrategy",
        strategy_version="1.0.0",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0980"),
        take_profit=Decimal("1.1050"),
        confidence=0.85,
        rationale="Test signal"
    )


class TestPreTradeChecker:
    """Tests for PreTradeChecker class"""

    @pytest.mark.asyncio
    async def test_valid_signal_approved(self, pre_trade_checker, valid_signal):
        """Test that valid signal is approved"""
        result = await pre_trade_checker.check_signal(valid_signal)
        assert result.approved is True
        assert len(result.reasons) == 0

    @pytest.mark.asyncio
    async def test_order_size_exceeded(self, pre_trade_checker, valid_signal):
        """Test rejection when order size exceeds limit"""
        valid_signal.quantity = 6000  # Exceeds max_order_size of 5000
        result = await pre_trade_checker.check_signal(valid_signal)
        assert result.approved is False
        assert any("Order size" in reason for reason in result.reasons)
