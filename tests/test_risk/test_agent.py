"""
Unit tests for risk agent.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from oanda_bot.utils.config import Config, RiskLimitsConfig, PerInstrumentLimits, CircuitBreakerConfig
from oanda_bot.utils.models import TradeSignal, Instrument, Side
from oanda_bot.agents.risk.agent import RiskAgent


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = MagicMock(spec=Config)
    config.risk = RiskLimitsConfig(
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
    return config


@pytest.fixture
def mock_message_bus():
    """Create mock message bus"""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def risk_agent(mock_config, mock_message_bus):
    """Create RiskAgent instance"""
    return RiskAgent(mock_config, mock_message_bus)


class TestRiskAgent:
    """Tests for RiskAgent class"""

    def test_initialization(self, risk_agent):
        """Test risk agent initialization"""
        assert risk_agent.running is False
        assert risk_agent.account_balance == Decimal("10000.0")
        assert risk_agent.risk_limits is not None
        assert risk_agent.pre_trade_checker is not None
        assert risk_agent.position_monitor is not None
        assert risk_agent.circuit_breaker is not None
