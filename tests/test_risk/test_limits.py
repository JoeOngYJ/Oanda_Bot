"""
Unit tests for risk limits.
"""

import pytest
from decimal import Decimal
from shared.config import RiskLimitsConfig, PerInstrumentLimits, CircuitBreakerConfig
from shared.models import Instrument
from agents.risk.limits import RiskLimits


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


class TestRiskLimits:
    """Tests for RiskLimits class"""

    def test_initialization(self, risk_limits):
        """Test risk limits initialization"""
        assert risk_limits.daily_pnl == Decimal("0")
        assert risk_limits.daily_trades == 0
        assert risk_limits.consecutive_losses == 0
        assert risk_limits.circuit_breaker_active is False

    def test_get_max_position_size(self, risk_limits):
        """Test getting max position size"""
        max_size = risk_limits.get_max_position_size(Instrument.EUR_USD)
        assert max_size == 10000

    def test_get_max_order_size(self, risk_limits):
        """Test getting max order size"""
        max_size = risk_limits.get_max_order_size(Instrument.EUR_USD)
        assert max_size == 5000

    def test_get_max_daily_loss(self, risk_limits):
        """Test getting max daily loss"""
        assert risk_limits.get_max_daily_loss() == 1000.0

    def test_get_max_drawdown(self, risk_limits):
        """Test getting max drawdown"""
        assert risk_limits.get_max_drawdown() == 0.15

    def test_requires_stop_loss(self, risk_limits):
        """Test stop loss requirement"""
        assert risk_limits.requires_stop_loss() is True
