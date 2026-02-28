"""
Unit tests for position monitor.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from shared.config import RiskLimitsConfig, PerInstrumentLimits, CircuitBreakerConfig
from shared.models import Position, MarketTick, Instrument, Side
from agents.risk.limits import RiskLimits
from agents.risk.position_monitor import PositionMonitor


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
def position_monitor(risk_limits):
    """Create PositionMonitor instance"""
    return PositionMonitor(risk_limits)


@pytest.fixture
def long_position():
    """Create a long position"""
    return Position(
        position_id="pos-1",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        entry_price=Decimal("1.1000"),
        current_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0980"),
        take_profit=Decimal("1.1050"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        opened_at=datetime.utcnow()
    )


class TestPositionMonitor:
    """Tests for PositionMonitor class"""

    def test_add_position(self, position_monitor, long_position):
        """Test adding a position"""
        position_monitor.add_position(long_position)
        assert "pos-1" in position_monitor.positions
        assert position_monitor.risk_limits.open_positions[Instrument.EUR_USD] == 1000
