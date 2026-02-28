"""
Unit tests for circuit breaker.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from shared.config import CircuitBreakerConfig
from agents.risk.limits import RiskLimits
from agents.risk.circuit_breaker import CircuitBreaker
from shared.config import RiskLimitsConfig, PerInstrumentLimits


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
def circuit_breaker(risk_config, risk_limits):
    """Create CircuitBreaker instance"""
    return CircuitBreaker(risk_config.circuit_breaker, risk_limits)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class"""

    def test_initialization(self, circuit_breaker):
        """Test circuit breaker initialization"""
        assert circuit_breaker.recent_trades == []
        assert circuit_breaker.peak_balance == Decimal("0")
        assert circuit_breaker.risk_limits.circuit_breaker_active is False
