"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import tempfile
import yaml


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_market_tick():
    """Sample market tick for testing"""
    from shared.models import MarketTick, Instrument

    return MarketTick(
        instrument=Instrument.EUR_USD,
        timestamp=datetime.utcnow(),
        bid=Decimal("1.1000"),
        ask=Decimal("1.1002"),
        spread=Decimal("0.0002"),
        source="oanda",
        data_version="1.0.0"
    )


@pytest.fixture
def sample_trade_signal():
    """Sample trade signal for testing"""
    from shared.models import TradeSignal, Instrument, Side

    return TradeSignal(
        signal_id="test-signal-123",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        confidence=0.85,
        rationale="Test signal for unit testing",
        strategy_name="TestStrategy",
        strategy_version="1.0.0",
        entry_price=Decimal("1.1002"),
        stop_loss=Decimal("1.0982"),
        take_profit=Decimal("1.1042"),
        timestamp=datetime.utcnow(),
        metadata={"test": True}
    )


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test configs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create test config files
        system_config = {
            "system": {"environment": "test", "timezone": "UTC"},
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 1,
                "streams": {
                    "market_data": "test:market_data",
                    "signals": "test:signals",
                    "risk_checks": "test:risk_checks",
                    "orders": "test:orders",
                    "executions": "test:executions",
                    "alerts": "test:alerts"
                }
            },
            "influxdb": {
                "url": "http://localhost:8086",
                "token": "test-token",
                "org": "test_org",
                "bucket": "test_bucket"
            },
            "prometheus": {
                "port": 8001,
                "push_gateway": "localhost:9091"
            },
            "logging": {
                "level": "DEBUG",
                "format": "json"
            }
        }

        oanda_config = {
            "oanda": {
                "account_id": "test-account",
                "api_token": "test-token",
                "environment": "practice",
                "endpoints": {
                    "practice": {
                        "api": "https://api-fxpractice.oanda.com",
                        "stream": "https://stream-fxpractice.oanda.com"
                    },
                    "live": {
                        "api": "https://api-fxtrade.oanda.com",
                        "stream": "https://stream-fxtrade.oanda.com"
                    }
                },
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 2.0,
                "instruments": ["EUR_USD", "GBP_USD"],
                "requests_per_second": 100
            }
        }

        risk_config = {
            "risk_limits": {
                "max_daily_loss": 1000.0,
                "max_drawdown": 2000.0,
                "max_total_exposure": 50000.0,
                "per_instrument": {
                    "max_position_size": 10000,
                    "max_order_size": 5000
                },
                "max_leverage": 10.0,
                "min_account_balance": 5000.0,
                "require_stop_loss": True,
                "max_stop_loss_distance": 0.005,
                "max_correlated_exposure": 0.3,
                "circuit_breaker": {
                    "consecutive_losses": 5,
                    "loss_velocity_1h": 500.0,
                    "volatility_spike_threshold": 3.0
                },
                "max_orders_per_minute": 10,
                "max_open_positions": 5
            }
        }

        strategies_config = {
            "strategies": [
                {
                    "name": "TestStrategy",
                    "version": "1.0.0",
                    "enabled": True,
                    "strategy_class": "TestStrategy",
                    "instruments": ["EUR_USD"],
                    "parameters": {"test_param": 42}
                }
            ]
        }

        monitoring_config = {
            "monitoring": {
                "health_check_interval": 30,
                "alert_thresholds": {
                    "market_data_latency_warning": 1.0,
                    "market_data_latency_critical": 5.0,
                    "market_data_stale_warning": 10.0,
                    "market_data_stale_critical": 30.0,
                    "order_fill_time_warning": 2.0,
                    "order_fill_time_critical": 10.0,
                    "order_rejection_rate_warning": 0.05,
                    "order_rejection_rate_critical": 0.15,
                    "cpu_usage_warning": 70.0,
                    "cpu_usage_critical": 90.0,
                    "memory_usage_warning": 75.0,
                    "memory_usage_critical": 90.0,
                    "strategy_error_rate_warning": 0.01,
                    "strategy_error_rate_critical": 0.05
                },
                "collection_interval": 10,
                "retention_days": 30
            }
        }

        # Write config files
        with open(config_dir / "system.yaml", "w") as f:
            yaml.dump(system_config, f)

        with open(config_dir / "oanda.yaml", "w") as f:
            yaml.dump(oanda_config, f)

        with open(config_dir / "risk_limits.yaml", "w") as f:
            yaml.dump(risk_config, f)

        with open(config_dir / "strategies.yaml", "w") as f:
            yaml.dump(strategies_config, f)

        with open(config_dir / "monitoring.yaml", "w") as f:
            yaml.dump(monitoring_config, f)

        yield config_dir
