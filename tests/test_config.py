"""
Unit tests for configuration loading system.
"""

import pytest
import os
from pathlib import Path
from pydantic import ValidationError

from shared.config import Config


class TestConfig:
    """Tests for Config class"""

    def test_load_valid_config(self, temp_config_dir):
        """Test loading valid configuration"""
        config = Config.load(str(temp_config_dir))

        assert config.system.environment == "test"
        assert config.redis.host == "localhost"
        assert config.redis.port == 6379
        assert config.oanda.account_id == "test-account"
        assert len(config.strategies) == 1

    def test_redis_streams_loaded(self, temp_config_dir):
        """Test that Redis streams are loaded correctly"""
        config = Config.load(str(temp_config_dir))

        assert "market_data" in config.redis.streams
        assert "signals" in config.redis.streams
        assert config.redis.streams["market_data"] == "test:market_data"

    def test_oanda_config_loaded(self, temp_config_dir):
        """Test that Oanda configuration is loaded"""
        config = Config.load(str(temp_config_dir))

        assert config.oanda.environment == "practice"
        assert "practice" in config.oanda.endpoints
        assert "live" in config.oanda.endpoints
        assert len(config.oanda.instruments) == 2

    def test_risk_limits_loaded(self, temp_config_dir):
        """Test that risk limits are loaded"""
        config = Config.load(str(temp_config_dir))

        assert config.risk.max_daily_loss == 1000.0
        assert config.risk.max_drawdown == 2000.0
        assert config.risk.require_stop_loss is True
        assert config.risk.per_instrument.max_order_size == 5000

    def test_strategies_loaded(self, temp_config_dir):
        """Test that strategies are loaded"""
        config = Config.load(str(temp_config_dir))

        assert len(config.strategies) == 1
        strategy = config.strategies[0]
        assert strategy.name == "TestStrategy"
        assert strategy.enabled is True
        assert "test_param" in strategy.parameters

    def test_monitoring_config_loaded(self, temp_config_dir):
        """Test that monitoring configuration is loaded"""
        config = Config.load(str(temp_config_dir))

        assert config.monitoring.health_check_interval == 30
        assert config.monitoring.alert_thresholds.cpu_usage_warning == 70.0

    def test_missing_config_dir_raises_error(self):
        """Test that missing config directory raises error"""
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/path")

    def test_env_var_substitution(self, temp_config_dir):
        """Test environment variable substitution"""
        # Set environment variable
        os.environ["TEST_VAR"] = "test_value"

        # Create config with env var
        config_content = "test_key: ${TEST_VAR}"
        result = Config._substitute_env_vars(config_content)

        assert result == "test_key: test_value"

        # Clean up
        del os.environ["TEST_VAR"]

    def test_missing_env_var_raises_error(self):
        """Test that missing environment variable raises error"""
        config_content = "test_key: ${MISSING_VAR}"

        with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not set"):
            Config._substitute_env_vars(config_content)
