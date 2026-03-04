"""
Unit tests for health checker.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from oanda_bot.agents.monitoring.health_checker import HealthChecker


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = MagicMock()
    config.monitoring.alert_thresholds.cpu_usage_warning = 70.0
    config.monitoring.alert_thresholds.cpu_usage_critical = 90.0
    config.monitoring.alert_thresholds.memory_usage_warning = 75.0
    config.monitoring.alert_thresholds.memory_usage_critical = 90.0
    config.monitoring.alert_thresholds.market_data_stale_warning = 10.0
    config.monitoring.alert_thresholds.market_data_stale_critical = 30.0
    return config


@pytest.fixture
def health_checker(mock_config):
    """Create health checker instance"""
    return HealthChecker(mock_config)


class TestHealthChecker:
    """Tests for HealthChecker class"""

    @pytest.mark.asyncio
    async def test_check_system_resources(self, health_checker):
        """Test system resource checks"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk:

            mock_mem.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            metrics = await health_checker._check_system_resources()

            assert len(metrics) == 3
            assert any(m.metric_name == "cpu_usage" for m in metrics)
            assert any(m.metric_name == "memory_usage" for m in metrics)
            assert any(m.metric_name == "disk_usage" for m in metrics)

    @pytest.mark.asyncio
    async def test_high_cpu_triggers_warning(self, health_checker):
        """Test that high CPU usage triggers warning status"""
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk:

            mock_mem.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            metrics = await health_checker._check_system_resources()

            cpu_metric = next(m for m in metrics if m.metric_name == "cpu_usage")
            assert cpu_metric.status == "warning"

    @pytest.mark.asyncio
    async def test_critical_cpu_triggers_critical(self, health_checker):
        """Test that critical CPU usage triggers critical status"""
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk:

            mock_mem.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0

            metrics = await health_checker._check_system_resources()

            cpu_metric = next(m for m in metrics if m.metric_name == "cpu_usage")
            assert cpu_metric.status == "critical"

    @pytest.mark.asyncio
    async def test_market_data_staleness_check(self, health_checker):
        """Test market data staleness detection"""
        # Set last tick time to 40 seconds ago
        health_checker.update_market_tick_time(
            "EUR_USD",
            datetime.utcnow() - timedelta(seconds=40)
        )

        metrics = await health_checker._check_market_data_feeds()

        assert len(metrics) == 1
        assert metrics[0].status == "critical"
        assert metrics[0].value > 30.0

    @pytest.mark.asyncio
    async def test_update_market_tick_time(self, health_checker):
        """Test updating market tick time"""
        now = datetime.utcnow()
        health_checker.update_market_tick_time("EUR_USD", now)

        assert "EUR_USD" in health_checker.last_market_tick
        assert health_checker.last_market_tick["EUR_USD"] == now
