"""
Unit tests for alert manager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock
from agents.monitoring.alerting import AlertManager


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = MagicMock()
    return config


@pytest.fixture
def alert_manager(mock_config):
    """Create alert manager instance"""
    return AlertManager(mock_config)


class TestAlertManager:
    """Tests for AlertManager class"""

    @pytest.mark.asyncio
    async def test_send_alert(self, alert_manager):
        """Test sending an alert"""
        await alert_manager.send_alert(
            severity="warning",
            component="market_data",
            message="Test alert",
            value=42.0
        )

        # Alert should be recorded
        alert_key = "market_data:Test alert"
        assert alert_key in alert_manager.last_alert_time

    @pytest.mark.asyncio
    async def test_alert_deduplication(self, alert_manager):
        """Test that duplicate alerts are deduplicated"""
        # Send first alert
        await alert_manager.send_alert(
            severity="warning",
            component="market_data",
            message="Duplicate test",
            value=1.0
        )

        # Send duplicate alert immediately
        await alert_manager.send_alert(
            severity="warning",
            component="market_data",
            message="Duplicate test",
            value=2.0
        )

        # Should only have one entry
        alert_key = "market_data:Duplicate test"
        assert alert_key in alert_manager.last_alert_time

    @pytest.mark.asyncio
    async def test_alert_not_deduplicated_after_interval(self, alert_manager):
        """Test that alerts are not deduplicated after interval"""
        alert_key = "market_data:Old alert"

        # Set last alert time to 10 minutes ago
        alert_manager.last_alert_time[alert_key] = datetime.utcnow() - timedelta(minutes=10)

        # Send alert - should not be deduplicated
        await alert_manager.send_alert(
            severity="warning",
            component="market_data",
            message="Old alert",
            value=1.0
        )

        # Time should be updated
        time_diff = (datetime.utcnow() - alert_manager.last_alert_time[alert_key]).total_seconds()
        assert time_diff < 5  # Should be recent

    @pytest.mark.asyncio
    async def test_process_alert_from_message(self, alert_manager):
        """Test processing alert from message"""
        message = {
            "severity": "critical",
            "component": "execution",
            "message": "Order failed",
            "value": 100.0,
            "metadata": {"order_id": "123"}
        }

        await alert_manager.process_alert(message)

        alert_key = "execution:Order failed"
        assert alert_key in alert_manager.last_alert_time
