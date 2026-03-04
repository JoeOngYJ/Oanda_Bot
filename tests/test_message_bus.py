"""
Unit tests for Redis message bus.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from oanda_bot.utils.message_bus import MessageBus


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.xgroup_create = AsyncMock()
    mock.xadd = AsyncMock(return_value="1234567890-0")
    mock.xreadgroup = AsyncMock(return_value=[])
    mock.xack = AsyncMock()
    mock.xinfo_stream = AsyncMock(return_value={'length': 0})
    mock.xinfo_groups = AsyncMock(return_value=[])
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = MagicMock()
    config.redis.host = "localhost"
    config.redis.port = 6379
    config.redis.db = 1
    config.redis.streams = {
        "market_data": "test:market_data",
        "signals": "test:signals",
        "risk_checks": "test:risk_checks",
        "orders": "test:orders",
        "executions": "test:executions",
        "alerts": "test:alerts",
        "execution_control": "test:execution_control",
    }
    return config


class TestMessageBus:
    """Tests for MessageBus class"""

    @pytest.mark.asyncio
    async def test_connect(self, mock_config, mock_redis):
        """Test connecting to Redis"""
        with patch('oanda_bot.utils.message_bus.redis.Redis', return_value=mock_redis):
            bus = MessageBus(mock_config)
            await bus.connect()

            assert bus.redis_client is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_config, mock_redis):
        """Test disconnecting from Redis"""
        with patch('oanda_bot.utils.message_bus.redis.Redis', return_value=mock_redis):
            bus = MessageBus(mock_config)
            await bus.connect()
            await bus.disconnect()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_message(self, mock_config, mock_redis):
        """Test publishing a message"""
        with patch('oanda_bot.utils.message_bus.redis.Redis', return_value=mock_redis):
            bus = MessageBus(mock_config)
            await bus.connect()

            message = {"test": "data", "value": 123}
            message_id = await bus.publish("market_data", message)

            assert message_id == "1234567890-0"
            mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_adds_timestamp(self, mock_config, mock_redis):
        """Test that publish adds timestamp if not present"""
        with patch('oanda_bot.utils.message_bus.redis.Redis', return_value=mock_redis):
            bus = MessageBus(mock_config)
            await bus.connect()

            message = {"test": "data"}
            await bus.publish("market_data", message)

            # Check that xadd was called with timestamp
            call_args = mock_redis.xadd.call_args
            fields = call_args.kwargs['fields']
            assert 'timestamp' in fields

    @pytest.mark.asyncio
    async def test_serialize_message(self, mock_config):
        """Test message serialization"""
        bus = MessageBus(mock_config)

        message = {
            "string": "test",
            "number": 123,
            "dict": {"nested": "value"},
            "list": [1, 2, 3],
            "datetime": datetime.utcnow()
        }

        serialized = bus._serialize_message(message)

        assert isinstance(serialized["string"], str)
        assert isinstance(serialized["number"], str)
        assert isinstance(serialized["dict"], str)
        assert isinstance(serialized["list"], str)
        assert isinstance(serialized["datetime"], str)

    @pytest.mark.asyncio
    async def test_deserialize_message(self, mock_config):
        """Test message deserialization"""
        bus = MessageBus(mock_config)

        fields = {
            "string": "test",
            "number": "123",
            "dict": '{"nested": "value"}',
            "list": "[1, 2, 3]"
        }

        deserialized = bus._deserialize_message(fields)

        assert deserialized["string"] == "test"
        assert deserialized["dict"] == {"nested": "value"}
        assert deserialized["list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_stream_stats(self, mock_config, mock_redis):
        """Test getting stream statistics"""
        mock_redis.xinfo_stream.return_value = {
            'length': 10,
            'first-entry': None,
            'last-entry': None
        }
        mock_redis.xinfo_groups.return_value = [{'lag': 5}]

        with patch('oanda_bot.utils.message_bus.redis.Redis', return_value=mock_redis):
            bus = MessageBus(mock_config)
            await bus.connect()

            stats = await bus.get_stream_stats()

            assert 'market_data' in stats
            assert stats['market_data']['length'] == 10
            assert stats['market_data']['lag'] == 5

    @pytest.mark.asyncio
    async def test_publish_without_connection_raises_error(self, mock_config):
        """Test that publishing without connection raises error"""
        bus = MessageBus(mock_config)

        with pytest.raises(RuntimeError, match="Message bus not connected"):
            await bus.publish("market_data", {"test": "data"})

    @pytest.mark.asyncio
    async def test_subscribe_without_connection_raises_error(self, mock_config):
        """Test that subscribing without connection raises error"""
        bus = MessageBus(mock_config)

        with pytest.raises(RuntimeError, match="Message bus not connected"):
            async for _ in bus.subscribe("market_data"):
                pass
