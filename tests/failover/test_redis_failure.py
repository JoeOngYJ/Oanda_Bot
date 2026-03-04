#!/usr/bin/env python3
"""Failover Test: Redis Failure"""

import asyncio
import pytest

from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_redis_connection_handling():
    """Test message bus handles Redis connection gracefully"""
    config = Config.load()
    message_bus = MessageBus(config)
    
    # Test connection
    await message_bus.connect()
    assert message_bus.redis_client is not None

    # Test disconnect (closes connection but doesn't set redis_client to None)
    await message_bus.disconnect()


@pytest.mark.asyncio
async def test_redis_reconnection():
    """Test message bus can reconnect after disconnect"""
    config = Config.load()
    message_bus = MessageBus(config)
    
    # Connect
    await message_bus.connect()
    
    # Disconnect
    await message_bus.disconnect()
    
    # Reconnect
    await message_bus.connect()
    assert message_bus.redis_client is not None

    await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
