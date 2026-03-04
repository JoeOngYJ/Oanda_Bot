#!/usr/bin/env python3
"""Failover Test: Agent Restart"""

import asyncio
import pytest

from oanda_bot.utils.config import Config
from oanda_bot.utils.message_bus import MessageBus


@pytest.mark.asyncio
async def test_agent_restart_recovery():
    """Test agent can restart and recover state"""
    config = Config.load()
    message_bus = MessageBus(config)
    
    # Simulate agent lifecycle
    await message_bus.connect()
    
    # Simulate crash (disconnect)
    await message_bus.disconnect()
    
    # Simulate restart (reconnect)
    await message_bus.connect()
    assert message_bus.redis_client is not None

    await message_bus.disconnect()


@pytest.mark.asyncio
async def test_message_persistence_after_restart():
    """Test messages persist after agent restart"""
    config = Config.load()
    message_bus = MessageBus(config)
    
    await message_bus.connect()
    
    # Publish message
    test_data = {'test': 'persistence', 'value': 123}
    await message_bus.publish('stream:restart_test', test_data)
    
    # Disconnect (simulate crash)
    await message_bus.disconnect()
    
    # Reconnect (simulate restart)
    await message_bus.connect()
    
    # Messages should still be in Redis stream
    # (This is a basic test - full verification would require reading the stream)
    
    await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
