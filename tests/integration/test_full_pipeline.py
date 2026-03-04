#!/usr/bin/env python3
"""Full pipeline integration test"""

import asyncio
import pytest
from datetime import datetime
from decimal import Decimal

from oanda_bot.utils.models import MarketTick, Instrument, Side
from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_message_bus_connectivity():
    """Test basic message bus connectivity"""
    print("Testing message bus connectivity...")

    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    # Publish test message
    test_data = {'test': 'message', 'timestamp': datetime.utcnow().isoformat()}
    await message_bus.publish('stream:test', test_data)

    print("✓ Message bus connectivity verified")
    await message_bus.disconnect()


@pytest.mark.asyncio
async def test_market_tick_publishing():
    """Test publishing market ticks to message bus"""
    print("Testing market tick publishing...")

    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    # Create test tick
    test_tick = MarketTick(
        instrument=Instrument.EUR_USD,
        timestamp=datetime.utcnow(),
        bid=Decimal("1.08500"),
        ask=Decimal("1.08505"),
        spread=Decimal("0.00005"),
        source="test",
        data_version="1.0.0"
    )

    # Publish tick
    await message_bus.publish('stream:market_data', test_tick.model_dump(mode='json'))

    print("✓ Market tick published successfully")
    await message_bus.disconnect()


if __name__ == '__main__':
    asyncio.run(test_message_bus_connectivity())
    asyncio.run(test_market_tick_publishing())
