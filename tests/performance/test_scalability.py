#!/usr/bin/env python3
"""Performance Test: Scalability"""

import asyncio
import pytest
import time
from datetime import datetime
from decimal import Decimal

from oanda_bot.utils.models import MarketTick, Instrument
from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_high_message_volume():
    """Test system handles high message volume"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        message_count = 5000
        start_time = time.time()
        
        # Publish many messages rapidly
        for i in range(message_count):
            tick = MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=datetime.utcnow(),
                bid=Decimal("1.08500"),
                ask=Decimal("1.08505"),
                spread=Decimal("0.00005"),
                source="test",
                data_version="1.0.0"
            )
            await message_bus.publish('stream:market_data', tick.model_dump(mode='json'))
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = message_count / duration
        
        print(f"\nProcessed {message_count} messages in {duration:.2f}s ({throughput:.2f} msg/s)")
        
        # System should handle at least 500 msg/s
        assert throughput >= 500, f"Throughput too low: {throughput:.2f} msg/s"
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
