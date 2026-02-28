#!/usr/bin/env python3
"""Stress Test: High Volume"""

import asyncio
import pytest
import time
from datetime import datetime
from decimal import Decimal

from shared.models import MarketTick, Instrument
from shared.message_bus import MessageBus
from shared.config import Config


@pytest.mark.asyncio
async def test_high_volume_tick_processing():
    """Process 10,000 ticks in rapid succession"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        tick_count = 10000
        start_time = time.time()
        
        # Generate and publish ticks rapidly
        for i in range(tick_count):
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
            
            if i % 1000 == 0:
                print(f"Published {i} ticks")
        
        duration = time.time() - start_time
        print(f"\nProcessed {tick_count} ticks in {duration:.2f}s")
        
        # Wait for processing to complete
        await asyncio.sleep(5)
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
