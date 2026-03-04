#!/usr/bin/env python3
"""Stress Test: Rapid Price Changes"""

import asyncio
import pytest
from datetime import datetime
from decimal import Decimal

from oanda_bot.utils.models import MarketTick, Instrument
from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_volatile_market_conditions():
    """Simulate rapid price movements (flash crash scenario)"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        base_price = Decimal("1.08500")
        
        # Generate ticks with 5% price swings
        for i in range(100):
            if i % 2 == 0:
                price = base_price * Decimal("1.05")
            else:
                price = base_price * Decimal("0.95")
            
            tick = MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=datetime.utcnow(),
                bid=price,
                ask=price + Decimal("0.00005"),
                spread=Decimal("0.00005"),
                source="test",
                data_version="1.0.0"
            )
            
            await message_bus.publish('stream:market_data', tick.model_dump(mode='json'))
            await asyncio.sleep(0.1)
        
        print("\nVolatile market simulation complete")
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
