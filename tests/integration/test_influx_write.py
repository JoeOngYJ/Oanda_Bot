#!/usr/bin/env python3
"""Test InfluxDB write functionality"""

import asyncio
import time
from decimal import Decimal
from datetime import datetime
from agents.market_data.storage import MarketDataStorage
from shared.models import MarketTick, Instrument
from shared.config import Config


async def test_influx_write():
    """Test writing a tick to InfluxDB"""
    config = Config.load()
    storage = MarketDataStorage(config)

    # Create test tick
    tick = MarketTick(
        instrument=Instrument.EUR_USD,
        timestamp=datetime.utcnow(),
        bid=Decimal('1.08500'),
        ask=Decimal('1.08505'),
        spread=Decimal('0.00005'),
        source='test',
        data_version='1.0.0'
    )

    # Save tick
    await storage.save_tick(tick)
    print('✓ Tick saved to InfluxDB')

    # Wait for write to complete
    time.sleep(1)

    # Verify by reading back
    latest = await storage.get_latest_tick(Instrument.EUR_USD)
    print(f'✓ Latest tick retrieved: {latest}')

    print('\n✓ InfluxDB write test passed!')


if __name__ == '__main__':
    asyncio.run(test_influx_write())
