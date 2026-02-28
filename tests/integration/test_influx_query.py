#!/usr/bin/env python3
"""Test InfluxDB query functionality"""

import asyncio
from datetime import datetime, timedelta
from agents.market_data.storage import MarketDataStorage
from shared.models import Instrument
from shared.config import Config


async def test_influx_query():
    """Test querying ticks from InfluxDB"""
    config = Config.load()
    storage = MarketDataStorage(config)

    # Query last hour of data
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    ticks = await storage.query_ticks(Instrument.EUR_USD, start_time, end_time)

    print(f'✓ Found {len(ticks)} ticks in last hour')

    if ticks:
        print(f'✓ Latest tick: {ticks[-1]}')
    else:
        print('ℹ No ticks found (this is normal if no data has been written yet)')

    print('\n✓ InfluxDB query test passed!')


if __name__ == '__main__':
    asyncio.run(test_influx_query())
