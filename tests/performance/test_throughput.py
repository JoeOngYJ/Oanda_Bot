#!/usr/bin/env python3
"""Performance Test: Throughput"""

import asyncio
import pytest
import time
from datetime import datetime
from decimal import Decimal

from shared.models import MarketTick, Instrument
from shared.message_bus import MessageBus
from shared.config import Config


def generate_random_tick(instrument=Instrument.EUR_USD):
    """Generate a random market tick for testing"""
    base_price = Decimal("1.08500")
    return MarketTick(
        instrument=instrument,
        timestamp=datetime.utcnow(),
        bid=base_price,
        ask=base_price + Decimal("0.00005"),
        spread=Decimal("0.00005"),
        source="test",
        data_version="1.0.0"
    )


@pytest.mark.asyncio
async def test_market_data_throughput():
    """Measure market data processing throughput"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        start_time = time.time()
        tick_count = 1000
        
        # Publish ticks rapidly
        for i in range(tick_count):
            tick = generate_random_tick()
            await message_bus.publish('stream:market_data', tick.model_dump(mode='json'))
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = tick_count / duration
        
        print(f"\nThroughput: {throughput:.2f} ticks/second")
        print(f"Published {tick_count} ticks in {duration:.2f} seconds")
        
        # Assert meets requirement (100+ ticks/sec)
        assert throughput >= 100, f"Throughput too low: {throughput:.2f} ticks/sec"
        
    finally:
        await message_bus.disconnect()


@pytest.mark.asyncio
async def test_signal_processing_throughput():
    """Measure signal processing throughput"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        start_time = time.time()
        signal_count = 100
        
        # Publish signals
        for i in range(signal_count):
            signal_data = {
                'signal_id': f'test-signal-{i}',
                'instrument': 'EUR_USD',
                'side': 'BUY',
                'quantity': 1000,
                'entry_price': '1.08500',
                'stop_loss': '1.08300',
                'take_profit': '1.08900',
                'confidence': 0.85,
                'rationale': 'Test signal',
                'strategy_name': 'TestStrategy',
                'strategy_version': '1.0.0',
                'timestamp': datetime.utcnow().isoformat()
            }
            await message_bus.publish('stream:signals', signal_data)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = signal_count / duration
        
        print(f"\nSignal throughput: {throughput:.2f} signals/second")
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
