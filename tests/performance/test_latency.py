#!/usr/bin/env python3
"""Performance Test: Latency"""

import asyncio
import pytest
import time
from datetime import datetime
from decimal import Decimal
import numpy as np

from shared.models import MarketTick, Instrument
from shared.message_bus import MessageBus
from shared.config import Config


@pytest.mark.asyncio
async def test_message_bus_latency():
    """Measure message bus publish/subscribe latency"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        latencies = []
        
        for i in range(100):
            publish_time = time.time()
            
            # Publish message with timestamp
            test_data = {
                'id': i,
                'publish_time': publish_time
            }
            await message_bus.publish('stream:latency_test', test_data)
            
            # Subscribe and measure latency
            async for msg in message_bus.subscribe('stream:latency_test'):
                if msg.get('id') == i:
                    receive_time = time.time()
                    latency = (receive_time - msg['publish_time']) * 1000  # ms
                    latencies.append(latency)
                    break
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"\nLatency - P50: {p50:.3f}ms, P95: {p95:.3f}ms, P99: {p99:.3f}ms")
        
        # Assert reasonable latency
        assert p95 < 100, f"P95 latency too high: {p95:.3f}ms"
        
    finally:
        await message_bus.disconnect()


@pytest.mark.asyncio
async def test_tick_processing_latency():
    """Measure tick processing latency"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        latencies = []
        
        for i in range(50):
            tick_time = datetime.utcnow()
            
            tick = MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=tick_time,
                bid=Decimal("1.08500"),
                ask=Decimal("1.08505"),
                spread=Decimal("0.00005"),
                source="test",
                data_version="1.0.0"
            )
            
            publish_time = time.time()
            await message_bus.publish('stream:market_data', tick.model_dump(mode='json'))
            
            # Small delay to allow processing
            await asyncio.sleep(0.1)
            
            process_time = time.time()
            latency = (process_time - publish_time) * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        print(f"\nAverage tick processing latency: {avg_latency:.3f}ms")
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
