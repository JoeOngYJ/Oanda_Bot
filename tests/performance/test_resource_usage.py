#!/usr/bin/env python3
"""Performance Test: Resource Usage"""

import asyncio
import pytest
import psutil
import time

from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_memory_stability():
    """Verify memory usage is stable over time"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        process = psutil.Process()
        memory_samples = []
        
        # Monitor for 30 seconds
        for i in range(30):
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)
            await asyncio.sleep(1)
        
        # Verify stable (no significant growth)
        initial_memory = memory_samples[0]
        final_memory = memory_samples[-1]
        growth = final_memory - initial_memory
        
        print(f"\nMemory: Initial={initial_memory:.2f}MB, Final={final_memory:.2f}MB, Growth={growth:.2f}MB")
        
        # Allow up to 50MB growth
        assert growth < 50, f"Memory growth too high: {growth:.2f}MB"
        
    finally:
        await message_bus.disconnect()


@pytest.mark.asyncio
async def test_cpu_usage():
    """Verify CPU usage is reasonable"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        cpu_samples = []
        
        # Monitor for 10 seconds
        for i in range(10):
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_samples.append(cpu_percent)
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        print(f"\nAverage CPU usage: {avg_cpu:.2f}%")
        
        # Assert reasonable CPU usage
        assert avg_cpu < 50, f"CPU usage too high: {avg_cpu:.2f}%"
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
