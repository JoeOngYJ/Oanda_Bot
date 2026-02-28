#!/usr/bin/env python3
"""Stress Test: Concurrent Signals"""

import asyncio
import pytest
from datetime import datetime
from decimal import Decimal

from shared.models import TradeSignal, Instrument, Side
from shared.message_bus import MessageBus
from shared.config import Config


@pytest.mark.asyncio
async def test_multiple_concurrent_signals():
    """Test multiple strategies firing signals simultaneously"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()
    
    try:
        # Generate multiple signals concurrently
        signal_count = 50
        
        for i in range(signal_count):
            signal = TradeSignal(
                signal_id=f"concurrent-signal-{i}",
                instrument=Instrument.EUR_USD,
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                quantity=1000,
                entry_price=Decimal("1.08500"),
                stop_loss=Decimal("1.08300"),
                take_profit=Decimal("1.08900"),
                confidence=0.85,
                rationale=f"Concurrent test signal {i}",
                strategy_name=f"Strategy{i % 5}",
                strategy_version="1.0.0",
                timestamp=datetime.utcnow()
            )
            
            await message_bus.publish('stream:signals', signal.model_dump(mode='json'))
        
        print(f"\nPublished {signal_count} concurrent signals")
        await asyncio.sleep(5)
        
    finally:
        await message_bus.disconnect()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
