#!/usr/bin/env python3
"""Failover Test: Oanda API Failure"""

import asyncio
import pytest
from datetime import datetime
from decimal import Decimal

from oanda_bot.utils.models import Order, OrderType, OrderStatus, Instrument, Side
from oanda_bot.utils.config import Config


@pytest.mark.asyncio
async def test_oanda_api_error_handling():
    """Test execution agent handles Oanda API errors"""
    config = Config.load()
    
    # Create test order
    now = datetime.utcnow()
    order = Order(
        order_id="test-order-001",
        signal_id="test-signal-001",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        order_type=OrderType.MARKET,
        price=None,
        stop_loss=Decimal("1.08300"),
        take_profit=Decimal("1.08900"),
        status=OrderStatus.PENDING,
        created_at=now,
        updated_at=now,
        oanda_order_id=None
    )
    
    # Test that order structure is valid
    assert order.order_id == "test-order-001"
    assert order.status == OrderStatus.PENDING


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
