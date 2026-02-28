#!/usr/bin/env python3
"""Integration test for execution flow"""

import asyncio
from datetime import datetime
from decimal import Decimal

from agents.execution.order_manager import OrderManager
from agents.execution.oanda_executor import OandaExecutor
from agents.execution.fill_tracker import FillTracker
from shared.models import TradeSignal, Side, Instrument
from shared.config import Config


async def test_execution_flow():
    """Test: Signal → Order → Execution flow"""
    print("Testing execution flow...")

    config = Config.load()
    order_manager = OrderManager()
    fill_tracker = FillTracker()

    # Create test signal
    signal = TradeSignal(
        signal_id="test-signal-123",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        entry_price=Decimal("1.08500"),
        stop_loss=Decimal("1.08300"),
        take_profit=Decimal("1.08900"),
        confidence=0.85,
        rationale="Integration test signal for execution flow",
        strategy_name="MA_Crossover",
        strategy_version="1.0.0",
        timestamp=datetime.utcnow()
    )

    # Create order from signal
    order = await order_manager.create_order_from_signal(signal)
    print(f"✓ Order created: {order.order_id}")

    # Verify order properties
    assert order.signal_id == signal.signal_id
    assert order.instrument == signal.instrument
    assert order.side == signal.side
    assert order.quantity == signal.quantity
    print(f"✓ Order properties verified")

    print("\n✓ Execution flow test passed!")


if __name__ == '__main__':
    asyncio.run(test_execution_flow())
