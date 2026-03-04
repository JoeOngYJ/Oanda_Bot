#!/usr/bin/env python3
"""Tests for OrderManager"""

import pytest
import copy
from datetime import datetime
from decimal import Decimal

from oanda_bot.agents.execution.order_manager import OrderManager
from oanda_bot.utils.models import TradeSignal, OrderStatus, OrderType, Side, Instrument


@pytest.fixture
def order_manager():
    """Create OrderManager instance"""
    return OrderManager()


@pytest.fixture
def sample_signal():
    """Create sample trade signal"""
    return TradeSignal(
        signal_id="signal-123",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        entry_price=Decimal("1.08500"),
        stop_loss=Decimal("1.08300"),
        take_profit=Decimal("1.08900"),
        confidence=0.85,
        rationale="Test signal for execution agent testing",
        strategy_name="MA_Crossover",
        strategy_version="1.0.0",
        timestamp=datetime.utcnow()
    )


@pytest.mark.asyncio
async def test_create_order_from_signal(order_manager, sample_signal):
    """Test creating order from signal"""
    order = await order_manager.create_order_from_signal(sample_signal)

    assert order.signal_id == sample_signal.signal_id
    assert order.instrument == sample_signal.instrument
    assert order.side == sample_signal.side
    assert order.quantity == sample_signal.quantity
    assert order.stop_loss == sample_signal.stop_loss
    assert order.take_profit == sample_signal.take_profit
    assert order.order_type == OrderType.MARKET
    assert order.status == OrderStatus.PENDING
    assert order.order_id is not None
    assert order.idempotency_key == f"sig:{sample_signal.signal_id}"


@pytest.mark.asyncio
async def test_update_order_status(order_manager, sample_signal):
    """Test updating order status"""
    order = await order_manager.create_order_from_signal(sample_signal)

    await order_manager.update_order_status(
        order.order_id,
        OrderStatus.SUBMITTED,
        oanda_order_id="oanda-123"
    )

    updated_order = order_manager.get_order(order.order_id)
    assert updated_order.status == OrderStatus.SUBMITTED
    assert updated_order.oanda_order_id == "oanda-123"


@pytest.mark.asyncio
async def test_get_order(order_manager, sample_signal):
    """Test retrieving order by ID"""
    order = await order_manager.create_order_from_signal(sample_signal)

    retrieved_order = order_manager.get_order(order.order_id)
    assert retrieved_order is not None
    assert retrieved_order.order_id == order.order_id

    # Test non-existent order
    assert order_manager.get_order("non-existent") is None


@pytest.mark.asyncio
async def test_get_orders_by_status(order_manager, sample_signal):
    """Test filtering orders by status"""
    signal2 = copy.deepcopy(sample_signal)
    signal2.signal_id = "signal-456"
    order1 = await order_manager.create_order_from_signal(sample_signal)
    order2 = await order_manager.create_order_from_signal(signal2)

    await order_manager.update_order_status(order1.order_id, OrderStatus.SUBMITTED)

    pending_orders = order_manager.get_orders_by_status(OrderStatus.PENDING)
    submitted_orders = order_manager.get_orders_by_status(OrderStatus.SUBMITTED)

    assert len(pending_orders) == 1
    assert len(submitted_orders) == 1
    assert order2.order_id in pending_orders
    assert order1.order_id in submitted_orders


@pytest.mark.asyncio
async def test_duplicate_signal_returns_same_order(order_manager, sample_signal):
    """Duplicate signal IDs should map to the same order for idempotency."""
    order1 = await order_manager.create_order_from_signal(sample_signal)
    order2 = await order_manager.create_order_from_signal(sample_signal)

    assert order1.order_id == order2.order_id
    assert order_manager.has_processed_signal(sample_signal.signal_id)
