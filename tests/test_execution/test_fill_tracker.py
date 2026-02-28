#!/usr/bin/env python3
"""Tests for FillTracker"""

import pytest
from datetime import datetime
from decimal import Decimal

from agents.execution.fill_tracker import FillTracker
from shared.models import Execution, Order, OrderStatus, OrderType, Side, Instrument


@pytest.fixture
def fill_tracker():
    """Create FillTracker instance"""
    return FillTracker()


@pytest.fixture
def sample_order():
    """Create sample order"""
    return Order(
        order_id="order-123",
        signal_id="signal-123",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        order_type=OrderType.MARKET,
        price=None,
        stop_loss=Decimal("1.08300"),
        take_profit=Decimal("1.08900"),
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        oanda_order_id=None
    )


@pytest.fixture
def sample_execution(sample_order):
    """Create sample execution"""
    return Execution(
        execution_id="exec-123",
        order_id=sample_order.order_id,
        instrument=sample_order.instrument,
        side=sample_order.side,
        filled_quantity=1000,
        fill_price=Decimal("1.08525"),
        commission=Decimal("0.50"),
        timestamp=datetime.utcnow(),
        oanda_transaction_id="oanda-tx-123"
    )


def test_record_execution(fill_tracker, sample_execution):
    """Test recording an execution"""
    fill_tracker.record_execution(sample_execution)

    assert sample_execution.execution_id in fill_tracker.executions
    assert sample_execution.order_id in fill_tracker.order_fills
    assert sample_execution.execution_id in fill_tracker.order_fills[sample_execution.order_id]


def test_get_order_executions(fill_tracker, sample_execution):
    """Test retrieving executions for an order"""
    fill_tracker.record_execution(sample_execution)

    executions = fill_tracker.get_order_executions(sample_execution.order_id)
    assert len(executions) == 1
    assert executions[0].execution_id == sample_execution.execution_id


def test_calculate_order_fill_percentage(fill_tracker, sample_order, sample_execution):
    """Test calculating fill percentage"""
    # Order for 1000 units, execution of 1000 units = 100%
    fill_tracker.record_execution(sample_execution)

    fill_pct = fill_tracker.calculate_order_fill_percentage(sample_order)
    assert fill_pct == 1.0

    # Test partial fill
    partial_execution = Execution(
        execution_id="exec-456",
        order_id="order-456",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        filled_quantity=500,
        fill_price=Decimal("1.08525"),
        commission=Decimal("0.25"),
        timestamp=datetime.utcnow(),
        oanda_transaction_id="oanda-tx-456"
    )

    partial_order = Order(
        order_id="order-456",
        signal_id="signal-456",
        instrument=Instrument.EUR_USD,
        side=Side.BUY,
        quantity=1000,
        order_type=OrderType.MARKET,
        price=None,
        stop_loss=None,
        take_profit=None,
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        oanda_order_id=None
    )

    fill_tracker.record_execution(partial_execution)
    fill_pct = fill_tracker.calculate_order_fill_percentage(partial_order)
    assert fill_pct == 0.5


def test_get_total_filled_quantity(fill_tracker, sample_execution):
    """Test getting total filled quantity"""
    fill_tracker.record_execution(sample_execution)

    total_filled = fill_tracker.get_total_filled_quantity(sample_execution.order_id)
    assert total_filled == 1000


def test_get_average_fill_price(fill_tracker, sample_execution):
    """Test calculating average fill price"""
    fill_tracker.record_execution(sample_execution)

    avg_price = fill_tracker.get_average_fill_price(sample_execution.order_id)
    assert avg_price == Decimal("1.08525")

    # Test with multiple fills
    execution2 = Execution(
        execution_id="exec-456",
        order_id=sample_execution.order_id,
        instrument=sample_execution.instrument,
        side=sample_execution.side,
        filled_quantity=500,
        fill_price=Decimal("1.08600"),
        commission=Decimal("0.25"),
        timestamp=datetime.utcnow(),
        oanda_transaction_id="oanda-tx-456"
    )

    fill_tracker.record_execution(execution2)
    avg_price = fill_tracker.get_average_fill_price(sample_execution.order_id)

    # Average of 1000@1.08525 and 500@1.08600
    expected = (Decimal("1.08525") * 1000 + Decimal("1.08600") * 500) / 1500
    assert abs(avg_price - expected) < Decimal("0.00001")

