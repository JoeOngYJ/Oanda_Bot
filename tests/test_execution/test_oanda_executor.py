#!/usr/bin/env python3
"""Tests for OandaExecutor"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from decimal import Decimal

from agents.execution.oanda_executor import OandaExecutor
from shared.models import Order, OrderStatus, OrderType, Side, Instrument
from shared.config import Config


@pytest.fixture
def config():
    """Create test config"""
    config = Config.load()
    return config


@pytest.fixture
def oanda_executor(config):
    """Create OandaExecutor instance"""
    return OandaExecutor(config)


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


def test_build_order_payload_buy(oanda_executor, sample_order):
    """Test building order payload for buy order"""
    payload = oanda_executor._build_order_payload(sample_order)

    assert payload["order"]["instrument"] == "EUR_USD"
    assert payload["order"]["units"] == "1000"  # Positive for buy
    assert payload["order"]["type"] == "MARKET"
    assert payload["order"]["timeInForce"] == "FOK"
    assert payload["order"]["stopLossOnFill"]["price"] == "1.08300"
    assert payload["order"]["takeProfitOnFill"]["price"] == "1.08900"


def test_build_order_payload_sell(oanda_executor):
    """Test building order payload for sell order"""
    sell_order = Order(
        order_id="order-456",
        signal_id="signal-456",
        instrument=Instrument.EUR_USD,
        side=Side.SELL,
        quantity=1000,
        order_type=OrderType.MARKET,
        price=None,
        stop_loss=Decimal("1.08900"),
        take_profit=Decimal("1.08300"),
        status=OrderStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        oanda_order_id=None
    )

    payload = oanda_executor._build_order_payload(sell_order)

    assert payload["order"]["units"] == "-1000"  # Negative for sell
    assert payload["order"]["stopLossOnFill"]["price"] == "1.08900"
    assert payload["order"]["takeProfitOnFill"]["price"] == "1.08300"


def test_parse_oanda_time(oanda_executor):
    """Test parsing Oanda RFC3339 timestamp"""
    time_str = "2024-02-08T12:34:56.789Z"
    dt = oanda_executor._parse_oanda_time(time_str)

    assert dt.year == 2024
    assert dt.month == 2
    assert dt.day == 8
    assert dt.hour == 12
    assert dt.minute == 34
    assert dt.second == 56


def test_parse_execution_fill_response(oanda_executor, sample_order):
    """Test parsing Oanda fill response"""
    mock_response = {
        "orderFillTransaction": {
            "id": "oanda-tx-123",
            "units": "1000",
            "price": "1.08525",
            "financing": "0.50",
            "time": "2024-02-08T12:34:56.789Z"
        }
    }

    execution = oanda_executor._parse_execution(mock_response, sample_order)

    assert execution.order_id == sample_order.order_id
    assert execution.instrument == sample_order.instrument
    assert execution.side == sample_order.side
    assert execution.filled_quantity == 1000
    assert execution.fill_price == Decimal("1.08525")
    assert execution.commission == Decimal("0.50")
    assert execution.oanda_transaction_id == "oanda-tx-123"


def test_parse_execution_rejection_response(oanda_executor, sample_order):
    """Test parsing Oanda rejection response"""
    mock_response = {
        "orderRejectTransaction": {
            "id": "oanda-tx-456",
            "rejectReason": "INSUFFICIENT_MARGIN"
        }
    }

    with pytest.raises(Exception) as exc_info:
        oanda_executor._parse_execution(mock_response, sample_order)

    assert "INSUFFICIENT_MARGIN" in str(exc_info.value)
