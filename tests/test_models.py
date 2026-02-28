"""
Unit tests for Pydantic data models.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from pydantic import ValidationError

from shared.models import (
    Instrument, Side, OrderType, OrderStatus,
    MarketTick, TradeSignal, RiskCheckResult,
    Order, Execution, Position, HealthMetric
)


class TestMarketTick:
    """Tests for MarketTick model"""

    def test_valid_market_tick(self):
        """Test creating a valid market tick"""
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.1000"),
            ask=Decimal("1.1002"),
            spread=Decimal("0.0002")
        )
        assert tick.instrument == Instrument.EUR_USD
        assert tick.bid == Decimal("1.1000")
        assert tick.ask == Decimal("1.1002")
        assert tick.spread == Decimal("0.0002")

    def test_spread_auto_calculation(self):
        """Test automatic spread calculation"""
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.1000"),
            ask=Decimal("1.1002"),
            spread=None
        )
        assert tick.spread == Decimal("0.0002")

    def test_invalid_bid_ask(self):
        """Test that ask must be >= bid"""
        with pytest.raises(ValidationError):
            MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=datetime.utcnow(),
                bid=Decimal("1.1002"),
                ask=Decimal("1.1000"),
                spread=Decimal("0.0002")
            )

    def test_negative_price_rejected(self):
        """Test that negative prices are rejected"""
        with pytest.raises(ValidationError):
            MarketTick(
                instrument=Instrument.EUR_USD,
                timestamp=datetime.utcnow(),
                bid=Decimal("-1.1000"),
                ask=Decimal("1.1002"),
                spread=Decimal("0.0002")
            )

    def test_json_serialization(self):
        """Test that model can be serialized to JSON"""
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.utcnow(),
            bid=Decimal("1.1000"),
            ask=Decimal("1.1002"),
            spread=Decimal("0.0002")
        )
        json_data = tick.model_dump(mode='json')
        assert isinstance(json_data['bid'], str)
        assert isinstance(json_data['ask'], str)


class TestTradeSignal:
    """Tests for TradeSignal model"""

    def test_valid_trade_signal(self):
        """Test creating a valid trade signal"""
        signal = TradeSignal(
            signal_id="test-123",
            instrument=Instrument.EUR_USD,
            side=Side.BUY,
            quantity=1000,
            confidence=0.85,
            rationale="Test signal rationale",
            strategy_name="TestStrategy",
            strategy_version="1.0.0",
            timestamp=datetime.utcnow()
        )
        assert signal.signal_id == "test-123"
        assert signal.quantity == 1000
        assert signal.confidence == 0.85

    def test_invalid_quantity(self):
        """Test that quantity must be positive"""
        with pytest.raises(ValidationError):
            TradeSignal(
                signal_id="test-123",
                instrument=Instrument.EUR_USD,
                side=Side.BUY,
                quantity=0,
                confidence=0.85,
                rationale="Test signal",
                strategy_name="TestStrategy",
                strategy_version="1.0.0",
                timestamp=datetime.utcnow()
            )

    def test_invalid_confidence(self):
        """Test that confidence must be between 0 and 1"""
        with pytest.raises(ValidationError):
            TradeSignal(
                signal_id="test-123",
                instrument=Instrument.EUR_USD,
                side=Side.BUY,
                quantity=1000,
                confidence=1.5,
                rationale="Test signal",
                strategy_name="TestStrategy",
                strategy_version="1.0.0",
                timestamp=datetime.utcnow()
            )

    def test_short_rationale_rejected(self):
        """Test that rationale must be at least 10 characters"""
        with pytest.raises(ValidationError):
            TradeSignal(
                signal_id="test-123",
                instrument=Instrument.EUR_USD,
                side=Side.BUY,
                quantity=1000,
                confidence=0.85,
                rationale="Short",
                strategy_name="TestStrategy",
                strategy_version="1.0.0",
                timestamp=datetime.utcnow()
            )


class TestRiskCheckResult:
    """Tests for RiskCheckResult model"""

    def test_approved_risk_check(self):
        """Test creating an approved risk check result"""
        result = RiskCheckResult(
            signal_id="test-123",
            approved=True,
            reasons=[],
            warnings=["Low confidence"],
            risk_metrics={"exposure": 10000},
            timestamp=datetime.utcnow()
        )
        assert result.approved is True
        assert len(result.warnings) == 1

    def test_rejected_risk_check(self):
        """Test creating a rejected risk check result"""
        result = RiskCheckResult(
            signal_id="test-123",
            approved=False,
            reasons=["Exceeds daily loss limit"],
            warnings=[],
            risk_metrics={},
            timestamp=datetime.utcnow()
        )
        assert result.approved is False
        assert len(result.reasons) == 1


class TestOrder:
    """Tests for Order model"""

    def test_valid_order(self):
        """Test creating a valid order"""
        now = datetime.utcnow()
        order = Order(
            order_id="order-123",
            signal_id="signal-123",
            instrument=Instrument.EUR_USD,
            side=Side.BUY,
            quantity=1000,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now
        )
        assert order.order_id == "order-123"
        assert order.status == OrderStatus.PENDING


class TestExecution:
    """Tests for Execution model"""

    def test_valid_execution(self):
        """Test creating a valid execution"""
        execution = Execution(
            execution_id="exec-123",
            order_id="order-123",
            instrument=Instrument.EUR_USD,
            side=Side.BUY,
            filled_quantity=1000,
            fill_price=Decimal("1.1002"),
            commission=Decimal("2.50"),
            timestamp=datetime.utcnow(),
            oanda_transaction_id="oanda-tx-123"
        )
        assert execution.filled_quantity == 1000
        assert execution.fill_price == Decimal("1.1002")


class TestPosition:
    """Tests for Position model"""

    def test_long_position(self):
        """Test creating a long position"""
        position = Position(
            position_id="pos-123",
            instrument=Instrument.EUR_USD,
            side=Side.BUY,
            quantity=1000,
            entry_price=Decimal("1.1000"),
            current_price=Decimal("1.1050"),
            unrealized_pnl=Decimal("50.00"),
            realized_pnl=Decimal("0.00")
        )
        assert position.side == Side.BUY
        assert position.quantity == 1000
        assert position.average_price == Decimal("1.1000")  # Test alias

    def test_short_position(self):
        """Test creating a short position"""
        position = Position(
            position_id="pos-456",
            instrument=Instrument.EUR_USD,
            side=Side.SELL,
            quantity=1000,
            entry_price=Decimal("1.1000"),
            current_price=Decimal("1.0950"),
            unrealized_pnl=Decimal("50.00"),
            realized_pnl=Decimal("0.00")
        )
        assert position.side == Side.SELL
        assert position.quantity == 1000
        assert position.average_price == Decimal("1.1000")  # Test alias


class TestHealthMetric:
    """Tests for HealthMetric model"""

    def test_healthy_metric(self):
        """Test creating a healthy metric"""
        metric = HealthMetric(
            component="market_data",
            metric_name="latency",
            value=0.5,
            timestamp=datetime.utcnow(),
            status="healthy"
        )
        assert metric.status == "healthy"

    def test_critical_metric(self):
        """Test creating a critical metric"""
        metric = HealthMetric(
            component="execution",
            metric_name="error_rate",
            value=0.25,
            timestamp=datetime.utcnow(),
            status="critical",
            metadata={"details": "High error rate"}
        )
        assert metric.status == "critical"
        assert "details" in metric.metadata
