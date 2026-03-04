#!/usr/bin/env python3
"""Order lifecycle and state management"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Optional
from decimal import Decimal

from oanda_bot.utils.models import Order, OrderStatus, OrderType, TradeSignal, Side

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle and state transitions"""

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.signal_cache: Dict[str, TradeSignal] = {}
        self.signal_to_order_id: Dict[str, str] = {}

    async def create_order_from_signal(self, signal: TradeSignal) -> Order:
        """
        Create Order from approved TradeSignal.

        Args:
            signal: TradeSignal to convert to order

        Returns:
            Order object ready for execution
        """
        if signal.signal_id in self.signal_to_order_id:
            existing_order_id = self.signal_to_order_id[signal.signal_id]
            existing = self.orders.get(existing_order_id)
            if existing is not None:
                logger.info(
                    f"Duplicate signal received, reusing existing order: {existing_order_id}",
                    extra={"signal_id": signal.signal_id, "order_id": existing_order_id},
                )
                return existing

        # Cache signal for reference
        self.signal_cache[signal.signal_id] = signal

        order = Order(
            order_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            instrument=signal.instrument,
            side=signal.side,
            quantity=signal.quantity,
            order_type=OrderType.MARKET,  # Start with market orders
            price=None,  # Market orders don't need price
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            oanda_order_id=None,
            idempotency_key=f"sig:{signal.signal_id}",
        )

        self.orders[order.order_id] = order
        self.signal_to_order_id[signal.signal_id] = order.order_id

        logger.info(
            f"Order created from signal: {order.order_id}",
            extra={
                "order_id": order.order_id,
                "signal_id": signal.signal_id,
                "instrument": signal.instrument.value,
                "side": signal.side.value,
                "quantity": signal.quantity
            }
        )

        return order

    def has_processed_signal(self, signal_id: str) -> bool:
        """Return True if a signal already has an internal order."""
        return signal_id in self.signal_to_order_id

    async def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        oanda_order_id: Optional[str] = None
    ) -> None:
        """
        Update order status and Oanda transaction ID.

        Args:
            order_id: Internal order ID
            status: New order status
            oanda_order_id: Oanda transaction ID (if available)

        Raises:
            ValueError: If order not found
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")

        order = self.orders[order_id]
        old_status = order.status
        order.status = status
        order.updated_at = datetime.utcnow()

        if oanda_order_id:
            order.oanda_order_id = oanda_order_id

        logger.info(
            f"Order status updated: {order_id} {old_status.value} → {status.value}",
            extra={
                "order_id": order_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "oanda_order_id": oanda_order_id
            }
        )

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Retrieve order by ID.

        Args:
            order_id: Internal order ID

        Returns:
            Order if found, None otherwise
        """
        return self.orders.get(order_id)

    def get_signal(self, signal_id: str) -> Optional[TradeSignal]:
        """
        Retrieve cached signal by ID.

        Args:
            signal_id: Signal ID

        Returns:
            TradeSignal if found, None otherwise
        """
        return self.signal_cache.get(signal_id)

    def get_all_orders(self) -> Dict[str, Order]:
        """Get all orders"""
        return self.orders.copy()

    def get_orders_by_status(self, status: OrderStatus) -> Dict[str, Order]:
        """
        Get all orders with specific status.

        Args:
            status: Order status to filter by

        Returns:
            Dictionary of orders with matching status
        """
        return {
            order_id: order
            for order_id, order in self.orders.items()
            if order.status == status
        }
