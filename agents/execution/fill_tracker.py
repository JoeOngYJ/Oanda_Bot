#!/usr/bin/env python3
"""Track and reconcile order fills"""

import logging
from typing import Dict, List
from decimal import Decimal

from shared.models import Execution, Order

logger = logging.getLogger(__name__)


class FillTracker:
    """Tracks all executions and reconciles with orders"""

    def __init__(self):
        self.executions: Dict[str, Execution] = {}
        self.order_fills: Dict[str, List[str]] = {}  # order_id → [execution_ids]

    def record_execution(self, execution: Execution) -> None:
        """
        Record an execution.

        Args:
            execution: Execution to record
        """
        self.executions[execution.execution_id] = execution

        if execution.order_id not in self.order_fills:
            self.order_fills[execution.order_id] = []

        self.order_fills[execution.order_id].append(execution.execution_id)

        logger.info(
            f"Execution recorded: {execution.execution_id}",
            extra={
                "execution_id": execution.execution_id,
                "order_id": execution.order_id,
                "filled_quantity": execution.filled_quantity,
                "fill_price": float(execution.fill_price)
            }
        )

    def get_execution(self, execution_id: str) -> Execution:
        """
        Get execution by ID.

        Args:
            execution_id: Execution ID

        Returns:
            Execution object

        Raises:
            KeyError if execution not found
        """
        return self.executions[execution_id]

    def get_order_executions(self, order_id: str) -> List[Execution]:
        """
        Get all executions for an order.

        Args:
            order_id: Order ID

        Returns:
            List of executions for the order
        """
        execution_ids = self.order_fills.get(order_id, [])
        return [self.executions[eid] for eid in execution_ids]

    def calculate_order_fill_percentage(self, order: Order) -> float:
        """
        Calculate what % of order has been filled.

        Args:
            order: Order to check

        Returns:
            Fill percentage (0.0 to 1.0)
        """
        executions = self.get_order_executions(order.order_id)
        total_filled = sum(e.filled_quantity for e in executions)
        return total_filled / order.quantity if order.quantity > 0 else 0.0

    def get_total_filled_quantity(self, order_id: str) -> int:
        """
        Get total filled quantity for an order.

        Args:
            order_id: Order ID

        Returns:
            Total filled quantity
        """
        executions = self.get_order_executions(order_id)
        return sum(e.filled_quantity for e in executions)

    def get_average_fill_price(self, order_id: str) -> Decimal:
        """
        Calculate average fill price for an order.

        Args:
            order_id: Order ID

        Returns:
            Average fill price

        Raises:
            ValueError if no executions found
        """
        executions = self.get_order_executions(order_id)
        if not executions:
            raise ValueError(f"No executions found for order {order_id}")

        total_value = sum(e.fill_price * e.filled_quantity for e in executions)
        total_quantity = sum(e.filled_quantity for e in executions)

        return total_value / total_quantity if total_quantity > 0 else Decimal("0")

    def get_all_executions(self) -> Dict[str, Execution]:
        """Get all executions"""
        return self.executions.copy()
