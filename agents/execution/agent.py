#!/usr/bin/env python3
"""Execution Agent - Executes approved trades on Oanda"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from shared.config import Config
from shared.message_bus import MessageBus
from shared.models import RiskCheckResult, Order, Execution, OrderStatus

from .order_manager import OrderManager
from .oanda_executor import OandaExecutor
from .fill_tracker import FillTracker

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Execution Agent - Takes approved trades and executes them on Oanda.

    Responsibilities:
    - Subscribe to approved signals from Risk Agent
    - Create orders from approved signals
    - Execute orders via Oanda API
    - Track order status and fills
    - Publish execution reports
    - Handle errors and retries
    """

    def __init__(self, config: Config):
        """
        Initialize Execution Agent.

        Args:
            config: System configuration
        """
        self.config = config
        self.message_bus = MessageBus(config)

        self.order_manager = OrderManager()
        self.oanda_executor = OandaExecutor(config)
        self.fill_tracker = FillTracker()

        self.running = False
        self.max_retries = 3
        self.retry_delay = 2.0

        logger.info("ExecutionAgent initialized")

    async def start(self) -> None:
        """Start execution agent and subscribe to risk checks"""
        logger.info("Starting ExecutionAgent...")

        await self.message_bus.connect()
        self.running = True

        logger.info("ExecutionAgent started - subscribing to risk checks")

        # Subscribe to approved risk checks
        await self._process_risk_checks()

    async def stop(self) -> None:
        """Gracefully shutdown execution agent"""
        logger.info("Stopping ExecutionAgent...")

        self.running = False
        await self.message_bus.disconnect()

        logger.info("ExecutionAgent stopped")

    async def _process_risk_checks(self) -> None:
        """Subscribe to risk checks and process approved signals"""
        async for message in self.message_bus.subscribe('risk_checks'):
            if not self.running:
                break

            try:
                # Parse risk check result
                risk_check = RiskCheckResult(**message)

                # Only process approved signals
                if risk_check.approved:
                    await self._handle_approved_signal(risk_check)

            except asyncio.CancelledError:
                logger.info("Risk check processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing risk checks: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _handle_approved_signal(self, risk_check: RiskCheckResult) -> None:
        """
        Handle approved signal by creating and executing order.

        Args:
            risk_check: Approved risk check result
        """
        logger.info(
            f"Processing approved signal: {risk_check.signal_id}",
            extra={"signal_id": risk_check.signal_id}
        )

        try:
            # Create order from signal
            order = await self.order_manager.create_order_from_signal(risk_check.signal)

            # Execute order
            execution = await self._execute_order(order)

            if execution:
                # Record execution
                self.fill_tracker.record_execution(execution)

                # Update order status
                await self.order_manager.update_order_status(
                    order.order_id,
                    OrderStatus.FILLED,
                    execution.oanda_transaction_id
                )

                # Publish execution
                await self._publish_execution(execution)

                # Send success alert
                await self._publish_alert(
                    severity='info',
                    message=f'Order executed: {order.order_id}',
                    order_id=order.order_id,
                    filled_quantity=execution.filled_quantity,
                    fill_price=float(execution.fill_price)
                )

        except Exception as e:
            logger.error(
                f"Error handling approved signal {risk_check.signal_id}: {e}",
                exc_info=True
            )
            # Send failure alert
            await self._publish_alert(
                severity='critical',
                message=f'Order execution failed: {e}',
                signal_id=risk_check.signal_id,
                error=str(e)
            )

    async def _execute_order(self, order: Order) -> Optional[Execution]:
        """
        Execute order with retry logic.

        Args:
            order: Order to execute

        Returns:
            Execution on success, None on failure
        """
        logger.info(
            f"Executing order: {order.order_id}",
            extra={
                "order_id": order.order_id,
                "instrument": order.instrument.value,
                "side": order.side.value,
                "quantity": order.quantity
            }
        )

        # Update order status to SUBMITTED
        await self.order_manager.update_order_status(
            order.order_id,
            OrderStatus.SUBMITTED
        )

        # Execute with retry logic
        execution = await self._execute_with_retry(order)

        if execution is None:
            # Update order status to FAILED
            await self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FAILED
            )

        return execution

    async def _execute_with_retry(self, order: Order) -> Optional[Execution]:
        """
        Execute order with retry logic.

        Args:
            order: Order to execute

        Returns:
            Execution on success, None on failure
        """
        for attempt in range(self.max_retries):
            try:
                execution = await self.oanda_executor.execute(order)
                return execution

            except Exception as e:
                error_msg = str(e).lower()

                # Check if error is retryable (network errors, timeouts)
                is_retryable = any(keyword in error_msg for keyword in [
                    'timeout', 'connection', 'network', 'unavailable'
                ])

                if is_retryable and attempt < self.max_retries - 1:
                    # Retry on network errors
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.max_retries}, retrying...",
                        extra={
                            "order_id": order.order_id,
                            "attempt": attempt + 1,
                            "error": str(e)
                        }
                    )
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Don't retry on business logic errors or max retries exceeded
                    logger.error(
                        f"Non-retryable error or max retries exceeded for order {order.order_id}",
                        extra={
                            "order_id": order.order_id,
                            "attempt": attempt + 1,
                            "error": str(e)
                        },
                        exc_info=True
                    )
                    return None

        return None

    async def _publish_execution(self, execution: Execution) -> None:
        """
        Publish execution to message bus.

        Args:
            execution: Execution to publish
        """
        await self.message_bus.publish(
            'stream:executions',
            execution.model_dump(mode='json')
        )

        logger.info(
            f"Execution published: {execution.execution_id}",
            extra={
                "execution_id": execution.execution_id,
                "order_id": execution.order_id,
                "filled_quantity": execution.filled_quantity,
                "fill_price": float(execution.fill_price)
            }
        )

    async def _publish_alert(self, severity: str, message: str, **kwargs) -> None:
        """
        Publish alert to message bus.

        Args:
            severity: Alert severity (info, warning, critical)
            message: Alert message
            **kwargs: Additional alert data
        """
        alert_data = {
            'component': 'execution',
            'severity': severity,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }

        await self.message_bus.publish('stream:alerts', alert_data)

        logger.info(
            f"Alert published: {severity} - {message}",
            extra=alert_data
        )


async def main():
    """Main entry point for Execution Agent"""
    config = Config.load()
    agent = ExecutionAgent(config)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()


if __name__ == '__main__':
    asyncio.run(main())



