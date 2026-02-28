#!/usr/bin/env python3
"""Execution Agent - Executes approved trades on Oanda"""

import asyncio
import logging
import os
from typing import Optional
from datetime import datetime
from decimal import Decimal
import uuid

from shared.config import Config
from shared.message_bus import MessageBus
from shared.models import (
    Execution,
    ExecutionControlCommand,
    Order,
    OrderStatus,
    RiskCheckResult,
)

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
        self.kill_switch_active = False
        self.shadow_mode = os.getenv("EXECUTION_SHADOW_MODE", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.live_execution_enabled = os.getenv("EXECUTION_LIVE_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._executed_signal_ids = set()

        logger.info(
            "ExecutionAgent initialized",
            extra={
                "kill_switch_active": self.kill_switch_active,
                "shadow_mode": self.shadow_mode,
                "live_execution_enabled": self.live_execution_enabled,
            },
        )

    async def start(self) -> None:
        """Start execution agent and subscribe to risk checks"""
        logger.info("Starting ExecutionAgent...")

        await self.message_bus.connect()
        self.running = True
        await self._apply_startup_guardrails()
        await self._persist_safety_state()

        logger.info("ExecutionAgent started - subscribing to risk checks")

        await asyncio.gather(
            self._process_risk_checks(),
            self._process_control_commands(),
            return_exceptions=True,
        )

    async def _apply_startup_guardrails(self) -> None:
        """Enforce explicit opt-in before sending live orders."""
        env = str(self.config.oanda.environment).lower()
        if env == "live" and not self.shadow_mode and not self.live_execution_enabled:
            self.kill_switch_active = True
            logger.critical(
                "Live execution guardrail activated: set EXECUTION_LIVE_ENABLED=true or enable shadow mode.",
                extra={
                    "environment": env,
                    "shadow_mode": self.shadow_mode,
                    "live_execution_enabled": self.live_execution_enabled,
                },
            )
            await self._publish_alert(
                severity="critical",
                message="Live execution blocked by guardrail",
                environment=env,
                shadow_mode=self.shadow_mode,
                live_execution_enabled=self.live_execution_enabled,
            )
            await self._persist_safety_state()

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

    async def _process_control_commands(self) -> None:
        """Listen for execution control commands (kill-switch and shadow mode)."""
        async for message in self.message_bus.subscribe("execution_control"):
            if not self.running:
                break

            try:
                command = ExecutionControlCommand(**message)
                await self._apply_control_command(command)
            except asyncio.CancelledError:
                logger.info("Execution control processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing execution control command: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _apply_control_command(self, command: ExecutionControlCommand) -> None:
        if command.action == "kill_switch_on":
            self.kill_switch_active = True
        elif command.action == "kill_switch_off":
            self.kill_switch_active = False
        elif command.action == "shadow_mode_on":
            self.shadow_mode = True
        elif command.action == "shadow_mode_off":
            self.shadow_mode = False
        else:
            return

        logger.warning(
            "Execution control command applied",
            extra={
                "action": command.action,
                "requested_by": command.requested_by,
                "reason": command.reason,
                "kill_switch_active": self.kill_switch_active,
                "shadow_mode": self.shadow_mode,
            },
        )
        await self._persist_safety_state()
        await self._publish_alert(
            severity="warning",
            message=f"Execution control: {command.action}",
            action=command.action,
            requested_by=command.requested_by,
            reason=command.reason,
            kill_switch_active=self.kill_switch_active,
            shadow_mode=self.shadow_mode,
        )

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
            if risk_check.signal is None:
                logger.error(
                    "Approved risk check missing embedded signal",
                    extra={"signal_id": risk_check.signal_id},
                )
                await self._publish_alert(
                    severity="critical",
                    message="Approved risk check missing signal payload",
                    signal_id=risk_check.signal_id,
                )
                return

            if self.kill_switch_active:
                logger.warning(
                    "Kill-switch active; skipping execution",
                    extra={"signal_id": risk_check.signal_id},
                )
                await self._publish_alert(
                    severity="critical",
                    message="Execution blocked by kill-switch",
                    signal_id=risk_check.signal_id,
                )
                return

            claimed = await self._claim_signal(risk_check.signal_id)
            if not claimed:
                logger.info(
                    "Duplicate approved signal ignored (durable idempotency)",
                    extra={"signal_id": risk_check.signal_id},
                )
                await self._publish_alert(
                    severity="info",
                    message="Duplicate approved signal ignored",
                    signal_id=risk_check.signal_id,
                )
                return

            # Create order from signal
            order = await self.order_manager.create_order_from_signal(risk_check.signal)
            if order.status in {
                OrderStatus.SUBMITTED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.FAILED,
                OrderStatus.REJECTED,
            }:
                logger.info(
                    "Order already processed; skipping signal",
                    extra={
                        "signal_id": risk_check.signal_id,
                        "order_id": order.order_id,
                        "status": order.status.value,
                    },
                )
                await self._release_signal_claim(risk_check.signal_id)
                return

            # Execute order
            execution = await self._execute_order(order)

            if execution:
                self._executed_signal_ids.add(risk_check.signal_id)
                await self._mark_signal_executed(risk_check.signal_id, order.order_id)
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
            else:
                await self._release_signal_claim(risk_check.signal_id)

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
            await self._release_signal_claim(risk_check.signal_id)

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

        if self.shadow_mode:
            execution = await self._execute_shadow(order)
        else:
            # Execute with retry logic
            execution = await self._execute_with_retry(order)

        if execution is None:
            # Update order status to FAILED
            await self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FAILED
            )

        return execution

    async def _execute_shadow(self, order: Order) -> Execution:
        """Simulate a fill without sending an order to broker."""
        signal = self.order_manager.get_signal(order.signal_id)
        fill_price = signal.entry_price if signal and signal.entry_price is not None else Decimal("0")
        execution = Execution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            commission=Decimal("0"),
            timestamp=datetime.utcnow(),
            oanda_transaction_id=f"shadow-{order.order_id}",
            execution_mode="shadow",
            metadata={"reason": "shadow_mode_enabled"},
        )
        logger.info(
            "Shadow execution completed",
            extra={
                "order_id": order.order_id,
                "signal_id": order.signal_id,
                "fill_price": str(fill_price),
            },
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

    async def _claim_signal(self, signal_id: str) -> bool:
        """
        Claim a signal for execution processing.

        Returns False if already executed/inflight (durable, Redis-backed when available).
        """
        if signal_id in self._executed_signal_ids:
            return False

        redis_client = self.message_bus.redis_client
        if redis_client is None:
            # Fallback if message bus client is unavailable.
            self._executed_signal_ids.add(signal_id)
            return True

        executed_key = f"execution:signal:executed:{signal_id}"
        inflight_key = f"execution:signal:inflight:{signal_id}"

        if await redis_client.exists(executed_key):
            return False
        claimed = await redis_client.set(inflight_key, "1", nx=True, ex=600)
        return bool(claimed)

    async def _release_signal_claim(self, signal_id: str) -> None:
        """Release inflight claim after failed processing."""
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            self._executed_signal_ids.discard(signal_id)
            return
        inflight_key = f"execution:signal:inflight:{signal_id}"
        await redis_client.delete(inflight_key)

    async def _mark_signal_executed(self, signal_id: str, order_id: str) -> None:
        """Persist executed signal marker and clear inflight key."""
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            return
        executed_key = f"execution:signal:executed:{signal_id}"
        inflight_key = f"execution:signal:inflight:{signal_id}"
        await redis_client.set(executed_key, order_id, ex=60 * 60 * 24 * 30)
        await redis_client.delete(inflight_key)

    async def _persist_safety_state(self) -> None:
        """Persist execution safety flags for operator visibility."""
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            return
        live_guardrail_blocked = (
            str(self.config.oanda.environment).lower() == "live"
            and not self.shadow_mode
            and not self.live_execution_enabled
        )
        await redis_client.set("execution:state:kill_switch", "1" if self.kill_switch_active else "0")
        await redis_client.set("execution:state:shadow_mode", "1" if self.shadow_mode else "0")
        await redis_client.set(
            "execution:state:live_guardrail_blocked",
            "1" if live_guardrail_blocked else "0",
        )
        await redis_client.set("execution:state:updated_at", datetime.utcnow().isoformat())


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
