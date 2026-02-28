"""
Risk Agent - Validates trade signals and monitors positions.
Enforces all risk limits and activates circuit breaker when needed.
"""

import asyncio
import logging
from typing import Dict
from decimal import Decimal
from shared.message_bus import MessageBus
from shared.models import TradeSignal, MarketTick, Order, Execution, Position
from shared.config import Config
from .limits import RiskLimits
from .pre_trade_checks import PreTradeChecker
from .position_monitor import PositionMonitor
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Validates trade signals against risk limits and monitors positions.
    Acts as gatekeeper between strategy signals and order execution.
    """

    def __init__(self, config: Config, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus

        # Initialize risk management components
        self.risk_limits = RiskLimits(config.risk)

        # Mock account balance - in production, fetch from broker
        self.account_balance = Decimal("10000.0")

        self.pre_trade_checker = PreTradeChecker(
            self.risk_limits,
            self.account_balance
        )
        self.position_monitor = PositionMonitor(self.risk_limits)
        self.circuit_breaker = CircuitBreaker(
            config.risk.circuit_breaker,
            self.risk_limits
        )

        self.running = False

    async def start(self):
        """Start risk agent with multiple concurrent tasks"""
        logger.info("Starting Risk Agent")
        self.running = True

        # Start concurrent tasks
        await asyncio.gather(
            self._process_signals(),
            self._monitor_positions(),
            self._process_executions(),
            return_exceptions=True
        )

    async def stop(self):
        """Gracefully stop risk agent"""
        logger.info("Stopping Risk Agent")
        self.running = False

    async def _process_signals(self):
        """
        Process incoming trade signals from strategy agent.
        Run pre-trade checks and publish approved orders.
        """
        logger.info("Starting signal processing")

        async for message in self.message_bus.subscribe('signals'):
            if not self.running:
                break

            try:
                # Parse trade signal
                signal = TradeSignal(**message)

                logger.info(
                    f"Received signal: {signal.strategy_name} - "
                    f"{signal.side.value} {signal.quantity} {signal.instrument.value}"
                )

                # Run pre-trade checks
                check_result = await self.pre_trade_checker.check_signal(signal)

                # Publish check result for monitoring
                await self.message_bus.publish(
                    'risk_checks',
                    check_result.model_dump(mode='json')
                )

                if check_result.approved:
                    # Convert signal to order
                    order = self._signal_to_order(signal)

                    # Publish order to execution agent
                    await self.message_bus.publish(
                        'orders',
                        order.model_dump(mode='json')
                    )

                    logger.info(
                        f"Order approved and published: {order.order_id}"
                    )
                else:
                    logger.warning(
                        f"Signal rejected: {', '.join(check_result.reasons)}"
                    )

            except Exception as e:
                logger.error(f"Signal processing error: {e}", exc_info=True)

    async def _monitor_positions(self):
        """
        Monitor open positions against market data.
        Generate close signals when stop loss or take profit is hit.
        """
        logger.info("Starting position monitoring")

        async for message in self.message_bus.subscribe('market_data'):
            if not self.running:
                break

            try:
                # Parse market tick
                tick = MarketTick(**message)

                # Check all positions against current price
                close_signals = await self.position_monitor.check_positions(tick)

                # Publish any close signals
                for signal in close_signals:
                    await self.message_bus.publish(
                        'signals',
                        signal.model_dump(mode='json')
                    )

                    logger.info(
                        f"Position close signal generated: {signal.rationale}"
                    )

            except Exception as e:
                logger.error(f"Position monitoring error: {e}", exc_info=True)

    async def _process_executions(self):
        """
        Process execution confirmations and update positions.
        Track P&L and check circuit breaker conditions.
        """
        logger.info("Starting execution processing")

        async for message in self.message_bus.subscribe('executions'):
            if not self.running:
                break

            try:
                # Parse execution
                execution = Execution(**message)

                logger.info(
                    f"Received execution: {execution.execution_id} - "
                    f"{execution.side.value} {execution.filled_quantity} "
                    f"{execution.instrument.value} @ {execution.fill_price}"
                )

                # Update or create position
                await self._update_position(execution)

                # If position was closed, check circuit breaker
                if execution.realized_pnl is not None:
                    self.circuit_breaker.check_and_update(
                        trade_pnl=execution.realized_pnl,
                        current_balance=self.account_balance
                    )

            except Exception as e:
                logger.error(f"Execution processing error: {e}", exc_info=True)

    def _signal_to_order(self, signal: TradeSignal) -> Order:
        """Convert approved trade signal to order"""
        import uuid
        from datetime import datetime
        from shared.models import OrderType, OrderStatus

        return Order(
            order_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            timestamp=datetime.utcnow(),
            instrument=signal.instrument,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=signal.quantity,
            limit_price=None,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.PENDING
        )

    async def _update_position(self, execution: Execution):
        """Update or create position based on execution"""
        from datetime import datetime

        # Check if we have an existing position for this instrument
        existing_position = None
        for pos in self.position_monitor.positions.values():
            if pos.instrument == execution.instrument:
                existing_position = pos
                break

        if existing_position:
            # Check if this closes or reduces the position
            if existing_position.side != execution.side:
                # Closing or reducing position
                if execution.filled_quantity >= existing_position.quantity:
                    # Full close
                    self.position_monitor.remove_position(existing_position.position_id)
                    logger.info(f"Position closed: {existing_position.position_id}")
                else:
                    # Partial close
                    existing_position.quantity -= execution.filled_quantity
                    logger.info(
                        f"Position reduced: {existing_position.position_id} - "
                        f"New quantity: {existing_position.quantity}"
                    )
            else:
                # Adding to position
                existing_position.quantity += execution.filled_quantity
                # Update average entry price
                total_value = (
                    existing_position.entry_price * existing_position.quantity +
                    execution.fill_price * execution.filled_quantity
                )
                total_quantity = existing_position.quantity + execution.filled_quantity
                existing_position.entry_price = total_value / total_quantity
                existing_position.quantity = total_quantity
                logger.info(
                    f"Position increased: {existing_position.position_id} - "
                    f"New quantity: {existing_position.quantity}"
                )
        else:
            # Create new position
            import uuid
            new_position = Position(
                position_id=str(uuid.uuid4()),
                instrument=execution.instrument,
                side=execution.side,
                quantity=execution.filled_quantity,
                entry_price=execution.fill_price,
                current_price=execution.fill_price,
                stop_loss=execution.stop_loss,
                take_profit=execution.take_profit,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                opened_at=datetime.utcnow()
            )
            self.position_monitor.add_position(new_position)


async def main():
    """Main entry point for Risk Agent"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    agent = RiskAgent(config, message_bus)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()
        await message_bus.disconnect()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

