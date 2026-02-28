"""
Strategy Agent - Analyzes market data and generates trade signals.
All strategy logic is deterministic, versioned, and backtested.
"""

import asyncio
import logging
from typing import Dict
from shared.message_bus import MessageBus
from shared.models import MarketTick, Instrument
from shared.config import Config
from .signal_generator import SignalGenerator
from .models.moving_average_crossover import MovingAverageCrossover
from .models.rsi_mean_reversion import RSIMeanReversion

logger = logging.getLogger(__name__)


class StrategyAgent:
    """
    Analyzes market data and generates trade signals.
    All strategy logic is deterministic, versioned, and backtested.
    """

    def __init__(self, config: Config, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus

        # Load and initialize strategies
        self.strategies = self._load_strategies()
        self.signal_generator = SignalGenerator(self.strategies)

        self.running = False

    def _load_strategies(self) -> Dict:
        """Load enabled strategies from configuration"""
        strategies = {}

        for strategy_config in self.config.strategies:
            if not strategy_config.enabled:
                continue

            strategy_class_name = strategy_config.strategy_class

            # Convert strategy config to dict
            config_dict = {
                'name': strategy_config.name,
                'version': strategy_config.version,
                'enabled': strategy_config.enabled,
                'instruments': strategy_config.instruments,
                'parameters': strategy_config.parameters
            }

            # Instantiate strategy
            if strategy_class_name == "MovingAverageCrossover":
                strategy = MovingAverageCrossover(config_dict)
            elif strategy_class_name == "RSIMeanReversion":
                strategy = RSIMeanReversion(config_dict)
            else:
                logger.warning(f"Unknown strategy class: {strategy_class_name}")
                continue

            strategies[strategy_config.name] = strategy
            logger.info(
                f"Loaded strategy: {strategy_config.name} "
                f"v{strategy_config.version}"
            )

        return strategies

    async def start(self):
        """Start strategy agent"""
        logger.info(f"Starting Strategy Agent with {len(self.strategies)} strategies")
        self.running = True

        # Subscribe to market data stream
        async for message in self.message_bus.subscribe('market_data'):
            if not self.running:
                break

            try:
                # Parse market tick
                tick = MarketTick(**message)

                # Generate signals from all strategies
                signals = await self.signal_generator.generate_signals(tick)

                # Publish signals to risk agent
                for signal in signals:
                    await self.message_bus.publish(
                        'signals',
                        signal.model_dump(mode='json')
                    )

                    logger.info(
                        f"Generated signal: {signal.strategy_name} - "
                        f"{signal.side.value} {signal.quantity} {signal.instrument.value} "
                        f"(confidence: {signal.confidence:.2f})"
                    )

            except Exception as e:
                logger.error(f"Strategy processing error: {e}", exc_info=True)

    async def stop(self):
        """Gracefully stop strategy agent"""
        logger.info("Stopping Strategy Agent")
        self.running = False


async def main():
    """Main entry point for Strategy Agent"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    agent = StrategyAgent(config, message_bus)

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
