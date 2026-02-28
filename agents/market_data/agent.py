"""
Market Data Agent - Ingests, normalizes, validates, and distributes market data.
All processing is deterministic and fully logged.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict
from shared.message_bus import MessageBus
from shared.models import MarketTick, Instrument
from shared.config import Config
from .oanda_client import OandaStreamClient
from .data_normalizer import DataNormalizer
from .data_validator import DataValidator
from .storage import MarketDataStorage

logger = logging.getLogger(__name__)


class MarketDataAgent:
    """
    Ingests market data from Oanda, normalizes, validates, and distributes.
    All processing is deterministic and fully logged.
    """

    def __init__(self, config: Config, message_bus: MessageBus):
        self.config = config
        self.message_bus = message_bus

        self.oanda_client = OandaStreamClient(config)
        self.normalizer = DataNormalizer()
        self.validator = DataValidator(config)
        self.storage = MarketDataStorage(config)

        self.running = False
        self.instruments = [Instrument(i) for i in config.oanda.instruments]

    async def start(self):
        """Start market data agent"""
        logger.info(f"Starting Market Data Agent for instruments: {self.instruments}")
        self.running = True

        # Start streaming for each instrument
        tasks = [
            asyncio.create_task(self._stream_instrument(instrument))
            for instrument in self.instruments
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Market data agent error: {e}", exc_info=True)
            await self.stop()

    async def _stream_instrument(self, instrument: Instrument):
        """Stream market data for a single instrument"""
        while self.running:
            try:
                async for raw_tick in self.oanda_client.stream_prices(instrument):
                    # 1. Normalize: convert to standard format
                    normalized_tick = self.normalizer.normalize(raw_tick, instrument)

                    # 2. Validate: check for anomalies
                    is_valid, issues = self.validator.validate(normalized_tick)

                    if not is_valid:
                        logger.warning(
                            f"Invalid tick for {instrument}: {issues}",
                            extra={"tick": normalized_tick.model_dump(mode='json')}
                        )
                        # Publish alert
                        await self.message_bus.publish(
                            'alerts',
                            {
                                'component': 'market_data',
                                'severity': 'warning',
                                'message': f'Invalid tick: {issues}',
                                'instrument': instrument.value
                            }
                        )
                        continue

                    # 3. Store: persist to InfluxDB
                    await self.storage.save_tick(normalized_tick)

                    # 4. Distribute: publish to message bus
                    await self.message_bus.publish(
                        'market_data',
                        normalized_tick.model_dump(mode='json')
                    )

                    logger.debug(
                        f"Processed tick: {instrument} @ {normalized_tick.timestamp}",
                        extra={
                            "instrument": instrument.value,
                            "bid": float(normalized_tick.bid),
                            "ask": float(normalized_tick.ask),
                            "spread": float(normalized_tick.spread)
                        }
                    )

            except asyncio.CancelledError:
                logger.info(f"Streaming cancelled for {instrument}")
                break
            except Exception as e:
                logger.error(
                    f"Error streaming {instrument}: {e}",
                    exc_info=True
                )
                # Wait before reconnecting
                await asyncio.sleep(self.config.oanda.retry_delay)

    async def get_historical_data(
        self,
        instrument: Instrument,
        start_time: datetime,
        end_time: datetime
    ) -> list[MarketTick]:
        """Retrieve historical market data from storage"""
        return await self.storage.query_ticks(instrument, start_time, end_time)

    async def stop(self):
        """Gracefully stop market data agent"""
        logger.info("Stopping Market Data Agent")
        self.running = False
        await self.oanda_client.close()
        self.storage.close()


async def main():
    """Main entry point for Market Data Agent"""
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    agent = MarketDataAgent(config, message_bus)

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
