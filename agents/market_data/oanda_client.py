"""
Manages streaming connection to Oanda pricing API.
Handles reconnections and heartbeats.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream
from shared.config import Config
from shared.models import Instrument

logger = logging.getLogger(__name__)


class OandaStreamClient:
    """Manages streaming connection to Oanda pricing API"""

    def __init__(self, config: Config):
        self.config = config
        self.account_id = config.oanda.account_id

        # Select environment
        env = config.oanda.environment
        endpoints = config.oanda.endpoints.get(env)
        if not endpoints:
            raise ValueError(f"Unknown Oanda environment: {env}")

        self.api_url = endpoints.api
        self.stream_url = endpoints.stream

        self.api = API(
            access_token=config.oanda.api_token,
            environment=env,
            headers={"Accept-Datetime-Format": "UNIX"}
        )

        self.active_streams = {}
        logger.info(f"Initialized Oanda client for {env} environment")

    async def stream_prices(
        self,
        instrument: Instrument
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream live prices for an instrument.
        Yields raw price ticks from Oanda.
        """
        params = {"instruments": instrument.value}

        retry_count = 0
        max_retries = self.config.oanda.max_retries

        while retry_count <= max_retries:
            try:
                logger.info(f"Starting price stream for {instrument.value}")
                request = PricingStream(accountID=self.account_id, params=params)

                # Execute streaming request
                for tick in self.api.request(request):
                    if tick["type"] == "PRICE":
                        yield {
                            "instrument": tick["instrument"],
                            "time": tick["time"],
                            "bids": tick["bids"],
                            "asks": tick["asks"],
                            "closeoutBid": tick.get("closeoutBid"),
                            "closeoutAsk": tick.get("closeoutAsk")
                        }
                    elif tick["type"] == "HEARTBEAT":
                        # Log heartbeat to confirm connection alive
                        logger.debug(f"Heartbeat received for {instrument.value}")

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Stream error for {instrument.value} (attempt {retry_count}/{max_retries}): {e}",
                    exc_info=True
                )

                if retry_count <= max_retries:
                    # Wait before retrying
                    await asyncio.sleep(self.config.oanda.retry_delay)
                    logger.info(f"Retrying stream for {instrument.value}...")
                else:
                    logger.error(f"Max retries exceeded for {instrument.value}")
                    raise

    async def close(self):
        """Close all streaming connections"""
        logger.info("Closing Oanda streaming connections")
        # Cleanup connections if needed
