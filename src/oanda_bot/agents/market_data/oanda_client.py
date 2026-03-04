"""
Manages streaming connection to Oanda pricing API.
Handles reconnections and heartbeats.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Dict, List
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream
from oanda_bot.utils.config import Config
from oanda_bot.utils.models import Instrument

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

    def get_recent_candles(
        self,
        instrument: Instrument,
        granularity: str,
        count: int,
        price_component: str = "M",
    ) -> List[Dict]:
        """
        Fetch recent completed candles for warmup/bootstrap.

        Returns a list of dictionaries:
        [{time, open, high, low, close, volume}, ...]
        """
        if count <= 0:
            return []

        max_per_request = 5000
        remaining = int(count)
        cursor_to: datetime | None = None
        out: List[Dict] = []
        safety_loops = 0

        while remaining > 0:
            safety_loops += 1
            if safety_loops > 32:
                logger.warning(
                    "Aborting candle pagination after safety limit",
                    extra={"instrument": instrument.value, "granularity": granularity, "requested": count},
                )
                break

            req_count = min(remaining, max_per_request)
            params: Dict[str, str | int] = {
                "granularity": granularity,
                "price": price_component,
                "count": int(req_count),
            }
            if cursor_to is not None:
                params["to"] = cursor_to.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

            request = InstrumentsCandles(instrument=instrument.value, params=params)
            resp = self.api.request(request)
            candles = resp.get("candles", []) or []
            if not candles:
                break

            chunk: List[Dict] = []
            for c in candles:
                if not c.get("complete", False):
                    continue
                mid = c.get("mid") or {}
                ts = self._parse_oanda_time(c.get("time"))
                if ts is None:
                    continue
                chunk.append(
                    {
                        "time": ts,
                        "open": mid.get("o"),
                        "high": mid.get("h"),
                        "low": mid.get("l"),
                        "close": mid.get("c"),
                        "volume": int(c.get("volume", 0)),
                    }
                )

            if not chunk:
                break

            out.extend(chunk)
            remaining -= len(chunk)

            earliest = chunk[0]["time"]
            if not isinstance(earliest, datetime):
                break
            cursor_to = earliest - timedelta(microseconds=1)
            if len(chunk) < req_count:
                break

        out.sort(key=lambda x: x["time"])
        deduped: List[Dict] = []
        last_ts: datetime | None = None
        for row in out:
            ts = row["time"]
            if ts == last_ts:
                continue
            deduped.append(row)
            last_ts = ts
        if len(deduped) > count:
            deduped = deduped[-count:]
        return deduped

    @staticmethod
    def _parse_oanda_time(value) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                return None
