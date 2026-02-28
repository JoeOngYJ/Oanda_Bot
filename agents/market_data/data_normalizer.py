"""
Normalizes raw Oanda data into standard MarketTick format.
All transformations are deterministic and logged.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict
from shared.models import MarketTick, Instrument


class DataNormalizer:
    """
    Normalizes raw Oanda data into standard MarketTick format.
    All transformations are deterministic and logged.
    """

    def normalize(self, raw_tick: Dict, instrument: Instrument) -> MarketTick:
        """
        Convert Oanda raw tick to normalized MarketTick.

        Oanda provides:
        - time: Unix timestamp string
        - bids: list of {"price": str, "liquidity": int}
        - asks: list of {"price": str, "liquidity": int}
        """

        # Parse timestamp (Oanda uses RFC3339 or Unix)
        timestamp = self._parse_timestamp(raw_tick["time"])

        # Extract best bid/ask (first in list)
        best_bid = Decimal(raw_tick["bids"][0]["price"])
        best_ask = Decimal(raw_tick["asks"][0]["price"])

        # Optional: extract liquidity
        bid_volume = raw_tick["bids"][0].get("liquidity")
        ask_volume = raw_tick["asks"][0].get("liquidity")

        # Calculate spread
        spread = best_ask - best_bid

        return MarketTick(
            instrument=instrument,
            timestamp=timestamp,
            bid=best_bid,
            ask=best_ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            spread=spread,
            source="oanda",
            data_version="1.0.0"
        )

    def _parse_timestamp(self, time_str: str) -> datetime:
        """Parse Oanda timestamp to UTC datetime"""
        # Oanda returns Unix timestamp as float
        return datetime.utcfromtimestamp(float(time_str))
