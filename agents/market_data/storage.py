"""
InfluxDB persistence layer for market data.
Handles tick storage and historical queries.
"""

import logging
from datetime import datetime
from typing import List, Optional
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from shared.models import MarketTick, Instrument
from shared.config import Config

logger = logging.getLogger(__name__)


class MarketDataStorage:
    """
    Handles persistence of market data to InfluxDB.
    Provides query interface for historical data.
    """

    def __init__(self, config: Config):
        self.config = config
        self.bucket = config.influxdb.bucket
        self.org = config.influxdb.org

        # Initialize InfluxDB client
        self.client = InfluxDBClient(
            url=config.influxdb.url,
            token=config.influxdb.token,
            org=self.org
        )

        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

        logger.info(f"Connected to InfluxDB at {config.influxdb.url}")

    async def save_tick(self, tick: MarketTick) -> None:
        """
        Persist tick to InfluxDB.

        Schema:
        - Measurement: forex_prices
        - Tags: instrument, source
        - Fields: bid, ask, spread, bid_volume, ask_volume
        - Timestamp: tick.timestamp
        """
        try:
            point = Point("forex_prices") \
                .tag("instrument", tick.instrument.value) \
                .tag("source", tick.source) \
                .field("bid", float(tick.bid)) \
                .field("ask", float(tick.ask)) \
                .field("spread", float(tick.spread)) \
                .time(tick.timestamp)

            # Add volume fields if available
            if tick.bid_volume is not None:
                point.field("bid_volume", tick.bid_volume)
            if tick.ask_volume is not None:
                point.field("ask_volume", tick.ask_volume)

            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)

            logger.debug(
                f"Saved tick to InfluxDB: {tick.instrument.value} @ {tick.timestamp}"
            )

        except Exception as e:
            logger.error(
                f"Failed to save tick to InfluxDB: {e}",
                exc_info=True,
                extra={"tick": tick.model_dump(mode='json')}
            )
            # Don't raise - storage failures shouldn't block publishing

    async def query_ticks(
        self,
        instrument: Instrument,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketTick]:
        """
        Query historical ticks from InfluxDB.

        Args:
            instrument: Instrument to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of MarketTick objects
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "forex_prices")
                |> filter(fn: (r) => r.instrument == "{instrument.value}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            result = self.query_api.query(org=self.org, query=query)

            ticks = []
            for table in result:
                for record in table.records:
                    tick = MarketTick(
                        instrument=instrument,
                        timestamp=record.get_time(),
                        bid=record.values.get("bid"),
                        ask=record.values.get("ask"),
                        spread=record.values.get("spread"),
                        bid_volume=record.values.get("bid_volume"),
                        ask_volume=record.values.get("ask_volume"),
                        source=record.values.get("source", "oanda"),
                        data_version="1.0.0"
                    )
                    ticks.append(tick)

            logger.info(
                f"Retrieved {len(ticks)} ticks for {instrument.value} "
                f"from {start_time} to {end_time}"
            )

            return ticks

        except Exception as e:
            logger.error(f"Failed to query ticks from InfluxDB: {e}", exc_info=True)
            return []

    async def get_latest_tick(self, instrument: Instrument) -> Optional[MarketTick]:
        """
        Get most recent tick for an instrument.

        Args:
            instrument: Instrument to query

        Returns:
            Latest MarketTick or None if not found
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "forex_prices")
                |> filter(fn: (r) => r.instrument == "{instrument.value}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> last()
            '''

            result = self.query_api.query(org=self.org, query=query)

            for table in result:
                for record in table.records:
                    return MarketTick(
                        instrument=instrument,
                        timestamp=record.get_time(),
                        bid=record.values.get("bid"),
                        ask=record.values.get("ask"),
                        spread=record.values.get("spread"),
                        bid_volume=record.values.get("bid_volume"),
                        ask_volume=record.values.get("ask_volume"),
                        source=record.values.get("source", "oanda"),
                        data_version="1.0.0"
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get latest tick: {e}", exc_info=True)
            return None

    def close(self):
        """Close InfluxDB connection"""
        if self.client:
            self.client.close()
            logger.info("Closed InfluxDB connection")
