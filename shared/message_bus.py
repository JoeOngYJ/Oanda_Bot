"""
Redis Streams-based message bus for inter-agent communication.
Provides async publish/subscribe with consumer groups.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional
from datetime import datetime
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Redis Streams wrapper for pub/sub messaging between agents.
    Supports named streams, consumer groups, and reliable delivery.
    """

    def __init__(self, config):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.streams = config.redis.streams
        self.consumer_groups = {}

    async def connect(self):
        """Establish connection to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )

            # Test connection
            await self.redis_client.ping()
            logger.info(
                f"Connected to Redis at {self.config.redis.host}:{self.config.redis.port}"
            )

            # Create consumer groups for all streams
            await self._create_consumer_groups()

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise

    async def _create_consumer_groups(self):
        """Create consumer groups for all streams"""
        for stream_name, stream_key in self.streams.items():
            try:
                # Try to create consumer group
                await self.redis_client.xgroup_create(
                    name=stream_key,
                    groupname=f"{stream_name}_group",
                    id="0",
                    mkstream=True
                )
                logger.info(f"Created consumer group for stream: {stream_key}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists
                    logger.debug(f"Consumer group already exists for {stream_key}")
                else:
                    logger.error(f"Error creating consumer group for {stream_key}: {e}")

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")

    async def publish(self, stream_name: str, message: dict):
        """
        Publish a message to a stream.

        Args:
            stream_name: Name of the stream (e.g., 'market_data', 'signals')
            message: Dictionary to publish (will be JSON serialized)
        """
        if not self.redis_client:
            raise RuntimeError("Message bus not connected")

        try:
            # Get stream key from config
            stream_key = self.streams.get(stream_name, stream_name)

            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.utcnow().isoformat()

            # Serialize message
            serialized = self._serialize_message(message)

            # Publish to stream
            message_id = await self.redis_client.xadd(
                name=stream_key,
                fields=serialized
            )

            logger.debug(
                f"Published to {stream_key}: {message_id}",
                extra={"stream": stream_key, "message_id": message_id}
            )

            return message_id

        except Exception as e:
            logger.error(
                f"Failed to publish to {stream_name}: {e}",
                exc_info=True
            )
            raise

    async def subscribe(
        self,
        stream_name: str,
        consumer_name: Optional[str] = None,
        block_ms: int = 1000
    ) -> AsyncGenerator[dict, None]:
        """
        Subscribe to a stream and yield messages.

        Args:
            stream_name: Name of the stream to subscribe to
            consumer_name: Optional consumer name for consumer group
            block_ms: Milliseconds to block waiting for messages

        Yields:
            Deserialized message dictionaries
        """
        if not self.redis_client:
            raise RuntimeError("Message bus not connected")

        stream_key = self.streams.get(stream_name, stream_name)
        group_name = f"{stream_name}_group"
        consumer_id = consumer_name or f"consumer_{id(self)}"

        logger.info(
            f"Subscribing to {stream_key} as {consumer_id} in group {group_name}"
        )

        # Ensure the consumer group exists for this stream
        try:
            await self.redis_client.xgroup_create(
                name=stream_key,
                groupname=group_name,
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group {group_name} for {stream_key}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group already exists for {stream_key}")
            else:
                logger.error(f"Error creating consumer group for {stream_key}: {e}")

        # Start reading from the latest undelivered message
        last_id = ">"

        try:
            while True:
                try:
                    # Read from consumer group
                    messages = await self.redis_client.xreadgroup(
                        groupname=group_name,
                        consumername=consumer_id,
                        streams={stream_key: last_id},
                        count=10,
                        block=block_ms
                    )

                    if not messages:
                        # No messages, continue waiting
                        await asyncio.sleep(0.01)
                        continue

                    # Process messages
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            try:
                                # Deserialize message
                                message = self._deserialize_message(fields)

                                # Yield message to consumer
                                yield message

                                # Acknowledge message
                                await self.redis_client.xack(
                                    stream_key,
                                    group_name,
                                    message_id
                                )

                            except Exception as e:
                                logger.error(
                                    f"Error processing message {message_id}: {e}",
                                    exc_info=True
                                )

                except asyncio.CancelledError:
                    logger.info(f"Subscription cancelled for {stream_key}")
                    break
                except Exception as e:
                    logger.error(
                        f"Error reading from {stream_key}: {e}",
                        exc_info=True
                    )
                    await asyncio.sleep(1)

        finally:
            logger.info(f"Unsubscribed from {stream_key}")

    async def get_stream_stats(self) -> dict:
        """
        Get statistics for all streams.

        Returns:
            Dictionary with stream statistics
        """
        if not self.redis_client:
            raise RuntimeError("Message bus not connected")

        stats = {}

        for stream_name, stream_key in self.streams.items():
            try:
                # Get stream info
                info = await self.redis_client.xinfo_stream(stream_key)

                # Get consumer group info
                groups = await self.redis_client.xinfo_groups(stream_key)

                stats[stream_name] = {
                    'length': info.get('length', 0),
                    'first_entry': info.get('first-entry'),
                    'last_entry': info.get('last-entry'),
                    'groups': len(groups),
                    'lag': groups[0].get('lag', 0) if groups else 0
                }

            except redis.ResponseError as e:
                logger.warning(f"Could not get stats for {stream_key}: {e}")
                stats[stream_name] = {'error': str(e)}

        return stats

    def _serialize_message(self, message: dict) -> dict:
        """Serialize message for Redis (convert to strings)"""
        serialized = {}
        for key, value in message.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = str(value)
        return serialized

    def _deserialize_message(self, fields: dict) -> dict:
        """Deserialize message from Redis"""
        deserialized = {}
        for key, value in fields.items():
            try:
                # Try to parse as JSON
                deserialized[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Keep as string
                deserialized[key] = value
        return deserialized
