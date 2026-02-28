import asyncio
import sys

print("test.py starting...", flush=True)

from shared.config import Config
from shared.message_bus import MessageBus


async def main():
    try:
        print("Loading config...", flush=True)
        config = Config.load()

        print("Connecting to Redis...", flush=True)
        bus = MessageBus(config)
        await bus.connect()
        print("Connected.", flush=True)

        await bus.publish("test_stream", {"msg": "hello", "count": 1})
        print("Published message", flush=True)

        async def read_one():
            async for msg in bus.subscribe("test_stream"):
                print("Received:", msg, flush=True)
                return

        try:
            await asyncio.wait_for(read_one(), timeout=5)
        except asyncio.TimeoutError:
            print("Timed out waiting for message.", flush=True)

        await bus.disconnect()
        print("Message bus test passed!", flush=True)

    except Exception as exc:
        print(f"Test failed: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
