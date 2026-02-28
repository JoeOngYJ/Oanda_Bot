import asyncio

from shared.config import Config
from shared.message_bus import MessageBus
from shared.models import TradeSignal


async def main(timeout_s: int = 60):
    bus = MessageBus(Config.load())
    await bus.connect()
    print("Waiting for signal...")

    async def read_one():
        async for msg in bus.subscribe("stream:signals"):
            signal = TradeSignal(**msg)
            print(f"Signal ID: {signal.signal_id}")
            print(f"Instrument: {signal.instrument}")
            print(f"Side: {signal.side}")
            print(f"Quantity: {signal.quantity}")
            print(f"Confidence: {signal.confidence}")
            print(f"Strategy: {signal.strategy_name} v{signal.strategy_version}")
            print(f"Rationale: {signal.rationale}")
            print(f"Entry: {signal.entry_price}")
            print(f"Stop Loss: {signal.stop_loss}")
            print(f"Take Profit: {signal.take_profit}")

            assert signal.signal_id
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            print("Signal validation passed")
            return

    try:
        await asyncio.wait_for(read_one(), timeout=timeout_s)
    except asyncio.TimeoutError:
        print("Timed out waiting for signal.")
    finally:
        await bus.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
