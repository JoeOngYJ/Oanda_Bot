from decimal import Decimal
from datetime import datetime, timezone

from oanda_bot.agents.strategy.models.moving_average_crossover import MovingAverageCrossover
from oanda_bot.utils.models import MarketTick, Instrument


def main():
    config = {
        "name": "Test_MA",
        "version": "1.0.0",
        "instruments": ["EUR_USD"],
        "parameters": {
            "fast_period": 3,
            "slow_period": 5,
            "signal_threshold": 0.0,
            "position_size": 1000,
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
        },
    }

    strategy = MovingAverageCrossover(config)

    # Craft a clear bullish crossover:
    # prev_fast == prev_slow, then fast > slow on last tick
    prices = [
        Decimal("1.00000"),
        Decimal("1.00000"),
        Decimal("1.00000"),
        Decimal("1.00000"),
        Decimal("1.00000"),
        Decimal("2.00000"),
    ]

    # Feed all but last tick to build history
    for price in prices[:-1]:
        tick = MarketTick(
            instrument=Instrument.EUR_USD,
            timestamp=datetime.now(timezone.utc),
            bid=price,
            ask=price + Decimal("0.00005"),
            spread=Decimal("0.00005"),
            source="test",
            data_version="1.0.0",
        )
        strategy.update(tick)

    # Final tick should trigger the crossover
    last_price = prices[-1]
    last_tick = MarketTick(
        instrument=Instrument.EUR_USD,
        timestamp=datetime.now(timezone.utc),
        bid=last_price,
        ask=last_price + Decimal("0.00005"),
        spread=Decimal("0.00005"),
        source="test",
        data_version="1.0.0",
    )
    strategy.update(last_tick)
    signal = strategy.check_signal(last_tick)

    assert signal is not None, "No MA crossover signal generated"
    print(f"Signal generated: {signal.side} at {signal.entry_price}")
    print(f"Confidence: {signal.confidence}")
    print(f"Rationale: {signal.rationale}")


if __name__ == "__main__":
    main()
