from decimal import Decimal
from datetime import datetime, timezone

from oanda_bot.agents.strategy.models.rsi_mean_reversion import RSIMeanReversion
from oanda_bot.utils.models import MarketTick, Instrument


def main():
    config = {
        "name": "Test_RSI",
        "version": "1.0.0",
        "instruments": ["GBP_USD"],
        "parameters": {
        "rsi_period": 6,
        "oversold_threshold": 55,
        "overbought_threshold": 65,
            "position_size": 1000,
            "stop_loss_pips": 25,
            "take_profit_pips": 50,
        },
    }

    strategy = RSIMeanReversion(config)

    # Build an uptrend then a sharp drop to force RSI cross below oversold
    prices = [
        Decimal("1.27000"),
        Decimal("1.27050"),
        Decimal("1.27100"),
        Decimal("1.27150"),
        Decimal("1.27200"),
        Decimal("1.27250"),
        Decimal("1.27300"),
        Decimal("1.27350"),
        Decimal("1.27400"),
        Decimal("1.26000"),
        Decimal("1.25500"),
        Decimal("1.25000"),
    ]

    signal = None
    for price in prices:
        tick = MarketTick(
            instrument=Instrument.GBP_USD,
            timestamp=datetime.now(timezone.utc),
            bid=price,
            ask=price + Decimal("0.00005"),
            spread=Decimal("0.00005"),
            source="test",
            data_version="1.0.0",
        )
        strategy.update(tick)
        signal = strategy.check_signal(tick) or signal
        if signal is None:
            # Print debug RSI when available
            prices_list = list(strategy.price_history[Instrument.GBP_USD])
            if len(prices_list) >= config["parameters"]["rsi_period"] + 1:
                from oanda_bot.agents.strategy.indicators import rsi
                current_rsi = rsi(prices_list, config["parameters"]["rsi_period"])
                if current_rsi is not None:
                    print(f"RSI debug: {current_rsi:.2f} at price {price}")

    assert signal is not None, "No RSI signal generated"
    print(f"Signal generated: {signal.side} at {signal.entry_price}")
    print(f"Confidence: {signal.confidence}")
    print(f"Rationale: {signal.rationale}")


if __name__ == "__main__":
    main()
