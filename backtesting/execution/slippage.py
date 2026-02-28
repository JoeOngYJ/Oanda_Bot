"""Execution price modeling for spread + slippage in backtests."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict

from backtesting.strategy.signal import SignalDirection

# Typical spread values (pips) from OANDA public pricing pages (major FX pairs).
# Source references are documented in final implementation summary.
OANDA_TYPICAL_SPREAD_PIPS: Dict[str, Decimal] = {
    "AUD_USD": Decimal("1.4"),
    "EUR_GBP": Decimal("1.7"),
    "EUR_JPY": Decimal("1.8"),
    "EUR_USD": Decimal("1.4"),
    "GBP_JPY": Decimal("3.1"),
    "GBP_USD": Decimal("2.0"),
    "NZD_USD": Decimal("1.7"),
    "USD_CAD": Decimal("2.2"),
    "USD_CHF": Decimal("1.8"),
    "USD_JPY": Decimal("1.4"),
}


def pip_size_for_instrument(instrument: str) -> Decimal:
    """Return pip size for an FX instrument."""
    return Decimal("0.01") if instrument.endswith("_JPY") else Decimal("0.0001")


class SlippageModel:
    """Models fill price from mid-price with spread and slippage."""

    def __init__(
        self,
        spread_pips_by_instrument: Dict[str, Decimal] | None = None,
        default_spread_pips: Decimal = Decimal("1.5"),
        slippage_pips: Decimal = Decimal("0"),
    ) -> None:
        self.spread_pips_by_instrument = {
            k: Decimal(str(v))
            for k, v in (spread_pips_by_instrument or OANDA_TYPICAL_SPREAD_PIPS).items()
        }
        self.default_spread_pips = Decimal(str(default_spread_pips))
        self.slippage_pips = Decimal(str(slippage_pips))

    def get_spread_pips(self, instrument: str) -> Decimal:
        return self.spread_pips_by_instrument.get(instrument, self.default_spread_pips)

    def apply(self, mid_price: Decimal, direction: SignalDirection, instrument: str) -> Decimal:
        """Convert mid-price to executable bid/ask price and apply slippage."""
        mid = Decimal(str(mid_price))
        pip = pip_size_for_instrument(instrument)
        half_spread = (self.get_spread_pips(instrument) * pip) / Decimal("2")
        slippage = self.slippage_pips * pip

        if direction == SignalDirection.LONG:
            # Buy fills at ask and adverse slippage increases entry price.
            return mid + half_spread + slippage
        # Sell fills at bid and adverse slippage decreases entry price.
        return mid - half_spread - slippage


def fixed_slippage(price, ticks=1):
    """Legacy compatibility helper."""
    return price
