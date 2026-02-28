"""Broker fee models for backtesting."""

from __future__ import annotations

from decimal import Decimal


class CommissionModel:
    """Base-like commission model used by the execution simulator."""

    def calculate(self, fill_price: Decimal, quantity: int) -> Decimal:
        raise NotImplementedError


class OandaSpreadOnlyCommissionModel(CommissionModel):
    """OANDA spread-only pricing: explicit trade commission is zero."""

    def calculate(self, fill_price: Decimal, quantity: int) -> Decimal:
        return Decimal("0")


class OandaCoreCommissionModel(CommissionModel):
    """OANDA core pricing style commission model.

    Default is 1.00 (account currency) per 10k units, per side.
    """

    def __init__(self, per_10k_units: Decimal = Decimal("1.00")) -> None:
        self.per_10k_units = Decimal(str(per_10k_units))

    def calculate(self, fill_price: Decimal, quantity: int) -> Decimal:
        units = Decimal(abs(quantity))
        return (units / Decimal("10000")) * self.per_10k_units


class FixedPerTradeCommissionModel(CommissionModel):
    """Flat commission charged for each executed fill."""

    def __init__(self, per_trade: Decimal = Decimal("0")) -> None:
        self.per_trade = Decimal(str(per_trade))

    def calculate(self, fill_price: Decimal, quantity: int) -> Decimal:
        return self.per_trade


def simple_commission(volume, price, rate=0.0001):
    """Legacy compatibility helper."""
    return abs(volume * price) * rate
