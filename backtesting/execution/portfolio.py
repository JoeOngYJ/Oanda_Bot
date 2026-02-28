"""Portfolio & position tracking utilities for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict


@dataclass
class Position:
    instrument: str
    quantity: int
    average_entry: Decimal


class Portfolio:
    def __init__(self, cash=100000):
        self.cash = Decimal(str(cash))
        self.positions: Dict[str, Position] = {}
        self.total_fees_paid = Decimal("0")
        self.total_financing = Decimal("0")

    def open_long(self, instrument: str, price: Decimal, quantity: int, commission: Decimal = Decimal("0")):
        self._open(instrument, Decimal(str(price)), abs(quantity), Decimal(str(commission)))

    def open_short(self, instrument: str, price: Decimal, quantity: int, commission: Decimal = Decimal("0")):
        self._open(instrument, Decimal(str(price)), -abs(quantity), Decimal(str(commission)))

    def _open(self, instrument: str, price: Decimal, signed_qty: int, commission: Decimal):
        existing = self.positions.get(instrument)
        if existing is None:
            self.positions[instrument] = Position(
                instrument=instrument,
                quantity=signed_qty,
                average_entry=price,
            )
        else:
            new_qty = existing.quantity + signed_qty
            if new_qty == 0:
                del self.positions[instrument]
            else:
                weighted = (
                    existing.average_entry * Decimal(abs(existing.quantity))
                    + price * Decimal(abs(signed_qty))
                )
                self.positions[instrument] = Position(
                    instrument=instrument,
                    quantity=new_qty,
                    average_entry=weighted / Decimal(abs(new_qty)),
                )

        self.cash -= commission
        self.total_fees_paid += commission

    def apply_financing(self, amount: Decimal) -> None:
        """Apply overnight financing (positive = cost, negative = rebate)."""
        val = Decimal(str(amount))
        self.cash -= val
        self.total_financing += val

    def update_on_fill(self, fill):
        pass

    def gross_notional_exposure(self) -> Decimal:
        """Return gross notional exposure across open positions."""
        total = Decimal("0")
        for pos in self.positions.values():
            total += Decimal(abs(pos.quantity)) * pos.average_entry
        return total
