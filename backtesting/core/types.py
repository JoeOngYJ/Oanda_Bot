# backtesting/core/types.py

from typing import NewType
from decimal import Decimal
from datetime import datetime
from enum import Enum


# Type aliases for domain-specific types
InstrumentSymbol = NewType("InstrumentSymbol", str)
Price = Decimal
Volume = int
Timestamp = datetime


class OrderSide(Enum):
    """Order side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# Export all public types
__all__ = [
    "InstrumentSymbol",
    "Price",
    "Volume",
    "Timestamp",
    "OrderSide",
    "OrderStatus",
]
