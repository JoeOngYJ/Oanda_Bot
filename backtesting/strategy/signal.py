"""Signal models for backtesting strategy output."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol


class SignalDirection(Enum):
    """Signal direction for order intent."""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Signal:
    """Backtest trading signal emitted by strategies."""

    timestamp: datetime
    instrument: InstrumentSymbol
    direction: SignalDirection
    strategy_name: str
    entry_price: Decimal
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    timeframe: Timeframe
    confidence: float
    quantity: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entry_price = Decimal(str(self.entry_price))
        if self.stop_loss is not None:
            self.stop_loss = Decimal(str(self.stop_loss))
        if self.take_profit is not None:
            self.take_profit = Decimal(str(self.take_profit))
        if self.quantity <= 0:
            raise ValueError(f"Signal quantity must be positive, got: {self.quantity}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Signal confidence must be between 0 and 1, got: {self.confidence}"
            )
