"""Dynamic backtesting system state and persistence utilities."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from backtesting.data.models import OHLCVBar


@dataclass
class PositionState:
    instrument: str
    quantity: int
    average_entry: Decimal


@dataclass
class SystemState:
    """Central dynamic state for real-time style simulation."""

    initial_cash: Decimal
    available_cash: Decimal
    total_equity: Decimal
    last_detected_regime: Optional[str] = None
    positions: Dict[str, PositionState] = field(default_factory=dict)
    historical_pnl: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    market_data_buffer: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    feature_buffer: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    last_prices: Dict[str, Decimal] = field(default_factory=dict)
    bar_count: int = 0
    realized_pnl: Decimal = Decimal("0")
    _open_lots: Dict[str, Deque[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(deque))

    @classmethod
    def create(cls, initial_cash: float | Decimal) -> "SystemState":
        cash = Decimal(str(initial_cash))
        return cls(
            initial_cash=cash,
            available_cash=cash,
            total_equity=cash,
            equity_curve=[float(cash)],
        )

    def ingest_bar(self, bar: OHLCVBar) -> None:
        self.last_prices[str(bar.instrument)] = Decimal(str(bar.close))
        row = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "instrument": str(bar.instrument),
                    "timeframe": bar.timeframe.name,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
            ]
        )
        self.market_data_buffer = pd.concat([self.market_data_buffer, row], ignore_index=True)

    def append_features(self, bar: OHLCVBar, features: Dict[str, Any]) -> None:
        base = {
            "timestamp": bar.timestamp,
            "instrument": str(bar.instrument),
            "timeframe": bar.timeframe.name,
        }
        for k, v in (features or {}).items():
            if isinstance(v, Decimal):
                base[k] = float(v)
            else:
                base[k] = v
        row = pd.DataFrame([base])
        self.feature_buffer = pd.concat([self.feature_buffer, row], ignore_index=True)

    def sync_from_portfolio(self, portfolio) -> None:
        self.available_cash = Decimal(str(portfolio.cash))
        self.positions = {
            instrument: PositionState(
                instrument=instrument,
                quantity=pos.quantity,
                average_entry=Decimal(str(pos.average_entry)),
            )
            for instrument, pos in portfolio.positions.items()
        }
        self.total_equity = self._mark_to_market_equity()
        self.historical_pnl.append(float(self.total_equity - self.initial_cash))
        self.equity_curve.append(float(self.total_equity))

    def append_fills(self, fills: List[Dict[str, Any]]) -> None:
        for fill in fills:
            direction_val = getattr(fill.get("direction"), "value", fill.get("direction"))
            side = str(direction_val)
            instrument = str(fill.get("instrument"))
            qty_remaining = int(fill.get("quantity", 0))
            fill_price = Decimal(str(fill.get("fill_price", 0)))
            commission = Decimal(str(fill.get("commission", 0)))
            exit_comm_per_unit = commission / Decimal(max(qty_remaining, 1))

            lots = self._open_lots[instrument]
            while qty_remaining > 0 and lots:
                lot = lots[0]
                if lot["direction"] == side:
                    break

                match_qty = min(qty_remaining, int(lot["quantity"]))
                entry_price = Decimal(str(lot["entry_price"]))
                entry_comm_per_unit = Decimal(str(lot["commission_per_unit"]))

                if lot["direction"] == "LONG" and side == "SHORT":
                    gross = (fill_price - entry_price) * Decimal(match_qty)
                else:
                    gross = (entry_price - fill_price) * Decimal(match_qty)

                trade_commission = (entry_comm_per_unit + exit_comm_per_unit) * Decimal(match_qty)
                self.realized_pnl += gross - trade_commission

                lot["quantity"] -= match_qty
                qty_remaining -= match_qty
                if lot["quantity"] == 0:
                    lots.popleft()

            if qty_remaining > 0:
                lots.append(
                    {
                        "direction": side,
                        "quantity": qty_remaining,
                        "entry_price": fill_price,
                        "commission_per_unit": exit_comm_per_unit,
                    }
                )

            rec = {
                "timestamp": str(fill.get("timestamp")),
                "instrument": instrument,
                "direction": side,
                "quantity": int(fill.get("quantity", 0)),
                "fill_price": float(fill.get("fill_price", 0)),
                "commission": float(fill.get("commission", 0)),
                "fill_reason": str(fill.get("fill_reason", "")),
            }
            self.trade_history.append(rec)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "bar_count": int(self.bar_count),
            "available_cash": float(self.available_cash),
            "total_equity": float(self.total_equity),
            "realized_pnl": float(self.realized_pnl),
            "last_detected_regime": self.last_detected_regime,
            "positions": {
                k: {
                    "quantity": int(v.quantity),
                    "average_entry": float(v.average_entry),
                }
                for k, v in self.positions.items()
            },
            "historical_pnl": self.historical_pnl[-1000:],
            "equity_curve_tail": self.equity_curve[-1000:],
            "trade_count": len(self.trade_history),
            "market_rows": int(len(self.market_data_buffer)),
            "feature_rows": int(len(self.feature_buffer)),
        }

    def persist_snapshot(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.snapshot()) + "\n")

    def _mark_to_market_equity(self) -> Decimal:
        unrealized = Decimal("0")
        reserved_entry_commission = Decimal("0")
        for inst, lots in self._open_lots.items():
            px = self.last_prices.get(inst, Decimal("0"))
            for lot in lots:
                qty = int(lot["quantity"])
                entry = Decimal(str(lot["entry_price"]))
                if lot["direction"] == "LONG":
                    unrealized += (px - entry) * Decimal(qty)
                else:
                    unrealized += (entry - px) * Decimal(qty)
                reserved_entry_commission += Decimal(str(lot["commission_per_unit"])) * Decimal(qty)
        total = self.initial_cash + self.realized_pnl + unrealized - reserved_entry_commission
        return total
