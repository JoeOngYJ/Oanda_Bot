# backtesting/execution/simulator.py

from collections import defaultdict, deque
from math import sqrt
from typing import Deque, Dict, List, Optional
from decimal import Decimal
from dataclasses import dataclass
from backtesting.strategy.signal import Signal, SignalDirection
from backtesting.data.models import OHLCVBar
from .portfolio import Portfolio
from .slippage import SlippageModel
from .commission import CommissionModel


@dataclass
class OpenPosition:
    instrument: str
    direction: SignalDirection
    quantity: int
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    entry_price: Decimal
    opened_timestamp: object


class ExecutionSimulator:
    """
    Simulates order execution during backtest.
    
    Features:
    - Realistic fill simulation (uses bar high/low)
    - Configurable slippage model
    - Commission calculation
    - Position tracking
    """
    
    def __init__(
        self,
        initial_capital: Decimal,
        slippage_model: SlippageModel,
        commission_model: CommissionModel,
        fill_mode: str = "touch",
        volatility_targeting_enabled: bool = False,
        target_annual_volatility: Decimal = Decimal("0.15"),
        volatility_lookback_bars: int = 96,
        max_concurrent_exposure_pct: Optional[Decimal] = None,
        min_quantity: int = 1,
        max_quantity: Optional[int] = None,
        base_timeframe_seconds: Optional[int] = None,
    ):
        self.portfolio = Portfolio(initial_capital)
        self.slippage_model = slippage_model
        self.commission_model = commission_model
        self.fill_mode = str(fill_mode).lower()
        if self.fill_mode not in {"touch", "next_open"}:
            raise ValueError(f"Unsupported fill_mode: {fill_mode}")
        self.initial_capital = Decimal(str(initial_capital))
        self.volatility_targeting_enabled = bool(volatility_targeting_enabled)
        self.target_annual_volatility = Decimal(str(target_annual_volatility))
        self.volatility_lookback_bars = max(int(volatility_lookback_bars), 10)
        self.max_concurrent_exposure_pct = (
            Decimal(str(max_concurrent_exposure_pct)) if max_concurrent_exposure_pct is not None else None
        )
        self.min_quantity = max(int(min_quantity), 1)
        self.max_quantity = int(max_quantity) if max_quantity is not None else None
        self.base_timeframe_seconds = int(base_timeframe_seconds) if base_timeframe_seconds else 3600
        self.close_history: Dict[str, Deque[Decimal]] = defaultdict(
            lambda: deque(maxlen=self.volatility_lookback_bars + 2)
        )
        
        self.pending_orders: List[Signal] = []
        self.filled_orders: List[Dict] = []
        self.open_positions: List[OpenPosition] = []
    
    def process_signal(self, signal: Signal):
        """Add signal to pending orders"""
        self.pending_orders.append(signal)
    
    def process_bar(self, bar: OHLCVBar):
        """
        Process bar and attempt to fill pending orders.
        
        Fill Logic:
        - LONG: Fill if bar.low <= entry_price
        - SHORT: Fill if bar.high >= entry_price
        - Stop loss: Fill if price hits SL
        """
        instrument = str(bar.instrument)
        self.close_history[instrument].append(Decimal(str(bar.close)))
        filled_this_bar = []
        
        for order in self.pending_orders:
            if str(order.instrument) != instrument:
                continue
            fill_price = self._check_fill(order, bar)
            
            if fill_price is not None:
                # Apply spread and slippage from mid price.
                fill_price = self.slippage_model.apply(
                    fill_price, order.direction, str(bar.instrument)
                )
                
                adjusted_qty = self._resolve_quantity(order, bar, fill_price)
                if adjusted_qty <= 0:
                    continue

                # Calculate commission
                commission = self.commission_model.calculate(
                    fill_price, quantity=adjusted_qty
                )
                
                # Execute in portfolio
                if order.direction == SignalDirection.LONG:
                    self.portfolio.open_long(
                        str(bar.instrument), fill_price, quantity=adjusted_qty, commission=commission
                    )
                elif order.direction == SignalDirection.SHORT:
                    self.portfolio.open_short(
                        str(bar.instrument), fill_price, quantity=adjusted_qty, commission=commission
                    )
                
                # Record fill
                self.filled_orders.append({
                    'timestamp': bar.timestamp,
                    'signal': order,
                    'instrument': str(order.instrument),
                    'direction': order.direction,
                    'quantity': adjusted_qty,
                    'fill_price': fill_price,
                    'commission': commission,
                    'fill_reason': 'entry',
                })
                self.open_positions.append(
                    OpenPosition(
                        instrument=str(order.instrument),
                        direction=order.direction,
                        quantity=adjusted_qty,
                        stop_loss=order.stop_loss,
                        take_profit=order.take_profit,
                        entry_price=fill_price,
                        opened_timestamp=bar.timestamp,
                    )
                )
                
                filled_this_bar.append(order)
        
        # Remove filled orders
        for order in filled_this_bar:
            self.pending_orders.remove(order)
        
        # Check stop loss / take profit on existing positions
        self._check_exits(bar)
    
    def _check_fill(self, signal: Signal, bar: OHLCVBar) -> Optional[Decimal]:
        """Check if order should fill on this bar"""
        if self.fill_mode == "next_open":
            # Real-time style: signal generated on bar close, fill on next bar open.
            if bar.timestamp > signal.timestamp:
                return bar.open
            return None

        if signal.direction == SignalDirection.LONG:
            if bar.low <= signal.entry_price:
                return signal.entry_price
        elif signal.direction == SignalDirection.SHORT:
            if bar.high >= signal.entry_price:
                return signal.entry_price
        return None

    def _resolve_quantity(self, signal: Signal, bar: OHLCVBar, fill_price: Decimal) -> int:
        qty = int(signal.quantity)
        if self.volatility_targeting_enabled:
            qty = self._vol_target_quantity(signal, bar, qty)
        qty = self._apply_exposure_cap(fill_price, qty)
        if self.max_quantity is not None:
            qty = min(qty, self.max_quantity)
        qty = max(qty, 0)
        if qty and qty < self.min_quantity:
            return 0
        return qty

    def _vol_target_quantity(self, signal: Signal, bar: OHLCVBar, base_qty: int) -> int:
        instrument = str(signal.instrument)
        closes = list(self.close_history[instrument])
        if len(closes) < self.volatility_lookback_bars + 1:
            return base_qty

        returns: List[float] = []
        for i in range(1, len(closes)):
            prev = float(closes[i - 1])
            cur = float(closes[i])
            if prev <= 0:
                continue
            returns.append((cur / prev) - 1.0)
        if len(returns) < self.volatility_lookback_bars:
            return base_qty

        sigma = float((sum((r - (sum(returns) / len(returns))) ** 2 for r in returns) / len(returns)) ** 0.5)
        if sigma <= 0:
            return base_qty

        periods_per_year = max(1.0, 365.25 * 24.0 * 3600.0 / float(self.base_timeframe_seconds))
        annualized_sigma = sigma * sqrt(periods_per_year)
        if annualized_sigma <= 0:
            return base_qty

        scale = float(self.target_annual_volatility) / annualized_sigma
        scale = max(0.10, min(3.0, scale))
        return max(1, int(base_qty * scale))

    def _apply_exposure_cap(self, fill_price: Decimal, qty: int) -> int:
        if self.max_concurrent_exposure_pct is None:
            return qty
        gross = self.portfolio.gross_notional_exposure()
        cap_notional = self.initial_capital * self.max_concurrent_exposure_pct
        room = cap_notional - gross
        if room <= 0:
            return 0
        max_qty = int(room / fill_price)
        if max_qty <= 0:
            return 0
        return min(qty, max_qty)
    
    def _check_exits(self, bar: OHLCVBar):
        """Check if any positions should be closed (SL/TP)"""
        remaining_positions: List[OpenPosition] = []
        instrument = str(bar.instrument)

        for pos in self.open_positions:
            if pos.instrument != instrument:
                remaining_positions.append(pos)
                continue

            exit_reason = None
            mid_exit_price: Optional[Decimal] = None

            if pos.direction == SignalDirection.LONG:
                sl_hit = pos.stop_loss is not None and bar.low <= pos.stop_loss
                tp_hit = pos.take_profit is not None and bar.high >= pos.take_profit

                # Conservative same-bar assumption: if both are hit, stop loss first.
                if sl_hit:
                    exit_reason = "stop_loss"
                    mid_exit_price = pos.stop_loss
                elif tp_hit:
                    exit_reason = "take_profit"
                    mid_exit_price = pos.take_profit

                close_direction = SignalDirection.SHORT
            else:
                sl_hit = pos.stop_loss is not None and bar.high >= pos.stop_loss
                tp_hit = pos.take_profit is not None and bar.low <= pos.take_profit

                # Conservative same-bar assumption: if both are hit, stop loss first.
                if sl_hit:
                    exit_reason = "stop_loss"
                    mid_exit_price = pos.stop_loss
                elif tp_hit:
                    exit_reason = "take_profit"
                    mid_exit_price = pos.take_profit

                close_direction = SignalDirection.LONG

            if exit_reason is None or mid_exit_price is None:
                remaining_positions.append(pos)
                continue

            # Apply spread/slippage to exit fill as well.
            exit_fill_price = self.slippage_model.apply(
                mid_exit_price, close_direction, pos.instrument
            )
            exit_commission = self.commission_model.calculate(
                exit_fill_price, quantity=pos.quantity
            )

            # Net out the position in portfolio.
            if close_direction == SignalDirection.LONG:
                self.portfolio.open_long(
                    pos.instrument,
                    exit_fill_price,
                    quantity=pos.quantity,
                    commission=exit_commission,
                )
            else:
                self.portfolio.open_short(
                    pos.instrument,
                    exit_fill_price,
                    quantity=pos.quantity,
                    commission=exit_commission,
                )

            self.filled_orders.append(
                {
                    "timestamp": bar.timestamp,
                    "signal": None,
                    "instrument": pos.instrument,
                    "direction": close_direction,
                    "quantity": pos.quantity,
                    "fill_price": exit_fill_price,
                    "commission": exit_commission,
                    "fill_reason": exit_reason,
                }
            )

        self.open_positions = remaining_positions
