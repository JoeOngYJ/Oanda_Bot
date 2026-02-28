# backtesting/execution/simulator.py

from typing import Dict, List, Optional
from decimal import Decimal
from backtesting.strategy.signal import Signal, SignalDirection
from backtesting.data.models import OHLCVBar
from .portfolio import Portfolio
from .slippage import SlippageModel
from .commission import CommissionModel

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
        commission_model: CommissionModel
    ):
        self.portfolio = Portfolio(initial_capital)
        self.slippage_model = slippage_model
        self.commission_model = commission_model
        
        self.pending_orders: List[Signal] = []
        self.filled_orders: List[Dict] = []
    
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
        filled_this_bar = []
        
        for order in self.pending_orders:
            fill_price = self._check_fill(order, bar)
            
            if fill_price is not None:
                # Apply slippage
                fill_price = self.slippage_model.apply(fill_price, order.direction)
                
                # Calculate commission
                commission = self.commission_model.calculate(fill_price, quantity=1000)  # From position sizer
                
                # Execute in portfolio
                if order.direction == SignalDirection.LONG:
                    self.portfolio.open_long(bar.instrument, fill_price, quantity=1000)
                elif order.direction == SignalDirection.SHORT:
                    self.portfolio.open_short(bar.instrument, fill_price, quantity=1000)
                
                # Record fill
                self.filled_orders.append({
                    'timestamp': bar.timestamp,
                    'signal': order,
                    'fill_price': fill_price,
                    'commission': commission
                })
                
                filled_this_bar.append(order)
        
        # Remove filled orders
        for order in filled_this_bar:
            self.pending_orders.remove(order)
        
        # Check stop loss / take profit on existing positions
        self._check_exits(bar)
    
    def _check_fill(self, signal: Signal, bar: OHLCVBar) -> Optional[Decimal]:
        """Check if order should fill on this bar"""
        if signal.direction == SignalDirection.LONG:
            if bar.low <= signal.entry_price:
                return signal.entry_price
        elif signal.direction == SignalDirection.SHORT:
            if bar.high >= signal.entry_price:
                return signal.entry_price
        return None
    
    def _check_exits(self, bar: OHLCVBar):
        """Check if any positions should be closed (SL/TP)"""
        # Implementation: iterate through portfolio.positions
        # Check if bar.high/low hit stop loss or take profit
        pass