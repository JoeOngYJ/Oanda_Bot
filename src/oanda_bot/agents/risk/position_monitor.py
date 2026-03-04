"""
Position Monitor - Tracks open positions and enforces stop loss/take profit.
Monitors positions in real-time and generates close signals when limits are hit.
"""

from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from oanda_bot.utils.models import Position, MarketTick, TradeSignal, Side, Instrument
from .limits import RiskLimits
import logging

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitors open positions and enforces stop loss/take profit levels.
    Generates close signals when risk limits are breached.
    """

    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.positions: Dict[str, Position] = {}  # position_id -> Position

    def add_position(self, position: Position) -> None:
        """
        Add a new position to monitor.

        Args:
            position: Position to track
        """
        self.positions[position.position_id] = position

        # Update risk limits tracking
        if position.instrument not in self.risk_limits.open_positions:
            self.risk_limits.open_positions[position.instrument] = Decimal("0")

        if position.side == Side.BUY:
            self.risk_limits.open_positions[position.instrument] += position.quantity
        else:
            self.risk_limits.open_positions[position.instrument] -= position.quantity

        # Update total exposure
        position_value = position.quantity * position.entry_price
        self.risk_limits.total_exposure += position_value

        logger.info(
            f"Position added: {position.position_id} - "
            f"{position.side.value} {position.quantity} {position.instrument.value} "
            f"@ {position.entry_price}"
        )

    def remove_position(self, position_id: str) -> Optional[Position]:
        """
        Remove a position from monitoring.

        Args:
            position_id: ID of position to remove

        Returns:
            Removed position or None if not found
        """
        position = self.positions.pop(position_id, None)

        if position:
            # Update risk limits tracking
            if position.side == Side.BUY:
                self.risk_limits.open_positions[position.instrument] -= position.quantity
            else:
                self.risk_limits.open_positions[position.instrument] += position.quantity

            # Update total exposure
            position_value = position.quantity * position.entry_price
            self.risk_limits.total_exposure -= position_value

            logger.info(f"Position removed: {position_id}")

        return position

    async def check_positions(self, tick: MarketTick) -> List[TradeSignal]:
        """
        Check all positions against current market price.
        Generate close signals if stop loss or take profit is hit.

        Args:
            tick: Current market tick

        Returns:
            List of close signals to execute
        """
        close_signals = []

        for position_id, position in list(self.positions.items()):
            # Only check positions for this instrument
            if position.instrument != tick.instrument:
                continue

            # Check stop loss
            if position.stop_loss:
                if self._is_stop_loss_hit(position, tick):
                    signal = self._create_close_signal(
                        position, tick, "Stop loss hit"
                    )
                    close_signals.append(signal)
                    logger.warning(
                        f"Stop loss hit for {position_id}: "
                        f"Price {tick.bid if position.side == Side.BUY else tick.ask} "
                        f"vs SL {position.stop_loss}"
                    )
                    continue

            # Check take profit
            if position.take_profit:
                if self._is_take_profit_hit(position, tick):
                    signal = self._create_close_signal(
                        position, tick, "Take profit hit"
                    )
                    close_signals.append(signal)
                    logger.info(
                        f"Take profit hit for {position_id}: "
                        f"Price {tick.bid if position.side == Side.BUY else tick.ask} "
                        f"vs TP {position.take_profit}"
                    )
                    continue

            # Update unrealized P&L
            position.unrealized_pnl = self._calculate_unrealized_pnl(position, tick)

        return close_signals

    def _is_stop_loss_hit(self, position: Position, tick: MarketTick) -> bool:
        """Check if stop loss level has been hit"""
        if position.side == Side.BUY:
            # For long positions, stop loss is below entry
            return tick.bid <= position.stop_loss
        else:
            # For short positions, stop loss is above entry
            return tick.ask >= position.stop_loss

    def _is_take_profit_hit(self, position: Position, tick: MarketTick) -> bool:
        """Check if take profit level has been hit"""
        if position.side == Side.BUY:
            # For long positions, take profit is above entry
            return tick.bid >= position.take_profit
        else:
            # For short positions, take profit is below entry
            return tick.ask <= position.take_profit

    def _calculate_unrealized_pnl(self, position: Position, tick: MarketTick) -> Decimal:
        """Calculate unrealized P&L for a position"""
        if position.side == Side.BUY:
            # Long position: profit when price goes up
            pnl = (tick.bid - position.entry_price) * position.quantity
        else:
            # Short position: profit when price goes down
            pnl = (position.entry_price - tick.ask) * position.quantity

        return pnl

    def _create_close_signal(
        self, position: Position, tick: MarketTick, reason: str
    ) -> TradeSignal:
        """Create a close signal for a position"""
        # Close signal is opposite side of position
        close_side = Side.SELL if position.side == Side.BUY else Side.BUY

        return TradeSignal(
            signal_id="",  # Will be set by signal generator
            timestamp=datetime.utcnow(),
            strategy_name=f"RiskManager_{reason.replace(' ', '_')}",
            instrument=position.instrument,
            side=close_side,
            quantity=position.quantity,
            entry_price=tick.bid if close_side == Side.SELL else tick.ask,
            stop_loss=None,
            take_profit=None,
            confidence=1.0,  # Risk management signals have max confidence
            rationale=f"{reason} for position {position.position_id}"
        )

    def get_position_summary(self) -> Dict:
        """Get summary of all open positions"""
        summary = {
            "total_positions": len(self.positions),
            "by_instrument": {},
            "total_unrealized_pnl": Decimal("0")
        }

        for position in self.positions.values():
            instrument = position.instrument.value

            if instrument not in summary["by_instrument"]:
                summary["by_instrument"][instrument] = {
                    "count": 0,
                    "long": 0,
                    "short": 0,
                    "unrealized_pnl": Decimal("0")
                }

            summary["by_instrument"][instrument]["count"] += 1

            if position.side == Side.BUY:
                summary["by_instrument"][instrument]["long"] += 1
            else:
                summary["by_instrument"][instrument]["short"] += 1

            summary["by_instrument"][instrument]["unrealized_pnl"] += position.unrealized_pnl
            summary["total_unrealized_pnl"] += position.unrealized_pnl

        return summary
