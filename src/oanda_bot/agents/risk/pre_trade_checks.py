"""
Pre-Trade Checks - Validates trade signals against risk limits.
All checks must pass before a signal is converted to an order.
"""

from typing import Optional, List
from decimal import Decimal
from oanda_bot.utils.models import TradeSignal, RiskCheckResult, Instrument
from .limits import RiskLimits
import logging

logger = logging.getLogger(__name__)


class PreTradeChecker:
    """
    Validates trade signals against risk limits.
    Returns detailed results for each check.
    """

    def __init__(self, risk_limits: RiskLimits, account_balance: Decimal):
        self.risk_limits = risk_limits
        self.account_balance = account_balance

    async def check_signal(self, signal: TradeSignal) -> RiskCheckResult:
        """
        Run all pre-trade checks on a signal.

        Args:
            signal: Trade signal to validate

        Returns:
            RiskCheckResult with approval status and reasons
        """
        checks = [
            self._check_order_size(signal),
            self._check_position_size(signal),
            self._check_stop_loss(signal),
            self._check_leverage(signal),
            self._check_account_balance(signal),
            self._check_total_exposure(signal),
            self._check_daily_loss_limit(signal),
            self._check_circuit_breaker(signal),
        ]

        # Collect all failed checks
        failed_checks = [check for check in checks if not check["passed"]]

        approved = len(failed_checks) == 0
        reasons = [check["reason"] for check in failed_checks] if failed_checks else []

        if not approved:
            logger.warning(
                f"Signal rejected: {signal.strategy_name} - "
                f"{signal.side.value} {signal.quantity} {signal.instrument.value}. "
                f"Reasons: {', '.join(reasons)}"
            )
        else:
            logger.info(
                f"Signal approved: {signal.strategy_name} - "
                f"{signal.side.value} {signal.quantity} {signal.instrument.value}"
            )

        return RiskCheckResult(
            signal_id=signal.signal_id,
            approved=approved,
            reasons=reasons
        )

    def _check_order_size(self, signal: TradeSignal) -> dict:
        """Check if order size is within limits"""
        max_order_size = self.risk_limits.get_max_order_size(signal.instrument)

        if signal.quantity > max_order_size:
            return {
                "passed": False,
                "reason": f"Order size {signal.quantity} exceeds max {max_order_size}"
            }

        return {"passed": True, "reason": ""}

    def _check_position_size(self, signal: TradeSignal) -> dict:
        """Check if resulting position size is within limits"""
        max_position_size = self.risk_limits.get_max_position_size(signal.instrument)

        # Get current position for this instrument
        current_position = self.risk_limits.open_positions.get(signal.instrument, Decimal("0"))

        # Calculate resulting position
        if signal.side.value == "BUY":
            resulting_position = abs(current_position + signal.quantity)
        else:  # SELL
            resulting_position = abs(current_position - signal.quantity)

        if resulting_position > max_position_size:
            return {
                "passed": False,
                "reason": f"Resulting position {resulting_position} exceeds max {max_position_size}"
            }

        return {"passed": True, "reason": ""}

    def _check_stop_loss(self, signal: TradeSignal) -> dict:
        """Check if stop loss is present and within limits"""
        if self.risk_limits.requires_stop_loss():
            if signal.stop_loss is None:
                return {
                    "passed": False,
                    "reason": "Stop loss is required but not provided"
                }

            # Check stop loss distance
            max_sl_distance = self.risk_limits.get_max_stop_loss_distance()
            sl_distance = abs(float(signal.entry_price - signal.stop_loss))

            if sl_distance > max_sl_distance:
                return {
                    "passed": False,
                    "reason": f"Stop loss distance {sl_distance:.5f} exceeds max {max_sl_distance}"
                }

        return {"passed": True, "reason": ""}

    def _check_leverage(self, signal: TradeSignal) -> dict:
        """Check if trade would exceed leverage limits"""
        max_leverage = self.risk_limits.get_max_leverage()

        # Calculate position value
        position_value = float(signal.quantity * signal.entry_price)

        # Calculate leverage
        leverage = position_value / float(self.account_balance)

        if leverage > max_leverage:
            return {
                "passed": False,
                "reason": f"Leverage {leverage:.2f}x exceeds max {max_leverage}x"
            }

        return {"passed": True, "reason": ""}

    def _check_account_balance(self, signal: TradeSignal) -> dict:
        """Check if account balance is above minimum"""
        min_balance = self.risk_limits.get_min_account_balance()

        if float(self.account_balance) < min_balance:
            return {
                "passed": False,
                "reason": f"Account balance {self.account_balance} below minimum {min_balance}"
            }

        return {"passed": True, "reason": ""}

    def _check_total_exposure(self, signal: TradeSignal) -> dict:
        """Check if total exposure would exceed limits"""
        max_exposure = self.risk_limits.get_max_total_exposure()

        # Calculate new exposure
        trade_value = float(signal.quantity * signal.entry_price)
        new_total_exposure = float(self.risk_limits.total_exposure) + trade_value

        if new_total_exposure > max_exposure:
            return {
                "passed": False,
                "reason": f"Total exposure {new_total_exposure:.2f} exceeds max {max_exposure}"
            }

        return {"passed": True, "reason": ""}

    def _check_daily_loss_limit(self, signal: TradeSignal) -> dict:
        """Check if daily loss limit has been reached"""
        max_daily_loss = self.risk_limits.get_max_daily_loss()

        if float(self.risk_limits.daily_pnl) <= -max_daily_loss:
            return {
                "passed": False,
                "reason": f"Daily loss limit reached: {self.risk_limits.daily_pnl}"
            }

        return {"passed": True, "reason": ""}

    def _check_circuit_breaker(self, signal: TradeSignal) -> dict:
        """Check if circuit breaker is active"""
        if self.risk_limits.circuit_breaker_active:
            return {
                "passed": False,
                "reason": "Circuit breaker is active - trading halted"
            }

        return {"passed": True, "reason": ""}
