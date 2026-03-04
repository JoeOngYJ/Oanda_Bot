"""
Circuit Breaker - Halts trading when risk thresholds are exceeded.
Monitors daily loss, drawdown, consecutive losses, and loss velocity.
"""

from typing import List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from oanda_bot.utils.config import CircuitBreakerConfig
from .limits import RiskLimits
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Monitors trading activity and activates circuit breaker
    when risk thresholds are exceeded.
    """

    def __init__(self, config: CircuitBreakerConfig, risk_limits: RiskLimits):
        self.config = config
        self.risk_limits = risk_limits

        # Track recent trades for loss velocity
        self.recent_trades: List[dict] = []

        # Track peak balance for drawdown calculation
        self.peak_balance = Decimal("0")
        self.current_balance = Decimal("0")

        logger.info("Circuit breaker initialized")
        logger.info(f"Consecutive losses threshold: {self.config.consecutive_losses}")
        logger.info(f"Loss velocity 1h threshold: {self.config.loss_velocity_1h}")
        logger.info(f"Volatility spike threshold: {self.config.volatility_spike_threshold}")

    def check_and_update(
        self,
        trade_pnl: Optional[Decimal] = None,
        current_balance: Optional[Decimal] = None
    ) -> bool:
        """
        Check if circuit breaker should be activated.

        Args:
            trade_pnl: P&L from a completed trade (if any)
            current_balance: Current account balance

        Returns:
            True if circuit breaker was activated, False otherwise
        """
        if current_balance:
            self.current_balance = current_balance

            # Update peak balance for drawdown calculation
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance

        # Record trade if provided
        if trade_pnl is not None:
            self._record_trade(trade_pnl)

        # Run all circuit breaker checks
        checks = [
            self._check_consecutive_losses(),
            self._check_loss_velocity(),
            self._check_daily_loss(),
            self._check_drawdown()
        ]

        # Activate if any check fails
        for check_name, should_activate, reason in checks:
            if should_activate:
                self._activate(reason)
                return True

        return False

    def _record_trade(self, pnl: Decimal) -> None:
        """Record a trade for loss velocity tracking"""
        trade_record = {
            "timestamp": datetime.utcnow(),
            "pnl": pnl,
            "is_loss": pnl < 0
        }
        self.recent_trades.append(trade_record)

        # Update consecutive losses counter
        if pnl < 0:
            self.risk_limits.consecutive_losses += 1
        else:
            self.risk_limits.consecutive_losses = 0

        # Update daily P&L
        self.risk_limits.daily_pnl += pnl

        # Clean up old trades (keep last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.recent_trades = [
            t for t in self.recent_trades
            if t["timestamp"] > cutoff_time
        ]

    def _check_consecutive_losses(self) -> tuple:
        """Check if consecutive losses threshold is exceeded"""
        threshold = self.config.consecutive_losses

        if self.risk_limits.consecutive_losses >= threshold:
            return (
                "consecutive_losses",
                True,
                f"Consecutive losses ({self.risk_limits.consecutive_losses}) "
                f"exceeded threshold ({threshold})"
            )

        return ("consecutive_losses", False, "")

    def _check_loss_velocity(self) -> tuple:
        """Check if loss velocity in last hour exceeds threshold"""
        threshold = self.config.loss_velocity_1h

        # Calculate losses in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_losses = [
            t["pnl"] for t in self.recent_trades
            if t["timestamp"] > one_hour_ago and t["is_loss"]
        ]

        if recent_losses:
            total_loss = abs(sum(recent_losses))

            if float(total_loss) > threshold:
                return (
                    "loss_velocity",
                    True,
                    f"Loss velocity in 1h ({total_loss:.2f}) "
                    f"exceeded threshold ({threshold})"
                )

        return ("loss_velocity", False, "")

    def _check_daily_loss(self) -> tuple:
        """Check if daily loss limit is exceeded"""
        max_daily_loss = self.risk_limits.get_max_daily_loss()

        if float(self.risk_limits.daily_pnl) <= -max_daily_loss:
            return (
                "daily_loss",
                True,
                f"Daily loss ({self.risk_limits.daily_pnl:.2f}) "
                f"exceeded limit ({max_daily_loss})"
            )

        return ("daily_loss", False, "")

    def _check_drawdown(self) -> tuple:
        """Check if drawdown exceeds maximum threshold"""
        max_drawdown = self.risk_limits.get_max_drawdown()

        if self.peak_balance > 0:
            current_drawdown = float(
                (self.peak_balance - self.current_balance) / self.peak_balance
            )

            if current_drawdown > max_drawdown:
                return (
                    "drawdown",
                    True,
                    f"Drawdown ({current_drawdown:.2%}) "
                    f"exceeded maximum ({max_drawdown:.2%})"
                )

        return ("drawdown", False, "")

    def _activate(self, reason: str) -> None:
        """Activate the circuit breaker"""
        self.risk_limits.circuit_breaker_active = True

        logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        logger.critical("All trading has been halted")

    def deactivate(self) -> None:
        """Manually deactivate the circuit breaker"""
        self.risk_limits.circuit_breaker_active = False
        logger.warning("Circuit breaker manually deactivated")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of trading day)"""
        self.risk_limits.daily_pnl = Decimal("0")
        self.risk_limits.daily_trades = 0
        self.risk_limits.consecutive_losses = 0

        logger.info("Daily statistics reset")

    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            "active": self.risk_limits.circuit_breaker_active,
            "consecutive_losses": self.risk_limits.consecutive_losses,
            "daily_pnl": float(self.risk_limits.daily_pnl),
            "current_drawdown": float(
                (self.peak_balance - self.current_balance) / self.peak_balance
            ) if self.peak_balance > 0 else 0.0,
            "recent_trades_1h": len([
                t for t in self.recent_trades
                if t["timestamp"] > datetime.utcnow() - timedelta(hours=1)
            ])
        }
