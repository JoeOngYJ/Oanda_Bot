"""
Risk Limits - Defines and enforces trading risk limits.
All limits are loaded from configuration and enforced pre-trade.
"""

from typing import Dict, Optional
from decimal import Decimal
from oanda_bot.utils.config import RiskLimitsConfig
from oanda_bot.utils.models import Instrument
import logging

logger = logging.getLogger(__name__)


class RiskLimits:
    """
    Manages and enforces risk limits for trading operations.
    All limits are configurable and validated against trades.
    """

    def __init__(self, config: RiskLimitsConfig):
        self.config = config

        # Daily tracking
        self.daily_pnl = Decimal("0")
        self.daily_trades = 0

        # Position tracking
        self.open_positions: Dict[Instrument, Decimal] = {}
        self.total_exposure = Decimal("0")

        # Circuit breaker state
        self.consecutive_losses = 0
        self.circuit_breaker_active = False

        logger.info("Risk limits initialized")
        logger.info(f"Max daily loss: {self.config.max_daily_loss}")
        logger.info(f"Max drawdown: {self.config.max_drawdown}")
        logger.info(f"Max total exposure: {self.config.max_total_exposure}")

    def get_max_position_size(self, instrument: Instrument) -> int:
        """Get maximum position size for an instrument"""
        return self.config.per_instrument.max_position_size

    def get_max_order_size(self, instrument: Instrument) -> int:
        """Get maximum order size for an instrument"""
        return self.config.per_instrument.max_order_size

    def get_max_daily_loss(self) -> float:
        """Get maximum daily loss limit"""
        return self.config.max_daily_loss

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown limit"""
        return self.config.max_drawdown

    def get_max_total_exposure(self) -> float:
        """Get maximum total exposure limit"""
        return self.config.max_total_exposure

    def get_max_leverage(self) -> float:
        """Get maximum leverage limit"""
        return self.config.max_leverage

    def get_min_account_balance(self) -> float:
        """Get minimum account balance requirement"""
        return self.config.min_account_balance

    def requires_stop_loss(self) -> bool:
        """Check if stop loss is required for all trades"""
        return self.config.require_stop_loss

    def get_max_stop_loss_distance(self) -> float:
        """Get maximum stop loss distance"""
        return self.config.max_stop_loss_distance

    def get_max_correlated_exposure(self) -> float:
        """Get maximum correlated exposure limit"""
        return self.config.max_correlated_exposure
