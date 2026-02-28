"""
Validates market data for anomalies, gaps, and sanity.
All checks are deterministic.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Tuple, List, Dict
from shared.models import MarketTick, Instrument
from shared.config import Config


class DataValidator:
    """
    Validates market data for anomalies, gaps, and sanity.
    All checks are deterministic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.last_tick: Dict[Instrument, MarketTick] = {}

        # Validation thresholds
        self.max_spread_pct = 0.005  # 0.5% max spread
        self.max_price_jump_pct = 0.02  # 2% max price jump
        self.max_staleness_seconds = 10.0

    def validate(self, tick: MarketTick) -> Tuple[bool, List[str]]:
        """
        Validate a market tick.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # 1. Price sanity: bid < ask
        if tick.bid >= tick.ask:
            issues.append(f"Invalid bid/ask: bid={tick.bid} >= ask={tick.ask}")

        # 2. Spread check: not too wide
        spread_pct = float(tick.spread / tick.bid)
        if spread_pct > self.max_spread_pct:
            issues.append(f"Excessive spread: {spread_pct:.4%}")

        # 3. Price jump check: compare to last tick
        if tick.instrument in self.last_tick:
            last = self.last_tick[tick.instrument]
            mid_price = (tick.bid + tick.ask) / 2
            last_mid = (last.bid + last.ask) / 2

            price_change_pct = abs(float((mid_price - last_mid) / last_mid))
            if price_change_pct > self.max_price_jump_pct:
                issues.append(
                    f"Large price jump: {price_change_pct:.4%} "
                    f"from {last_mid} to {mid_price}"
                )

            # 4. Timestamp ordering
            if tick.timestamp < last.timestamp:
                issues.append(
                    f"Out-of-order tick: {tick.timestamp} < {last.timestamp}"
                )

        # 5. Timestamp freshness
        now = datetime.utcnow()
        age = (now - tick.timestamp).total_seconds()
        if age > self.max_staleness_seconds:
            issues.append(f"Stale tick: {age:.1f}s old")

        # Update last tick
        self.last_tick[tick.instrument] = tick

        is_valid = len(issues) == 0
        return is_valid, issues
