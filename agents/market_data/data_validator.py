"""
Validates market data for anomalies, gaps, and sanity.
All checks are deterministic.
"""

import os
from datetime import datetime, timezone
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
        self.max_staleness_seconds = float(
            os.getenv("MARKET_DATA_MAX_STALENESS_SECONDS", "10.0")
        )

    @staticmethod
    def _is_fx_market_open(now_utc: datetime) -> bool:
        """Approximate OTC FX/metals open window: Sun 21:00 UTC -> Fri 22:00 UTC."""
        wd = now_utc.weekday()  # Mon=0 ... Sun=6
        hour = now_utc.hour
        if wd in (0, 1, 2, 3):
            return True
        if wd == 4:
            return hour < 22
        if wd == 6:
            return hour >= 21
        return False

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
        now = datetime.now(timezone.utc)
        tick_ts = tick.timestamp
        if tick_ts.tzinfo is None:
            tick_ts = tick_ts.replace(tzinfo=timezone.utc)
        else:
            tick_ts = tick_ts.astimezone(timezone.utc)
        age = (now - tick_ts).total_seconds()
        if self._is_fx_market_open(now) and age > self.max_staleness_seconds:
            issues.append(f"Stale tick: {age:.1f}s old")

        # Update last tick
        self.last_tick[tick.instrument] = tick

        is_valid = len(issues) == 0
        return is_valid, issues
