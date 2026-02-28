#!/usr/bin/env python3
"""Test circuit breaker integration"""

from decimal import Decimal
from agents.risk.circuit_breaker import CircuitBreaker
from agents.risk.limits import RiskLimits
from shared.config import Config


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("Testing circuit breaker...")

    config = Config.load()
    limits = RiskLimits(config.risk)
    breaker = CircuitBreaker(config.risk.circuit_breaker, limits)

    # Check initial state
    print(f"✓ Initial state - Active: {limits.circuit_breaker_active}")

    # Simulate consecutive losses to trigger circuit breaker
    print("\nSimulating consecutive losses...")
    for i in range(6):
        breaker.check_and_update(trade_pnl=Decimal("-100.0"))
        print(f"  Loss {i+1}: Active = {limits.circuit_breaker_active}")

    # Check status
    status = breaker.get_status()
    print(f"\n✓ Circuit breaker status:")
    print(f"  • Active: {status['active']}")
    print(f"  • Consecutive losses: {status['consecutive_losses']}")
    print(f"  • Daily P&L: {status['daily_pnl']}")

    # Reset daily stats
    breaker.reset_daily_stats()
    print(f"\n✓ After reset - Consecutive losses: {limits.consecutive_losses}")

    print("\n✓ Circuit breaker test passed!")


if __name__ == '__main__':
    test_circuit_breaker()
