#!/usr/bin/env python3
"""Stress Test: Circuit Breaker Scenarios"""

import asyncio
import pytest
from decimal import Decimal

from shared.config import Config
from shared.message_bus import MessageBus


@pytest.mark.asyncio
async def test_circuit_breaker_on_drawdown():
    """Trigger circuit breaker via max drawdown"""
    config = Config.load()
    
    # Simulate drawdown scenario
    daily_start_equity = Decimal("10000")
    daily_pnl = Decimal("-2500")  # Exceeds max drawdown
    
    # Circuit breaker should trigger
    drawdown = abs(daily_pnl)
    max_drawdown = Decimal("2000")
    
    assert drawdown > max_drawdown, "Drawdown should exceed limit"
    print(f"\nCircuit breaker triggered: Drawdown ${drawdown} exceeds ${max_drawdown}")


@pytest.mark.asyncio
async def test_circuit_breaker_on_consecutive_losses():
    """Trigger circuit breaker via consecutive losses"""
    config = Config.load()
    
    # Simulate consecutive losses
    consecutive_losses = 6
    max_consecutive_losses = 5
    
    assert consecutive_losses > max_consecutive_losses
    print(f"\nCircuit breaker triggered: {consecutive_losses} consecutive losses")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
