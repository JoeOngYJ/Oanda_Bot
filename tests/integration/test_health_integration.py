#!/usr/bin/env python3
"""Test health checker integration"""

import asyncio
from agents.monitoring.health_checker import HealthChecker
from shared.config import Config


async def test_health_checker():
    """Test health checker functionality"""
    print("Testing health checker...")

    config = Config.load()
    checker = HealthChecker(config)

    # Check all components
    metrics = await checker.check_all_components()

    print(f"\n✓ Health check completed - {len(metrics)} metrics collected:")
    for m in metrics:
        print(f"  • {m.component}.{m.metric_name}: {m.value:.2f} ({m.status})")

    print("\n✓ Health checker test passed!")


if __name__ == '__main__':
    asyncio.run(test_health_checker())
