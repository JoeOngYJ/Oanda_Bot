#!/usr/bin/env python3
"""Test alert manager integration"""

import asyncio
from agents.monitoring.alerting import AlertManager
from shared.config import Config
from shared.message_bus import MessageBus


async def test_alert_manager():
    """Test alert manager functionality"""
    print("Testing alert manager...")

    config = Config.load()
    bus = MessageBus(config)
    await bus.connect()

    alert_mgr = AlertManager(config)

    # Send test alert
    await alert_mgr.send_alert(
        severity='warning',
        component='test',
        message='Test alert message',
        value=42.5
    )

    print('✓ Alert sent successfully')

    await bus.disconnect()
    print('\n✓ Alert manager test passed!')


if __name__ == '__main__':
    asyncio.run(test_alert_manager())
