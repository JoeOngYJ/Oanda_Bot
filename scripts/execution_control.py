#!/usr/bin/env python3
"""Publish execution control commands (kill-switch / shadow mode)."""

import argparse
import asyncio

from shared.config import Config
from shared.message_bus import MessageBus


def parse_args():
    parser = argparse.ArgumentParser(description="Execution control command publisher.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["kill_switch_on", "kill_switch_off", "shadow_mode_on", "shadow_mode_off"],
    )
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--requested-by", default="operator")
    return parser.parse_args()


async def main():
    args = parse_args()
    config = Config.load()
    bus = MessageBus(config)
    await bus.connect()
    try:
        payload = {
            "action": args.action,
            "reason": args.reason,
            "requested_by": args.requested_by,
        }
        await bus.publish("execution_control", payload)
        print(f"Published execution control command: {payload}")
    finally:
        await bus.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
