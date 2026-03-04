#!/usr/bin/env python3
"""Operator API for execution safety state."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from aiohttp import web

from oanda_bot.utils.config import Config
from oanda_bot.utils.message_bus import MessageBus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution operator status API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def _count_by_pattern(redis_client, pattern: str) -> int:
    count = 0
    async for _ in redis_client.scan_iter(match=pattern):
        count += 1
    return count


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "timestamp": _now_iso()})


async def handle_execution_state(request: web.Request) -> web.Response:
    bus: MessageBus = request.app["message_bus"]
    redis_client = bus.redis_client
    if redis_client is None:
        return web.json_response(
            {"status": "error", "error": "redis_not_connected", "timestamp": _now_iso()},
            status=503,
        )

    kill_switch = await redis_client.get("execution:state:kill_switch")
    shadow_mode = await redis_client.get("execution:state:shadow_mode")
    guardrail = await redis_client.get("execution:state:live_guardrail_blocked")
    updated_at = await redis_client.get("execution:state:updated_at")

    inflight_count = await _count_by_pattern(redis_client, "execution:signal:inflight:*")
    executed_count = await _count_by_pattern(redis_client, "execution:signal:executed:*")

    payload: Dict[str, Any] = {
        "status": "ok",
        "timestamp": _now_iso(),
        "execution_safety": {
            "kill_switch_active": _as_bool(kill_switch),
            "shadow_mode_active": _as_bool(shadow_mode),
            "live_guardrail_blocked": _as_bool(guardrail),
            "updated_at": updated_at,
        },
        "idempotency": {
            "signals_inflight": inflight_count,
            "signals_executed_cached": executed_count,
        },
    }
    return web.json_response(payload)


async def create_app() -> web.Application:
    config = Config.load()
    message_bus = MessageBus(config)
    await message_bus.connect()

    app = web.Application()
    app["message_bus"] = message_bus
    app.router.add_get("/health", handle_health)
    app.router.add_get("/execution/state", handle_execution_state)

    async def on_shutdown(app: web.Application):
        await app["message_bus"].disconnect()

    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    args = parse_args()
    app = asyncio.run(create_app())
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
