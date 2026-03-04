#!/usr/bin/env python3
"""Discord bot that posts signal/order/execution updates with current balance."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import aiohttp
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountSummary

from oanda_bot.utils.config import Config
from oanda_bot.utils.message_bus import MessageBus
from oanda_bot.utils.models import Execution, Order, TradeSignal


DEFAULT_CHANNEL_ID = "1477609642258337954"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discord execution notifier bot")
    p.add_argument("--balance-timeout-seconds", type=float, default=10.0, help="Timeout for account balance lookup")
    return p.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[discord-exec-bot] {_now_iso()} {msg}", flush=True)


class DiscordExecutionNotifierBot:
    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.message_bus = MessageBus(config)
        self.running = False

        self.discord_token = (
            os.getenv("DISCORD_EXEC_BOT_TOKEN", "").strip()
            or os.getenv("DISCORD_BOT_TOKEN", "").strip()
        )
        self.discord_channel_id = (
            os.getenv("DISCORD_EXEC_CHANNEL_ID", "").strip()
            or os.getenv("DISCORD_CHANNEL_ID", "").strip()
            or DEFAULT_CHANNEL_ID
        )
        if not self.discord_token:
            raise SystemExit("Set DISCORD_EXEC_BOT_TOKEN or DISCORD_BOT_TOKEN")

        self._http_headers = {
            "Authorization": f"Bot {self.discord_token}",
            "Content-Type": "application/json",
        }

        self.oanda_api = API(
            access_token=config.oanda.api_token,
            environment=config.oanda.environment,
        )
        self.oanda_account_id = config.oanda.account_id

    async def start(self) -> None:
        self.running = True
        await self.message_bus.connect()
        _log("connected to redis/message bus")

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            self.http = session
            await self._send_discord_message(
                (
                    "Discord execution notifier online.\n"
                    f"- channel_id: `{self.discord_channel_id}`\n"
                    f"- started_at: `{_now_iso()}`"
                )
            )
            await asyncio.gather(
                self._signal_loop(),
                self._order_loop(),
                self._execution_loop(),
            )

    async def stop(self) -> None:
        self.running = False
        await self.message_bus.disconnect()

    async def _execution_loop(self) -> None:
        async for message in self.message_bus.subscribe("executions"):
            if not self.running:
                break

            try:
                execution = Execution(**message)
                text = await self._build_execution_text(execution)
                await self._send_discord_message(text)
            except Exception as e:
                _log(f"execution loop error: {type(e).__name__}: {e}")
                await asyncio.sleep(0.5)

    async def _signal_loop(self) -> None:
        async for message in self.message_bus.subscribe("signals"):
            if not self.running:
                break
            try:
                signal = TradeSignal(**message)
                text = (
                    "SIGNAL\n"
                    f"- signal_id: `{signal.signal_id}`\n"
                    f"- instrument: `{signal.instrument.value}`\n"
                    f"- side: `{signal.side.value}`\n"
                    f"- quantity: `{signal.quantity}`\n"
                    f"- confidence: `{signal.confidence}`\n"
                    f"- strategy: `{signal.strategy_name}`\n"
                    f"- time: `{signal.timestamp}`"
                )
                await self._send_discord_message(text)
            except Exception as e:
                _log(f"signal loop error: {type(e).__name__}: {e}")
                await asyncio.sleep(0.5)

    async def _order_loop(self) -> None:
        async for message in self.message_bus.subscribe("orders"):
            if not self.running:
                break
            try:
                order = Order(**message)
                balance_text, nav_text, _ = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(None, self._fetch_balance_and_nav),
                    timeout=max(1.0, float(self.args.balance_timeout_seconds)),
                )
                text = (
                    "ORDER\n"
                    f"- order_id: `{order.order_id}`\n"
                    f"- signal_id: `{order.signal_id}`\n"
                    f"- instrument: `{order.instrument.value}`\n"
                    f"- side: `{order.side.value}`\n"
                    f"- quantity: `{order.quantity}`\n"
                    f"- status: `{order.status.value}`\n"
                    f"- balance: `{balance_text}`\n"
                    f"- NAV: `{nav_text}`\n"
                    f"- time: `{order.updated_at}`"
                )
                await self._send_discord_message(text)
            except Exception as e:
                _log(f"order loop error: {type(e).__name__}: {e}")
                await asyncio.sleep(0.5)

    async def _build_execution_text(self, execution: Execution) -> str:
        event_name = self._classify_event(execution)

        balance_text = "unavailable"
        nav_text = "unavailable"
        balance_error: Optional[str] = None

        try:
            balance_text, nav_text, balance_error = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, self._fetch_balance_and_nav),
                timeout=max(1.0, float(self.args.balance_timeout_seconds)),
            )
        except Exception as e:
            balance_error = f"{type(e).__name__}: {e}"

        lines = [
            f"{event_name}",
            f"- instrument: `{execution.instrument.value}`",
            f"- side: `{execution.side.value}`",
            f"- quantity: `{execution.filled_quantity}`",
            f"- fill_price: `{execution.fill_price}`",
            f"- mode: `{execution.execution_mode}`",
            f"- order_id: `{execution.order_id}`",
            f"- execution_id: `{execution.execution_id}`",
            f"- balance: `{balance_text}`",
            f"- NAV: `{nav_text}`",
            f"- time: `{execution.timestamp}`",
        ]
        if balance_error:
            lines.append(f"- balance_error: `{balance_error}`")
        return "\n".join(lines)

    def _classify_event(self, execution: Execution) -> str:
        metadata = execution.metadata if isinstance(execution.metadata, dict) else {}
        raw_fill = metadata.get("raw_fill_transaction", {})
        if not isinstance(raw_fill, dict):
            raw_fill = {}

        opened = bool(raw_fill.get("tradeOpened"))
        closed = bool(raw_fill.get("tradeReduced")) or bool(raw_fill.get("tradesClosed"))

        if opened and closed:
            return "ORDER PLACED + ORDER CLOSED"
        if opened:
            return "ORDER PLACED"
        if closed:
            return "ORDER CLOSED"
        return "ORDER FILLED"

    def _fetch_balance_and_nav(self) -> Tuple[str, str, Optional[str]]:
        try:
            request = AccountSummary(accountID=self.oanda_account_id)
            response = self.oanda_api.request(request)
            account = response.get("account", {}) if isinstance(response, dict) else {}
            return str(account.get("balance", "unknown")), str(account.get("NAV", "unknown")), None
        except Exception as e:
            return "unavailable", "unavailable", f"{type(e).__name__}: {e}"

    async def _send_discord_message(self, content: str) -> None:
        text = content[:1900]
        url = f"https://discord.com/api/v10/channels/{self.discord_channel_id}/messages"
        payload = {"content": text}
        async with self.http.post(url, headers=self._http_headers, json=payload) as resp:
            if resp.status >= 300:
                body = await resp.text()
                raise RuntimeError(f"Discord post failed: {resp.status} {body}")


async def amain() -> None:
    args = parse_args()
    config = Config.load()
    bot = DiscordExecutionNotifierBot(config, args)
    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(amain())
