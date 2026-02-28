#!/usr/bin/env python3
"""Discord operator bot for execution controls, status, and profit reporting."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiohttp
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountSummary

from shared.config import Config
from shared.message_bus import MessageBus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discord operator bot")
    p.add_argument("--poll-seconds", type=float, default=3.0, help="Discord command polling interval")
    p.add_argument(
        "--alert-min-severity",
        choices=["info", "warning", "critical"],
        default="warning",
        help="Minimum alert severity to forward to Discord",
    )
    p.add_argument("--commands-prefix", default="!", help="Command prefix, e.g. !status")
    return p.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sev_rank(sev: str) -> int:
    table = {"info": 0, "warning": 1, "critical": 2}
    return table.get(str(sev).lower(), 0)


def _log(msg: str) -> None:
    print(f"[discord-bot] {datetime.now(timezone.utc).isoformat()} {msg}", flush=True)


class DiscordOperatorBot:
    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.message_bus = MessageBus(config)
        self.running = False
        self.last_message_id: Optional[str] = None

        self.discord_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
        self.discord_channel_id = os.getenv("DISCORD_CHANNEL_ID", "").strip()
        if not self.discord_token or not self.discord_channel_id:
            raise SystemExit("Set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID")

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
            await self._initialize_cursor()
            _log(f"initialized cursor last_message_id={self.last_message_id}")
            await self._send_discord_message(
                f"Discord operator bot online at `{_now_iso()}`. Type `{self.args.commands_prefix}help`."
            )
            _log("online message sent")
            await asyncio.gather(
                self._command_loop(),
                self._alert_loop(),
            )

    async def stop(self) -> None:
        self.running = False
        await self.message_bus.disconnect()

    async def _initialize_cursor(self) -> None:
        url = f"https://discord.com/api/v10/channels/{self.discord_channel_id}/messages?limit=1"
        async with self.http.get(url, headers=self._http_headers) as resp:
            data = await resp.json()
            if resp.status >= 300:
                raise RuntimeError(f"Discord init failed: {resp.status} {data}")
            if isinstance(data, list) and data:
                self.last_message_id = str(data[0].get("id"))

    async def _command_loop(self) -> None:
        while self.running:
            try:
                msgs = await self._fetch_new_messages()
                if msgs:
                    _log(f"fetched {len(msgs)} new messages")
                for msg in msgs:
                    await self._process_discord_message(msg)
            except Exception as e:
                _log(f"command loop error: {type(e).__name__}: {e}")
                await self._send_discord_message(f"Command loop error: `{type(e).__name__}: {e}`")
            await asyncio.sleep(max(self.args.poll_seconds, 1.0))

    async def _alert_loop(self) -> None:
        min_rank = _sev_rank(self.args.alert_min_severity)
        async for message in self.message_bus.subscribe("alerts"):
            if not self.running:
                break
            try:
                severity = str(message.get("severity", "info")).lower()
                if _sev_rank(severity) < min_rank:
                    continue
                component = str(message.get("component", "unknown"))
                text = str(message.get("message", ""))
                await self._send_discord_message(f"[ALERT:{severity.upper()}] {component}: {text}")
            except Exception:
                # Keep alert forwarding loop alive even if Discord post fails.
                continue

    async def _fetch_new_messages(self) -> list[Dict[str, Any]]:
        base = f"https://discord.com/api/v10/channels/{self.discord_channel_id}/messages?limit=50"
        if self.last_message_id:
            base += f"&after={self.last_message_id}"
        async with self.http.get(base, headers=self._http_headers) as resp:
            data = await resp.json()
            if resp.status >= 300:
                raise RuntimeError(f"Discord fetch failed: {resp.status} {data}")
            if not isinstance(data, list):
                return []
            data.sort(key=lambda m: int(m.get("id", "0")))
            if data:
                self.last_message_id = str(data[-1].get("id"))
            return data

    async def _process_discord_message(self, msg: Dict[str, Any]) -> None:
        author = msg.get("author", {}) or {}
        if bool(author.get("bot", False)):
            return
        content = str(msg.get("content", "")).strip()
        if not content.startswith(self.args.commands_prefix):
            return

        cmdline = content[len(self.args.commands_prefix):].strip()
        parts = [p for p in cmdline.split() if p]
        if not parts:
            return
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in {"help", "commands"}:
            _log("processing !help")
            await self._send_discord_message(
                "Commands: `!status`, `!profit`, `!report`, `!journal daily|monthly`, `!session start [shadow|live]`, `!session stop`, `!kill on|off`, `!shadow on|off`, `!app start [shadow|live]`, `!app stop`, `!sleep 60m`, `!wake`, `!streams`"
            )
            return

        if cmd == "report":
            text = await self._build_report_text()
            await self._send_discord_message(text)
            return

        if cmd == "journal":
            if not args or args[0].lower() not in {"daily", "monthly"}:
                await self._send_discord_message("Usage: `!journal daily` or `!journal monthly`")
                return
            text = await self._build_journal_text(args[0].lower())
            await self._send_discord_message(text)
            return

        if cmd == "status":
            text = await self._build_status_text()
            await self._send_discord_message(text)
            return

        if cmd in {"profit", "pnl"}:
            text = await self._build_profit_text()
            await self._send_discord_message(text)
            return

        if cmd == "app":
            if not args or args[0].lower() not in {"start", "stop"}:
                await self._send_discord_message("Usage: `!app start [shadow|live]` or `!app stop`")
                return
            if args[0].lower() == "start":
                mode = args[1].lower() if len(args) > 1 else "shadow"
                if mode not in {"shadow", "live"}:
                    mode = "shadow"
                await self._publish_ops("app_start", mode=mode)
                await self._send_discord_message(f"App start requested. mode=`{mode}`")
            else:
                await self._publish_ops("app_stop")
                await self._send_discord_message("App stop requested.")
            return

        if cmd == "sleep":
            if not args:
                await self._send_discord_message("Usage: `!sleep 60m` or `!sleep 2h`")
                return
            minutes = self._parse_duration_to_minutes(args[0])
            if minutes is None or minutes <= 0:
                await self._send_discord_message("Invalid duration. Use `30m` or `2h`.")
                return
            await self._publish_ops("sleep", minutes=minutes)
            await self._send_discord_message(f"Sleep requested for `{minutes}` minutes.")
            return

        if cmd == "wake":
            await self._publish_ops("wake")
            await self._send_discord_message("Wake requested.")
            return

        if cmd == "streams":
            text = await self._build_streams_text()
            await self._send_discord_message(text)
            return

        if cmd == "session":
            if not args or args[0].lower() not in {"start", "stop"}:
                await self._send_discord_message("Usage: `!session start [shadow|live]` or `!session stop`")
                return
            if args[0].lower() == "start":
                mode = args[1].lower() if len(args) > 1 else ""
                if mode == "shadow":
                    await self._publish_control("shadow_mode_on", "discord session start shadow")
                elif mode == "live":
                    await self._publish_control("shadow_mode_off", "discord session start live")
                await self._publish_control("kill_switch_off", "discord session start")
                await self._send_discord_message("Session start requested: `kill_switch_off` published.")
            else:
                await self._publish_control("kill_switch_on", "discord session stop")
                await self._send_discord_message("Session stop requested: `kill_switch_on` published.")
            return

        if cmd in {"kill", "shadow"}:
            if not args or args[0].lower() not in {"on", "off"}:
                await self._send_discord_message(f"Usage: `!{cmd} on` or `!{cmd} off`")
                return
            mode = args[0].lower()
            action = f"{cmd}_mode_{mode}" if cmd == "shadow" else f"kill_switch_{mode}"
            await self._publish_control(action, f"discord {cmd} {mode}")
            await self._send_discord_message(f"Published: `{action}`")
            return

        await self._send_discord_message("Unknown command. Use `!help`.")

    async def _publish_control(self, action: str, reason: str) -> None:
        await self.message_bus.publish(
            "execution_control",
            {
                "action": action,
                "reason": reason,
                "requested_by": "discord_operator_bot",
            },
        )

    async def _publish_ops(self, action: str, **kwargs) -> None:
        payload = {"action": action, "requested_by": "discord_operator_bot", **kwargs}
        await self.message_bus.publish("ops_control", payload)

    async def _build_status_text(self) -> str:
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            return "Status unavailable: Redis disconnected."

        kill_switch = _as_bool(await redis_client.get("execution:state:kill_switch"))
        shadow_mode = _as_bool(await redis_client.get("execution:state:shadow_mode"))
        guardrail = _as_bool(await redis_client.get("execution:state:live_guardrail_blocked"))
        updated_at = await redis_client.get("execution:state:updated_at")
        streams = await self._stream_lengths()
        supervisor = await self._supervisor_state()

        return (
            "Execution status\n"
            f"- kill_switch: `{kill_switch}`\n"
            f"- shadow_mode: `{shadow_mode}`\n"
            f"- live_guardrail_blocked: `{guardrail}`\n"
            f"- updated_at: `{updated_at}`\n"
            f"- streams: `{streams}`\n"
            f"- supervisor: `{supervisor}`"
        )

    async def _build_streams_text(self) -> str:
        streams = await self._stream_lengths()
        return f"Stream lengths: `{streams}`"

    async def _build_report_text(self) -> str:
        status = await self._build_status_text()
        pnl = await self._build_profit_text()
        return f"{status}\n\n{pnl}"

    async def _build_journal_text(self, period: str) -> str:
        summary = self._read_latest_journal_summary(period)
        if summary is None:
            return f"No {period} journal summary found yet."
        return (
            f"Journal {period} summary\n"
            f"- executions: `{summary.get('executions')}`\n"
            f"- total_quantity: `{summary.get('total_quantity')}`\n"
            f"- total_notional: `{summary.get('total_notional')}`\n"
            f"- total_commission: `{summary.get('total_commission')}`\n"
            f"- by_mode: `{summary.get('by_mode')}`\n"
            f"- updated_at: `{summary.get('updated_at')}`"
        )

    async def _stream_lengths(self) -> Dict[str, int]:
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            return {}
        keys = {
            "market_data": "stream:market_data",
            "signals": "stream:signals",
            "risk_checks": "stream:risk_checks",
            "orders": "stream:orders",
            "executions": "stream:executions",
            "alerts": "stream:alerts",
        }
        out: Dict[str, int] = {}
        for k, stream_key in keys.items():
            try:
                out[k] = int(await redis_client.xlen(stream_key))
            except Exception:
                out[k] = -1
        return out

    async def _supervisor_state(self) -> Dict[str, Any]:
        redis_client = self.message_bus.redis_client
        if redis_client is None:
            return {}
        names = ["market_data", "risk", "execution", "strategy_regime", "monitoring", "journal"]
        procs: Dict[str, str] = {}
        for n in names:
            procs[n] = str(await redis_client.get(f"ops:supervisor:proc_status:{n}") or "stopped")
        return {
            "running": _as_bool(await redis_client.get("ops:supervisor:running")),
            "sleep_until": await redis_client.get("ops:supervisor:sleep_until"),
            "procs": procs,
        }

    def _read_latest_journal_summary(self, period: str) -> Optional[Dict[str, Any]]:
        period = str(period).lower()
        if period not in {"daily", "monthly"}:
            return None
        root = Path(
            os.getenv(
                "TRADING_JOURNAL_OUTPUT_DIR",
                "data/reports/trading_journal",
            )
        ).resolve()
        pattern = f"summary_{period}_*.json"
        files = sorted(root.glob(pattern))
        if not files:
            return None
        latest = files[-1]
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None

    async def _build_profit_text(self) -> str:
        summary, err = await asyncio.get_running_loop().run_in_executor(None, self._fetch_oanda_summary)
        if err:
            return f"Profit query failed: `{err}`"
        account = summary.get("account", {}) if isinstance(summary, dict) else {}
        return (
            "OANDA account summary\n"
            f"- balance: `{account.get('balance')}`\n"
            f"- NAV: `{account.get('NAV')}`\n"
            f"- unrealizedPL: `{account.get('unrealizedPL')}`\n"
            f"- realizedPL: `{account.get('pl')}`\n"
            f"- financing: `{account.get('financing')}`\n"
            f"- commission: `{account.get('commission')}`"
        )

    @staticmethod
    def _parse_duration_to_minutes(text: str) -> Optional[int]:
        s = str(text).strip().lower()
        if s.endswith("m"):
            try:
                return int(float(s[:-1]))
            except Exception:
                return None
        if s.endswith("h"):
            try:
                return int(float(s[:-1]) * 60)
            except Exception:
                return None
        try:
            return int(float(s))
        except Exception:
            return None

    def _fetch_oanda_summary(self) -> Tuple[Dict[str, Any], Optional[str]]:
        try:
            request = AccountSummary(accountID=self.oanda_account_id)
            resp = self.oanda_api.request(request)
            return resp, None
        except Exception as e:
            return {}, f"{type(e).__name__}: {e}"

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
    bot = DiscordOperatorBot(config, args)
    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(amain())
