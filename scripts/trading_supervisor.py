#!/usr/bin/env python3
"""Process supervisor for trading agents, controlled through Redis ops commands."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from shared.config import Config
from shared.message_bus import MessageBus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading process supervisor")
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    p.add_argument("--poll-seconds", type=float, default=2.0)
    return p.parse_args()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ProcSpec:
    name: str
    cmd: list[str]


class TradingSupervisor:
    def __init__(self, config: Config, project_root: str, poll_seconds: float):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.poll_seconds = max(float(poll_seconds), 1.0)
        self.bus = MessageBus(config)
        self.procs: Dict[str, subprocess.Popen] = {}
        self.sleep_until: Optional[datetime] = None
        self.running = False

    async def start(self) -> None:
        self.running = True
        await self.bus.connect()
        await self._set_state("ops:supervisor:running", "1")
        await self._set_state("ops:supervisor:started_at", _iso(_now()))
        await asyncio.gather(
            self._ops_loop(),
            self._watchdog_loop(),
        )

    async def stop(self) -> None:
        self.running = False
        await self._stop_stack(reason="supervisor_shutdown")
        await self._set_state("ops:supervisor:running", "0")
        await self.bus.disconnect()

    async def _ops_loop(self) -> None:
        async for msg in self.bus.subscribe("ops_control"):
            if not self.running:
                break
            action = str(msg.get("action", "")).strip().lower()
            if action == "app_start":
                mode = str(msg.get("mode", "shadow")).strip().lower()
                await self._start_stack(mode=mode)
            elif action == "app_stop":
                await self._stop_stack(reason="ops_command")
            elif action == "sleep":
                minutes = int(msg.get("minutes", 60))
                await self._sleep(minutes)
            elif action == "wake":
                await self._wake()
            elif action == "report":
                await self._publish_ops_event("report_requested")

    async def _watchdog_loop(self) -> None:
        while self.running:
            await self._persist_proc_state()
            if self.sleep_until is not None and _now() >= self.sleep_until:
                self.sleep_until = None
                await self._set_state("ops:supervisor:sleep_until", "")
                await self._publish_control("kill_switch_off", "auto wake by supervisor")
                await self._publish_ops_event("wake_auto")
            await asyncio.sleep(self.poll_seconds)

    def _build_specs(self, mode: str) -> Dict[str, ProcSpec]:
        py = str((self.project_root / ".venv" / "bin" / "python").resolve())
        model_json = os.getenv("REGIME_MODEL_JSON", "").strip()
        if not model_json:
            model_json = self._latest_runtime_model_json()
        if not model_json:
            raise RuntimeError("REGIME_MODEL_JSON not set and no multiframe_regime_model_*.json found")

        strategy_cmd = [
            py,
            "-m",
            "agents.strategy.regime_runtime_agent",
            "--model-json",
            model_json,
            "--instrument",
            os.getenv("REGIME_INSTRUMENT", "XAU_USD"),
            "--decision-mode",
            os.getenv("REGIME_DECISION_MODE", "ensemble"),
            "--quantity",
            os.getenv("REGIME_QUANTITY", "3"),
            "--gpu",
            os.getenv("REGIME_GPU", "auto"),
            "--warmup",
            os.getenv("REGIME_WARMUP", "on"),
            "--warmup-m15-bars",
            os.getenv("REGIME_WARMUP_M15_BARS", "5000"),
            "--warmup-h1-bars",
            os.getenv("REGIME_WARMUP_H1_BARS", "3000"),
            "--warmup-h4-bars",
            os.getenv("REGIME_WARMUP_H4_BARS", "1500"),
            "--warmup-d1-bars",
            os.getenv("REGIME_WARMUP_D1_BARS", "750"),
        ]

        env_mode = mode if mode in {"shadow", "live"} else "shadow"
        exec_env_shadow = "true" if env_mode == "shadow" else "false"
        exec_env_live = "true" if env_mode == "live" else "false"

        return {
            "market_data": ProcSpec("market_data", [py, "-m", "agents.market_data.agent"]),
            "risk": ProcSpec("risk", [py, "-m", "agents.risk.agent"]),
            "execution": ProcSpec(
                "execution",
                [py, "-m", "agents.execution.agent"],
            ),
            "strategy_regime": ProcSpec("strategy_regime", strategy_cmd),
            "monitoring": ProcSpec("monitoring", [py, "-m", "agents.monitoring.agent"]),
            "journal": ProcSpec(
                "journal",
                [
                    py,
                    "scripts/trading_journal_agent.py",
                    "--output-dir",
                    os.getenv("TRADING_JOURNAL_OUTPUT_DIR", "data/reports/trading_journal"),
                ],
            ),
        }, {
            "EXECUTION_SHADOW_MODE": exec_env_shadow,
            "EXECUTION_LIVE_ENABLED": exec_env_live,
        }

    def _latest_runtime_model_json(self) -> str:
        root = self.project_root / "data" / "research"
        files = sorted(root.glob("multiframe_regime_model_*.json"))
        if not files:
            return ""
        return str(files[-1].resolve())

    async def _start_stack(self, mode: str) -> None:
        specs, extra_env = self._build_specs(mode)
        for name in ["market_data", "risk", "execution", "strategy_regime", "monitoring", "journal"]:
            if name in self.procs and self.procs[name].poll() is None:
                continue
            spec = specs[name]
            env = os.environ.copy()
            env.update(extra_env)
            env["PYTHONPATH"] = f"{self.project_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
            proc = subprocess.Popen(
                spec.cmd,
                cwd=str(self.project_root),
                env=env,
                start_new_session=True,
            )
            self.procs[name] = proc
            await self._set_state(f"ops:supervisor:proc:{name}", str(proc.pid))
        await self._publish_ops_event(f"app_started mode={mode}")

    async def _stop_stack(self, reason: str) -> None:
        for name in ["journal", "monitoring", "strategy_regime", "execution", "risk", "market_data"]:
            proc = self.procs.get(name)
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        proc.kill()
            await self._set_state(f"ops:supervisor:proc:{name}", "")
            self.procs.pop(name, None)
        await self._publish_ops_event(f"app_stopped reason={reason}")

    async def _sleep(self, minutes: int) -> None:
        mins = max(int(minutes), 1)
        self.sleep_until = _now() + timedelta(minutes=mins)
        await self._set_state("ops:supervisor:sleep_until", _iso(self.sleep_until))
        await self._publish_control("kill_switch_on", f"sleep for {mins}m")
        await self._publish_ops_event(f"sleep started minutes={mins}")

    async def _wake(self) -> None:
        self.sleep_until = None
        await self._set_state("ops:supervisor:sleep_until", "")
        await self._publish_control("kill_switch_off", "wake requested")
        await self._publish_ops_event("wake requested")

    async def _publish_control(self, action: str, reason: str) -> None:
        await self.bus.publish(
            "execution_control",
            {"action": action, "reason": reason, "requested_by": "trading_supervisor"},
        )

    async def _publish_ops_event(self, message: str) -> None:
        await self.bus.publish(
            "alerts",
            {
                "component": "supervisor",
                "severity": "info",
                "message": message,
                "timestamp": _iso(_now()),
            },
        )

    async def _set_state(self, key: str, value: str) -> None:
        redis_client = self.bus.redis_client
        if redis_client is None:
            return
        await redis_client.set(key, value)

    async def _persist_proc_state(self) -> None:
        redis_client = self.bus.redis_client
        if redis_client is None:
            return
        for name, proc in list(self.procs.items()):
            code = proc.poll()
            if code is None:
                await redis_client.set(f"ops:supervisor:proc_status:{name}", "running")
            else:
                await redis_client.set(f"ops:supervisor:proc_status:{name}", f"exited:{code}")
                self.procs.pop(name, None)


async def amain() -> None:
    args = parse_args()
    config = Config.load()
    supervisor = TradingSupervisor(config=config, project_root=args.project_root, poll_seconds=args.poll_seconds)
    try:
        await supervisor.start()
    except KeyboardInterrupt:
        pass
    finally:
        await supervisor.stop()


if __name__ == "__main__":
    asyncio.run(amain())
