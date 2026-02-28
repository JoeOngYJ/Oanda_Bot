#!/usr/bin/env python3
"""Persist execution events into daily and monthly trading logs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict

from shared.config import Config
from shared.message_bus import MessageBus
from shared.models import Execution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading journal agent")
    p.add_argument(
        "--output-dir",
        default="data/reports/trading_journal",
        help="Directory for daily/monthly CSV and JSON summary logs",
    )
    return p.parse_args()


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _read_summary(path: Path) -> Dict:
    if not path.exists():
        return {
            "executions": 0,
            "total_quantity": 0,
            "total_notional": 0.0,
            "total_commission": 0.0,
            "by_instrument": {},
            "by_mode": {},
            "updated_at": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "executions": 0,
            "total_quantity": 0,
            "total_notional": 0.0,
            "total_commission": 0.0,
            "by_instrument": {},
            "by_mode": {},
            "updated_at": None,
        }


def _write_summary(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


class TradingJournalAgent:
    def __init__(self, config: Config, output_dir: str):
        self.config = config
        self.bus = MessageBus(config)
        self.output_dir = Path(output_dir).resolve()
        self.running = False

    async def start(self) -> None:
        self.running = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        await self.bus.connect()
        async for message in self.bus.subscribe("executions"):
            if not self.running:
                break
            try:
                execution = Execution(**message)
            except Exception:
                continue
            self._persist_execution(execution)

    async def stop(self) -> None:
        self.running = False
        await self.bus.disconnect()

    def _persist_execution(self, execution: Execution) -> None:
        ts = _as_utc(execution.timestamp)
        day = ts.strftime("%Y-%m-%d")
        month = ts.strftime("%Y-%m")

        row = {
            "timestamp_utc": ts.isoformat(),
            "execution_id": execution.execution_id,
            "order_id": execution.order_id,
            "instrument": execution.instrument.value,
            "side": execution.side.value,
            "filled_quantity": int(execution.filled_quantity),
            "fill_price": str(execution.fill_price),
            "notional": str(Decimal(execution.filled_quantity) * Decimal(execution.fill_price)),
            "commission": str(execution.commission),
            "execution_mode": execution.execution_mode,
            "oanda_transaction_id": execution.oanda_transaction_id,
        }

        daily_csv = self.output_dir / f"executions_daily_{day}.csv"
        monthly_csv = self.output_dir / f"executions_monthly_{month}.csv"
        self._append_csv(daily_csv, row)
        self._append_csv(monthly_csv, row)

        self._update_summary(self.output_dir / f"summary_daily_{day}.json", execution, ts)
        self._update_summary(self.output_dir / f"summary_monthly_{month}.json", execution, ts)

    @staticmethod
    def _append_csv(path: Path, row: Dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _update_summary(path: Path, execution: Execution, ts: datetime) -> None:
        summary = _read_summary(path)
        instrument = execution.instrument.value
        mode = str(execution.execution_mode)
        qty = int(execution.filled_quantity)
        notional = float(Decimal(qty) * Decimal(execution.fill_price))
        commission = float(execution.commission)

        summary["executions"] = int(summary.get("executions", 0)) + 1
        summary["total_quantity"] = int(summary.get("total_quantity", 0)) + qty
        summary["total_notional"] = float(summary.get("total_notional", 0.0)) + notional
        summary["total_commission"] = float(summary.get("total_commission", 0.0)) + commission

        by_inst = defaultdict(lambda: {"executions": 0, "quantity": 0, "notional": 0.0, "commission": 0.0})
        by_inst.update(summary.get("by_instrument", {}))
        by_inst[instrument]["executions"] += 1
        by_inst[instrument]["quantity"] += qty
        by_inst[instrument]["notional"] += notional
        by_inst[instrument]["commission"] += commission
        summary["by_instrument"] = dict(by_inst)

        by_mode = defaultdict(int)
        by_mode.update(summary.get("by_mode", {}))
        by_mode[mode] += 1
        summary["by_mode"] = dict(by_mode)

        summary["updated_at"] = ts.isoformat()
        _write_summary(path, summary)


async def amain() -> None:
    args = parse_args()
    config = Config.load()
    agent = TradingJournalAgent(config=config, output_dir=args.output_dir)
    try:
        await agent.start()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(amain())
