#!/usr/bin/env python3
"""Create a prefilled weekly strategy scorecard CSV.

Output path convention:
    data/scorecards/YYYY/YYYY-Www_<strategy_name>_<env>.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path


HEADER = [
    "review_week",
    "strategy_name",
    "version",
    "environment",
    "reviewer",
    "trades",
    "net_pnl",
    "expectancy_per_trade",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
    "sharpe_like",
    "avg_slippage_pips",
    "slippage_vs_model_delta_pct",
    "order_rejection_rate_pct",
    "signal_to_execution_latency_ms",
    "missing_fills_incidents",
    "daily_loss_limit_breaches",
    "weekly_loss_limit_breaches",
    "circuit_breaker_triggers",
    "risk_control_bypass",
    "contract_schema_errors",
    "gate_a_status",
    "gate_b_status",
    "gate_c_status",
    "decision",
    "size_multiplier",
    "effective_date",
    "rationale",
    "action_item_1",
    "action_item_2",
    "action_item_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a weekly strategy scorecard CSV template."
    )
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["research", "paper", "live_micro", "live_scaled"],
        help="Strategy environment/state",
    )
    parser.add_argument("--reviewer", default="", help="Reviewer name or handle")
    parser.add_argument("--version", default="", help="Strategy version")
    parser.add_argument(
        "--week",
        default="",
        help="Override ISO week in format YYYY-Www (example: 2026-W09)",
    )
    parser.add_argument(
        "--effective-date",
        default="",
        help="Decision effective date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--decision",
        default="hold",
        choices=["promote", "hold", "de-risk", "demote", "halt"],
        help="Initial decision value",
    )
    parser.add_argument(
        "--size-multiplier",
        default="1.0",
        help="Initial size multiplier",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional base output directory (defaults to data/scorecards)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite file if it already exists",
    )
    return parser.parse_args()


def current_week_label() -> str:
    today = dt.date.today()
    iso = today.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def sanitize_name(value: str) -> str:
    out = []
    for ch in value.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "strategy"


def main() -> int:
    args = parse_args()

    week = args.week or current_week_label()
    if not week.startswith("20") or "-W" not in week:
        raise SystemExit(f"Invalid --week format: {week}. Use YYYY-Www")

    year = week.split("-W", 1)[0]
    strategy_file_part = sanitize_name(args.strategy)
    env_file_part = sanitize_name(args.environment)
    filename = f"{week}_{strategy_file_part}_{env_file_part}.csv"

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if args.output_dir:
        base_output = Path(args.output_dir).resolve()
    else:
        base_output = repo_root / "data" / "scorecards"
    output_dir = base_output / year
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists() and not args.force:
        raise SystemExit(
            f"File already exists: {output_path}\nUse --force to overwrite."
        )

    effective_date = args.effective_date or dt.date.today().isoformat()

    row = {
        "review_week": week,
        "strategy_name": args.strategy,
        "version": args.version,
        "environment": args.environment,
        "reviewer": args.reviewer,
        "trades": "0",
        "net_pnl": "0",
        "expectancy_per_trade": "0",
        "win_rate": "0",
        "profit_factor": "0",
        "max_drawdown_pct": "0",
        "sharpe_like": "0",
        "avg_slippage_pips": "0",
        "slippage_vs_model_delta_pct": "0",
        "order_rejection_rate_pct": "0",
        "signal_to_execution_latency_ms": "0",
        "missing_fills_incidents": "0",
        "daily_loss_limit_breaches": "0",
        "weekly_loss_limit_breaches": "0",
        "circuit_breaker_triggers": "0",
        "risk_control_bypass": "no",
        "contract_schema_errors": "no",
        "gate_a_status": "hold",
        "gate_b_status": "hold",
        "gate_c_status": "hold",
        "decision": args.decision,
        "size_multiplier": args.size_multiplier,
        "effective_date": effective_date,
        "rationale": "",
        "action_item_1": "",
        "action_item_2": "",
        "action_item_3": "",
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerow(row)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

