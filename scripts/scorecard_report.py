#!/usr/bin/env python3
"""Aggregate weekly strategy scorecards into a portfolio KPI report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate scorecards into a portfolio KPI report."
    )
    parser.add_argument(
        "--input-dir",
        default="data/scorecards",
        help="Directory containing weekly scorecard CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports",
        help="Directory where report files will be written.",
    )
    parser.add_argument(
        "--week",
        default="",
        help="Optional week filter in format YYYY-Www.",
    )
    parser.add_argument(
        "--prefix",
        default="portfolio_kpi_report",
        help="Output file prefix.",
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_rows(input_dir: Path, week: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(input_dir.rglob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if week and row.get("review_week", "") != week:
                    continue
                row["_source_file"] = str(path)
                rows.append(row)
    return rows


def aggregate_weekly(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_week: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "strategies": 0,
            "trades": 0.0,
            "net_pnl": 0.0,
            "weighted_win_num": 0.0,
            "weighted_slippage_num": 0.0,
            "weighted_reject_num": 0.0,
            "max_drawdown_pct": 0.0,
        }
    )
    seen_strategy_per_week = set()

    for row in rows:
        week = row.get("review_week", "unknown")
        strategy = row.get("strategy_name", "unknown")
        key = (week, strategy)
        trades = max(to_float(row.get("trades", "0")), 0.0)
        net_pnl = to_float(row.get("net_pnl", "0"))
        win_rate = to_float(row.get("win_rate", "0"))
        slippage = to_float(row.get("avg_slippage_pips", "0"))
        reject_rate = to_float(row.get("order_rejection_rate_pct", "0"))
        dd = to_float(row.get("max_drawdown_pct", "0"))

        agg = by_week[week]
        if key not in seen_strategy_per_week:
            agg["strategies"] += 1
            seen_strategy_per_week.add(key)

        agg["trades"] += trades
        agg["net_pnl"] += net_pnl
        agg["weighted_win_num"] += win_rate * trades
        agg["weighted_slippage_num"] += slippage * trades
        agg["weighted_reject_num"] += reject_rate * trades
        agg["max_drawdown_pct"] = max(agg["max_drawdown_pct"], dd)

    result: List[Dict[str, object]] = []
    for week in sorted(by_week.keys()):
        agg = by_week[week]
        trades = agg["trades"]
        expectancy = (agg["net_pnl"] / trades) if trades > 0 else 0.0
        avg_win = (agg["weighted_win_num"] / trades) if trades > 0 else 0.0
        avg_slippage = (agg["weighted_slippage_num"] / trades) if trades > 0 else 0.0
        avg_reject = (agg["weighted_reject_num"] / trades) if trades > 0 else 0.0
        result.append(
            {
                "review_week": week,
                "strategy_count": int(agg["strategies"]),
                "total_trades": int(trades),
                "total_net_pnl": round(agg["net_pnl"], 6),
                "expectancy_per_trade": round(expectancy, 6),
                "weighted_win_rate": round(avg_win, 6),
                "weighted_slippage_pips": round(avg_slippage, 6),
                "weighted_rejection_rate_pct": round(avg_reject, 6),
                "max_drawdown_pct": round(agg["max_drawdown_pct"], 6),
            }
        )
    return result


def latest_rows_by_strategy(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        strategy = row.get("strategy_name", "unknown")
        week = row.get("review_week", "")
        if strategy not in latest or week > latest[strategy].get("review_week", ""):
            latest[strategy] = row
    return [latest[k] for k in sorted(latest.keys())]


def write_weekly_csv(path: Path, weekly: List[Dict[str, object]]) -> None:
    fieldnames = [
        "review_week",
        "strategy_count",
        "total_trades",
        "total_net_pnl",
        "expectancy_per_trade",
        "weighted_win_rate",
        "weighted_slippage_pips",
        "weighted_rejection_rate_pct",
        "max_drawdown_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in weekly:
            writer.writerow(row)


def write_markdown(
    path: Path,
    rows: List[Dict[str, str]],
    weekly: List[Dict[str, object]],
    latest_rows: List[Dict[str, str]],
) -> None:
    total_trades = sum(to_int(r.get("trades", "0")) for r in rows)
    total_net_pnl = sum(to_float(r.get("net_pnl", "0")) for r in rows)
    expectancy = (total_net_pnl / total_trades) if total_trades else 0.0
    decision_counts = Counter(r.get("decision", "").strip() or "unknown" for r in rows)

    lines: List[str] = []
    lines.append("# Portfolio KPI Report\n")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
    lines.append("## Portfolio Summary\n")
    lines.append(f"- Scorecards processed: {len(rows)}")
    lines.append(f"- Total trades: {total_trades}")
    lines.append(f"- Total net PnL: {total_net_pnl:.6f}")
    lines.append(f"- Portfolio expectancy per trade: {expectancy:.6f}\n")

    lines.append("## Decision Mix\n")
    for decision, count in sorted(decision_counts.items()):
        lines.append(f"- {decision}: {count}")
    lines.append("")

    lines.append("## Weekly Aggregates\n")
    if weekly:
        lines.append("| Week | Strategies | Trades | Net PnL | Expectancy | Win Rate | Slippage (pips) | Reject Rate (%) | Max DD (%) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for w in weekly:
            lines.append(
                "| {review_week} | {strategy_count} | {total_trades} | {total_net_pnl} | {expectancy_per_trade} | {weighted_win_rate} | {weighted_slippage_pips} | {weighted_rejection_rate_pct} | {max_drawdown_pct} |".format(
                    **w
                )
            )
    else:
        lines.append("No weekly aggregates available.")
    lines.append("")

    lines.append("## Latest Strategy Rows\n")
    if latest_rows:
        lines.append("| Strategy | Week | Env | Trades | Net PnL | Expectancy | Win Rate | Decision |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for r in latest_rows:
            lines.append(
                f"| {r.get('strategy_name','')} | {r.get('review_week','')} | {r.get('environment','')} | "
                f"{r.get('trades','0')} | {r.get('net_pnl','0')} | {r.get('expectancy_per_trade','0')} | "
                f"{r.get('win_rate','0')} | {r.get('decision','')} |"
            )
    else:
        lines.append("No strategy rows found.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_dir = (repo_root / args.input_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rows = read_rows(input_dir, week=args.week)
    if not rows:
        raise SystemExit(f"No scorecard rows found in {input_dir}")

    weekly = aggregate_weekly(rows)
    latest = latest_rows_by_strategy(rows)

    stamp = args.week or dt.date.today().isoformat()
    base = f"{args.prefix}_{stamp}"
    md_path = output_dir / f"{base}.md"
    csv_path = output_dir / f"{base}.csv"

    write_markdown(md_path, rows, weekly, latest)
    write_weekly_csv(csv_path, weekly)

    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

