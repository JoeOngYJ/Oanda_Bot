"""Report generation helpers for strategy research outputs."""

from __future__ import annotations

from typing import Dict, List


def build_report(results: List[Dict]) -> str:
    """Build a markdown leaderboard report from strategy result rows."""
    if not results:
        return "# Strategy Research Report\n\nNo results.\n"

    lines: List[str] = []
    lines.append("# Strategy Research Report\n")
    lines.append("| Rank | Strategy | Params | Trades | Net PnL | Expectancy | Win Rate | Profit Factor | Max DD |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")

    for idx, row in enumerate(results, 1):
        lines.append(
            "| {rank} | {strategy_name} | `{params}` | {total_trades} | {net_pnl:.2f} | "
            "{expectancy:.4f} | {win_rate:.2%} | {profit_factor:.3f} | {max_drawdown:.2%} |".format(
                rank=idx,
                strategy_name=row.get("strategy_name", ""),
                params=row.get("params", ""),
                total_trades=int(row.get("total_trades", 0)),
                net_pnl=float(row.get("net_pnl", 0.0)),
                expectancy=float(row.get("expectancy", 0.0)),
                win_rate=float(row.get("win_rate", 0.0)),
                profit_factor=float(row.get("profit_factor", 0.0)),
                max_drawdown=float(row.get("max_drawdown", 0.0)),
            )
        )

    return "\n".join(lines) + "\n"
