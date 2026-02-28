"""Performance metrics for backtest result evaluation."""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List


def sharpe(returns: Iterable[float], risk_free: float = 0.0) -> float:
    values = list(returns)
    if len(values) < 2:
        return 0.0
    excess = [r - risk_free for r in values]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    if variance <= 0:
        return 0.0
    return mean / sqrt(variance)


def win_rate(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)
    return wins / len(trades)


def profit_factor(trades: List[Dict]) -> float:
    gross_profit = sum(float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) > 0)
    gross_loss = -sum(float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy_per_trade(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    return sum(float(t.get("pnl", 0.0)) for t in trades) / len(trades)


def max_drawdown_from_equity(equity_curve: List[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd
