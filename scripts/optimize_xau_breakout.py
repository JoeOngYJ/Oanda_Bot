#!/usr/bin/env python3
"""Grid optimization for XAU breakout strategy with train/test split."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oanda_bot.backtesting.analysis.metrics import expectancy_per_trade, profit_factor
from oanda_bot.backtesting.core.backtester import Backtester
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.strategy.examples.xau_session_breakout import XAUSessionBreakout


def _float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Optimize XAU breakout parameters.")
    p.add_argument("--train-start", default="2023-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--adx-grid", default="23,25,28")
    p.add_argument("--stop-atr-grid", default="1.0,1.2,1.5")
    p.add_argument("--rr-grid", default="2.0,2.5,3.0")
    p.add_argument("--max-spread-grid", default="25,30,35")
    p.add_argument("--output-dir", default="data/research")
    return p.parse_args()


def _run_once(start: str, end: str, params: dict) -> dict:
    tf = Timeframe.M15
    cfg = {
        "data": {
            "instrument": "XAU_USD",
            "base_timeframe": tf,
            "start_date": dt.datetime.fromisoformat(start),
            "end_date": dt.datetime.fromisoformat(end),
        },
        "strategy": {
            "name": "XAU_Session_Breakout",
            "class": XAUSessionBreakout,
            "timeframes": [Timeframe.M15, Timeframe.H1, Timeframe.H4],
            "exec_tf": "M15",
            "bias_tf": "H1",
            "alt_bias_tf": "H4",
            "risk_per_trade": 0.01,
            "account_equity": 10000.0,
            "max_trades_per_day": 2,
            "max_consecutive_losses": 2,
            "adx_trade_min": params["adx_trade_min"],
            "stop_atr_mult": params["stop_atr_mult"],
            "min_rr": params["min_rr"],
            "max_spread_pips": params["max_spread_pips"],
        },
        "execution": {
            "initial_capital": 10000.0,
            "fill_mode": "next_open",
            "slippage_pips": 0.2,
            "pricing_model": "oanda_core",
            "spreads_pips": {"XAU_USD": 20.0},
            "core_commission_per_10k_units": 1.0,
        },
    }
    res = Backtester(context=cfg).run()
    net = float(res.final_equity - 10000.0)
    return {
        "trades": int(res.total_trades),
        "net_pnl": net,
        "return_pct": (net / 10000.0) * 100.0,
        "win_rate": float(res.win_rate),
        "profit_factor": float(profit_factor(res.trades)),
        "expectancy": float(expectancy_per_trade(res.trades)),
        "sharpe": float(res.sharpe_ratio),
        "max_drawdown": float(res.max_drawdown),
        "fees": float(res.total_fees_paid),
    }


def main() -> int:
    args = parse_args()
    adx_grid = _float_list(args.adx_grid)
    stop_grid = _float_list(args.stop_atr_grid)
    rr_grid = _float_list(args.rr_grid)
    spread_grid = _int_list(args.max_spread_grid)

    rows = []
    for adx, stop_atr, rr, max_spread in itertools.product(adx_grid, stop_grid, rr_grid, spread_grid):
        params = {
            "adx_trade_min": adx,
            "stop_atr_mult": stop_atr,
            "min_rr": rr,
            "max_spread_pips": max_spread,
        }
        train = _run_once(args.train_start, args.train_end, params)
        test = _run_once(args.test_start, args.test_end, params)
        rows.append(
            {
                **params,
                "train_net_pnl": train["net_pnl"],
                "train_pf": train["profit_factor"],
                "train_dd": train["max_drawdown"],
                "train_sharpe": train["sharpe"],
                "test_net_pnl": test["net_pnl"],
                "test_pf": test["profit_factor"],
                "test_dd": test["max_drawdown"],
                "test_sharpe": test["sharpe"],
                "test_trades": test["trades"],
                "test_return_pct": test["return_pct"],
            }
        )

    rows.sort(
        key=lambda r: (r["test_net_pnl"], r["test_pf"], -r["test_dd"], r["test_sharpe"]),
        reverse=True,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"xau_breakout_opt_{stamp}.csv"
    md_path = out_dir / f"xau_breakout_opt_{stamp}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# XAU Breakout Optimization",
        "",
        f"Train: {args.train_start} -> {args.train_end}",
        f"Test: {args.test_start} -> {args.test_end}",
        "",
        "| Rank | ADX min | Stop ATR | RR | Max Spread | Test Net | Test Return | Test PF | Test DD | Test Sharpe | Trades |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows[:15], 1):
        lines.append(
            f"| {i} | {r['adx_trade_min']} | {r['stop_atr_mult']} | {r['min_rr']} | {r['max_spread_pips']} | "
            f"{r['test_net_pnl']:.2f} | {r['test_return_pct']:.2f}% | {r['test_pf']:.3f} | {r['test_dd']:.2%} | "
            f"{r['test_sharpe']:.3f} | {r['test_trades']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")
    if rows:
        top = rows[0]
        print(
            "Top test config: "
            f"ADX={top['adx_trade_min']} stop_atr={top['stop_atr_mult']} rr={top['min_rr']} max_spread={top['max_spread_pips']} "
            f"net={top['test_net_pnl']:.2f} dd={top['test_dd']:.2%} pf={top['test_pf']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
