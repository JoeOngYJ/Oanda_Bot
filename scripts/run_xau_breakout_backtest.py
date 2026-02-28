#!/usr/bin/env python3
"""Run XAU session breakout strategy backtest."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.analysis.metrics import expectancy_per_trade, profit_factor
from backtesting.core.backtester import Backtester
from backtesting.core.timeframe import Timeframe
from backtesting.strategy.examples.xau_session_breakout import XAUSessionBreakout


def parse_args():
    p = argparse.ArgumentParser(description="Run XAU breakout-only backtest.")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--exec-tf", default="M15")
    p.add_argument("--bias-tf", default="H1")
    p.add_argument("--alt-bias-tf", default="H4")
    p.add_argument("--fill-mode", default="next_open", choices=["touch", "next_open"])
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-trades-per-day", type=int, default=2)
    p.add_argument("--max-consecutive-losses", type=int, default=2)
    p.add_argument("--min-rr", type=float, default=2.0)
    p.add_argument("--adx-min", type=float, default=25.0)
    p.add_argument("--max-spread-pips", type=float, default=35.0)
    p.add_argument("--output-dir", default="data/research")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exec_tf = Timeframe.from_oanda_granularity(args.exec_tf)
    cfg = {
        "data": {
            "instrument": args.instrument,
            "base_timeframe": exec_tf,
            "start_date": dt.datetime.fromisoformat(args.start),
            "end_date": dt.datetime.fromisoformat(args.end),
        },
        "strategy": {
            "name": "XAU_Session_Breakout",
            "class": XAUSessionBreakout,
            "timeframes": [exec_tf, Timeframe.from_oanda_granularity(args.bias_tf), Timeframe.from_oanda_granularity(args.alt_bias_tf)],
            "exec_tf": args.exec_tf,
            "bias_tf": args.bias_tf,
            "alt_bias_tf": args.alt_bias_tf,
            "risk_per_trade": args.risk_per_trade,
            "account_equity": args.initial_capital,
            "max_trades_per_day": args.max_trades_per_day,
            "max_consecutive_losses": args.max_consecutive_losses,
            "min_rr": args.min_rr,
            "adx_trade_min": args.adx_min,
            "max_spread_pips": args.max_spread_pips,
        },
        "execution": {
            "initial_capital": args.initial_capital,
            "fill_mode": args.fill_mode,
            "slippage_pips": 0.2,
            "pricing_model": "oanda_core",
            "spreads_pips": {
                "XAU_USD": 20.0,
                "EUR_USD": 1.4,
                "GBP_USD": 2.0,
                "USD_JPY": 1.4,
            },
            "core_commission_per_10k_units": 1.0,
        },
    }

    res = Backtester(context=cfg).run()
    net = float(res.final_equity - args.initial_capital)
    pf = float(profit_factor(res.trades))
    exp = float(expectancy_per_trade(res.trades))

    print(f"Trades: {res.total_trades}")
    print(f"Final equity: {res.final_equity:.2f}")
    print(f"Net PnL: {net:.2f}")
    print(f"Return: {(net / args.initial_capital):.2%}")
    print(f"Win rate: {res.win_rate:.2%}")
    print(f"Profit factor: {pf:.3f}")
    print(f"Expectancy/trade: {exp:.4f}")
    print(f"Sharpe: {res.sharpe_ratio:.4f}")
    print(f"Max drawdown: {res.max_drawdown:.2%}")
    print(f"Fees: {res.total_fees_paid:.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"xau_breakout_run_{stamp}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instrument",
                "start",
                "end",
                "trades",
                "final_equity",
                "net_pnl",
                "return_pct",
                "win_rate",
                "profit_factor",
                "expectancy",
                "sharpe",
                "max_drawdown",
                "fees",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "instrument": args.instrument,
                "start": args.start,
                "end": args.end,
                "trades": res.total_trades,
                "final_equity": round(float(res.final_equity), 6),
                "net_pnl": round(net, 6),
                "return_pct": round((net / args.initial_capital) * 100.0, 6),
                "win_rate": round(float(res.win_rate), 6),
                "profit_factor": round(pf, 6),
                "expectancy": round(exp, 6),
                "sharpe": round(float(res.sharpe_ratio), 6),
                "max_drawdown": round(float(res.max_drawdown), 6),
                "fees": round(float(res.total_fees_paid), 6),
            }
        )
    print(f"Summary CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
