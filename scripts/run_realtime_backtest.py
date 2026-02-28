#!/usr/bin/env python3
"""Run the real-time style Backtester pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.core.backtester import Backtester
from backtesting.core.timeframe import Timeframe
from backtesting.strategy.examples.mean_reversion import MeanReversion


def parse_args():
    p = argparse.ArgumentParser(description="Run real-time style backtest pipeline.")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--tf", default="M15")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--fill-mode", default="next_open", choices=["touch", "next_open"])
    p.add_argument("--pricing-model", default="oanda_core", choices=["spread_only", "oanda_core"])
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--state-snapshot-path", default="", help="Optional JSONL path for SystemState snapshots.")
    p.add_argument("--snapshot-every-bars", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)

    cfg = {
        "data": {
            "instrument": args.instrument,
            "base_timeframe": tf,
            "start_date": dt.datetime.fromisoformat(args.start),
            "end_date": dt.datetime.fromisoformat(args.end),
        },
        "strategy": {
            "name": "MeanRev_RT",
            "class": MeanReversion,
            "timeframes": [tf],
            "sma_period": 20,
            "deviation_pct": 0.002,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": 10000,
        },
        "execution": {
            "initial_capital": float(args.initial_capital),
            "fill_mode": args.fill_mode,
            "slippage_pips": 0.2,
            "pricing_model": args.pricing_model,
            "spreads_pips": {
                "EUR_USD": 1.4,
                "GBP_USD": 2.0,
                "USD_JPY": 1.4,
                "XAU_USD": 20.0,
            },
            "core_commission_per_10k_units": 1.0,
        },
        "state": {
            "snapshot_path": args.state_snapshot_path or None,
            "snapshot_every_bars": int(max(args.snapshot_every_bars, 1)),
        },
    }

    result = Backtester(context=cfg).run()
    print(f"Trades: {result.total_trades}")
    print(f"Final equity: {result.final_equity:.2f}")
    print(f"Net PnL: {result.final_equity - args.initial_capital:.2f}")
    print(f"Win rate: {result.win_rate:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Fees: {result.total_fees_paid:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
