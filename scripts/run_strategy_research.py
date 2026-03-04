#!/usr/bin/env python3
"""Run strategy research grid search with train/test split."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable when script runs directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oanda_bot.backtesting.analysis.metrics import expectancy_per_trade, profit_factor
from oanda_bot.backtesting.analysis.reports import build_report
from oanda_bot.backtesting.core.engine import BacktestEngine
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.strategy.examples.atr_breakout import ATRBreakout
from oanda_bot.backtesting.strategy.examples.breakout import Breakout
from oanda_bot.backtesting.strategy.examples.ema_pullback import EMATrendPullback
from oanda_bot.backtesting.strategy.examples.ensemble_vote import EnsembleVoteStrategy
from oanda_bot.backtesting.strategy.examples.mean_reversion import MeanReversion
from oanda_bot.backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from oanda_bot.backtesting.strategy.examples.volatility_compression_breakout import (
    VolatilityCompressionBreakout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtesting strategy research.")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--tf", default="H1", help="Base timeframe, example: H1, M15")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--pricing-model", default="oanda_core", choices=["spread_only", "oanda_core"])
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument(
        "--demo-bars",
        type=int,
        default=0,
        help="Generate synthetic bars instead of loading historical data.",
    )
    return parser.parse_args()


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _build_candidates(tf: Timeframe) -> List[Tuple[str, Dict]]:
    candidates: List[Tuple[str, Dict]] = []

    for lookback in (20, 40, 60):
        candidates.append(
            (
                "Breakout",
                {
                    "name": f"Breakout_L{lookback}",
                    "class": Breakout,
                    "timeframes": [tf],
                    "lookback": lookback,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.008,
                    "min_breakout_pct": 0.0002,
                    "quantity": 10000,
                },
            )
        )

    for sma_period in (20, 30, 50):
        for dev in (0.0020, 0.0030):
            candidates.append(
                (
                    "MeanReversion",
                    {
                        "name": f"MeanRev_SMA{sma_period}_D{dev}",
                        "class": MeanReversion,
                        "timeframes": [tf],
                        "sma_period": sma_period,
                        "deviation_pct": dev,
                        "stop_loss_pct": 0.004,
                        "take_profit_pct": 0.003,
                        "quantity": 10000,
                    },
                )
            )

    for fast, slow in ((20, 100), (50, 200)):
        for pullback in (0.0006, 0.0010):
            candidates.append(
                (
                    "EMATrendPullback",
                    {
                        "name": f"EMAPullback_{fast}_{slow}_P{pullback}",
                        "class": EMATrendPullback,
                        "timeframes": [tf],
                        "fast_period": fast,
                        "slow_period": slow,
                        "pullback_pct": pullback,
                        "stop_loss_pct": 0.004,
                        "take_profit_pct": 0.009,
                        "quantity": 10000,
                    },
                )
            )

    for lookback in (20, 30):
        for atr_mult in (1.0, 1.4):
            candidates.append(
                (
                    "ATRBreakout",
                    {
                        "name": f"ATRBreakout_L{lookback}_M{atr_mult}",
                        "class": ATRBreakout,
                        "timeframes": [tf],
                        "lookback": lookback,
                        "atr_period": 14,
                        "atr_mult": atr_mult,
                        "stop_loss_pct": 0.0045,
                        "take_profit_pct": 0.009,
                        "quantity": 10000,
                    },
                )
            )

    for std in (1.8, 2.2):
        for rsi_bounds in ((30, 70), (25, 75)):
            low, high = rsi_bounds
            candidates.append(
                (
                    "RSIBollingerReversion",
                    {
                        "name": f"RSIBB_STD{std}_R{low}_{high}",
                        "class": RSIBollingerReversion,
                        "timeframes": [tf],
                        "window": 20,
                        "std_mult": std,
                        "rsi_period": 14,
                        "rsi_oversold": low,
                        "rsi_overbought": high,
                        "stop_loss_pct": 0.004,
                        "take_profit_pct": 0.003,
                        "quantity": 10000,
                    },
                )
            )

    for ratio in (0.7, 0.8):
        candidates.append(
            (
                "VolatilityCompressionBreakout",
                {
                    "name": f"VolCompression_{ratio}",
                    "class": VolatilityCompressionBreakout,
                    "timeframes": [tf],
                    "range_lookback": 20,
                    "atr_period": 14,
                    "compression_window": 40,
                    "compression_ratio": ratio,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.01,
                    "quantity": 10000,
                },
            )
        )

    candidates.append(
        (
            "EnsembleVote",
            {
                "name": "Ensemble_TrendBreakout",
                "class": EnsembleVoteStrategy,
                "timeframes": [tf],
                "min_votes": 2,
                "cooldown_bars": 8,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.008,
                "quantity": 10000,
                "components": [
                    {
                        "class": EMATrendPullback,
                        "timeframes": [tf],
                        "fast_period": 20,
                        "slow_period": 100,
                        "pullback_pct": 0.0008,
                    },
                    {
                        "class": ATRBreakout,
                        "timeframes": [tf],
                        "lookback": 20,
                        "atr_period": 14,
                        "atr_mult": 1.2,
                    },
                    {
                        "class": Breakout,
                        "timeframes": [tf],
                        "lookback": 40,
                    },
                ],
            },
        )
    )

    candidates.append(
        (
            "EnsembleVote",
            {
                "name": "Ensemble_MeanRev",
                "class": EnsembleVoteStrategy,
                "timeframes": [tf],
                "min_votes": 2,
                "cooldown_bars": 6,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.0035,
                "quantity": 10000,
                "components": [
                    {
                        "class": MeanReversion,
                        "timeframes": [tf],
                        "sma_period": 30,
                        "deviation_pct": 0.002,
                    },
                    {
                        "class": RSIBollingerReversion,
                        "timeframes": [tf],
                        "window": 20,
                        "std_mult": 2.0,
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                    },
                ],
            },
        )
    )

    return candidates


def _make_demo_df(start: dt.datetime, tf: Timeframe, bars: int):
    rng = np.random.default_rng(seed=42)
    step = dt.timedelta(seconds=tf.seconds)
    idx = [start + i * step for i in range(bars)]

    price = 1.1000
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for _ in range(bars):
        drift = rng.normal(0.0, 0.0007)
        close = max(0.0001, price + drift)
        spread = abs(rng.normal(0.0009, 0.0003))
        high = max(price, close) + spread
        low = min(price, close) - spread
        opens.append(price)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        volumes.append(int(rng.integers(50, 600)))
        price = close

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.to_datetime(idx, utc=True),
    )


def _run_single(
    instrument: str,
    tf: Timeframe,
    df_slice,
    strategy_cfg: Dict,
    execution_cfg: Dict,
) -> Dict:
    data_dict = {tf: df_slice}
    cfg = {
        "data": {
            "instrument": instrument,
            "base_timeframe": tf,
            "start_date": df_slice.index.min().to_pydatetime(),
            "end_date": df_slice.index.max().to_pydatetime(),
        },
        "strategy": strategy_cfg,
        "execution": execution_cfg,
        "data_dict": data_dict,
    }
    result = BacktestEngine(cfg).run()
    net_pnl = result.final_equity - execution_cfg["initial_capital"]
    return {
        "total_trades": result.total_trades,
        "net_pnl": net_pnl,
        "win_rate": result.win_rate,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "fees": result.total_fees_paid,
        "expectancy": expectancy_per_trade(result.trades),
        "profit_factor": profit_factor(result.trades),
    }


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    if not (0.5 <= args.train_ratio <= 0.9):
        raise SystemExit("--train-ratio must be between 0.5 and 0.9")

    execution_cfg = {
        "initial_capital": float(args.initial_capital),
        "slippage_pips": float(args.slippage_pips),
        "pricing_model": args.pricing_model,
        "spreads_pips": {
            "EUR_USD": 1.4,
            "GBP_USD": 2.0,
            "USD_JPY": 1.4,
            "USD_CAD": 2.2,
            "AUD_USD": 1.4,
        },
        "core_commission_per_10k_units": 1.0,
    }

    if args.demo_bars > 0:
        df = _make_demo_df(start, tf, args.demo_bars)
    else:
        dm = DataManager({})
        data = dm.ensure_data(
            instrument=args.instrument,
            base_timeframe=tf,
            start_date=start,
            end_date=end,
            timeframes=[tf],
        )
        df = data[tf].sort_index()
    if len(df) < 200:
        raise SystemExit(f"Not enough bars for research: {len(df)}")

    split_idx = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    if test_df.empty:
        raise SystemExit("Empty test split. Adjust date range or train ratio.")

    rows: List[Dict] = []
    for strategy_name, strategy_cfg in _build_candidates(tf):
        train = _run_single(args.instrument, tf, train_df, strategy_cfg, execution_cfg)
        test = _run_single(args.instrument, tf, test_df, strategy_cfg, execution_cfg)
        row = {
            "strategy_name": strategy_name,
            "params": {k: v for k, v in strategy_cfg.items() if k not in {"class", "timeframes"}},
            "train_trades": train["total_trades"],
            "train_net_pnl": round(train["net_pnl"], 6),
            "train_expectancy": round(train["expectancy"], 6),
            "test_trades": test["total_trades"],
            "test_net_pnl": round(test["net_pnl"], 6),
            "test_expectancy": round(test["expectancy"], 6),
            "test_win_rate": round(test["win_rate"], 6),
            "test_profit_factor": round(test["profit_factor"], 6),
            "test_max_drawdown": round(test["max_drawdown"], 6),
            "test_sharpe": round(test["sharpe_ratio"], 6),
            "test_fees": round(test["fees"], 6),
            # fields expected by markdown report helper
            "total_trades": test["total_trades"],
            "net_pnl": test["net_pnl"],
            "expectancy": test["expectancy"],
            "win_rate": test["win_rate"],
            "profit_factor": test["profit_factor"],
            "max_drawdown": test["max_drawdown"],
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            float(r["test_net_pnl"]),
            float(r["test_profit_factor"]),
            -float(r["test_max_drawdown"]),
        ),
        reverse=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"strategy_research_{args.instrument}_{args.tf}_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    md_path = out_dir / f"{base}.md"

    fieldnames = [
        "strategy_name",
        "params",
        "train_trades",
        "train_net_pnl",
        "train_expectancy",
        "test_trades",
        "test_net_pnl",
        "test_expectancy",
        "test_win_rate",
        "test_profit_factor",
        "test_max_drawdown",
        "test_sharpe",
        "test_fees",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    md_path.write_text(build_report(rows), encoding="utf-8")

    print(f"Train bars: {len(train_df)}")
    print(f"Test bars: {len(test_df)}")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")
    if rows:
        best = rows[0]
        print(
            "Best candidate: "
            f"{best['strategy_name']} {best['params']} "
            f"net_pnl={best['test_net_pnl']} pf={best['test_profit_factor']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
