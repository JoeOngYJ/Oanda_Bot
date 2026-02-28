#!/usr/bin/env python3
"""Run regime-classifier driven strategy selection in Backtester runtime."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.core.backtester import Backtester
from backtesting.core.regime_runtime import (
    KMeansRegimePredictor,
    MultiTimeframeRegimeFeatureEngineer,
    RegimeFeatureEngineer,
    RegimeModel,
)
from backtesting.core.timeframe import Timeframe
from backtesting.strategy.examples.atr_breakout import ATRBreakout
from backtesting.strategy.examples.breakout import Breakout
from backtesting.strategy.examples.ema_pullback import EMATrendPullback
from backtesting.strategy.examples.mean_reversion import MeanReversion
from backtesting.strategy.examples.regime_ensemble_decision import RegimeEnsembleDecisionStrategy
from backtesting.strategy.examples.regime_switch_router import RegimeSwitchRouter
from backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from backtesting.strategy.examples.volatility_compression_breakout import VolatilityCompressionBreakout


def parse_args():
    p = argparse.ArgumentParser(description="Run regime runtime backtest with strategy auto-selection.")
    p.add_argument("--model-json", required=True, help="Path from run_regime_gpu_research *_runtime_model.json")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--tf", default="M15")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fill-mode", default="next_open", choices=["touch", "next_open"])
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--decision-mode", default="ensemble", choices=["ensemble", "router"])
    return p.parse_args()


def _strategy_library(tf: Timeframe) -> Dict[str, Dict]:
    return {
        "Breakout": {
            "name": "Breakout_runtime",
            "class": Breakout,
            "timeframes": [tf],
            "lookback": 40,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.008,
            "min_breakout_pct": 0.0002,
            "quantity": 10000,
        },
        "MeanReversion": {
            "name": "MeanRev_runtime",
            "class": MeanReversion,
            "timeframes": [tf],
            "sma_period": 20,
            "deviation_pct": 0.003,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": 10000,
        },
        "EMATrendPullback": {
            "name": "EMAPullback_runtime",
            "class": EMATrendPullback,
            "timeframes": [tf],
            "fast_period": 20,
            "slow_period": 100,
            "pullback_pct": 0.0008,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.009,
            "quantity": 10000,
        },
        "ATRBreakout": {
            "name": "ATRBreakout_runtime",
            "class": ATRBreakout,
            "timeframes": [tf],
            "lookback": 20,
            "atr_period": 14,
            "atr_mult": 1.2,
            "stop_loss_pct": 0.0045,
            "take_profit_pct": 0.009,
            "quantity": 10000,
        },
        "RSIBollingerReversion": {
            "name": "RSIBB_runtime",
            "class": RSIBollingerReversion,
            "timeframes": [tf],
            "window": 20,
            "std_mult": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": 10000,
        },
        "VolatilityCompressionBreakout": {
            "name": "VolCompression_runtime",
            "class": VolatilityCompressionBreakout,
            "timeframes": [tf],
            "range_lookback": 20,
            "atr_period": 14,
            "compression_window": 40,
            "compression_ratio": 0.75,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.01,
            "quantity": 10000,
        },
    }

def _regime_style_weights(regime_to_strategy: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for regime, strategy_name in regime_to_strategy.items():
        weights = {"trend": 0.7, "mean_reversion": 0.7, "breakout": 0.7, "stat_arb": 0.4}
        n = str(strategy_name)
        if "MeanReversion" in n or "RSI" in n:
            weights["mean_reversion"] = 1.8
        if "Breakout" in n or "Compression" in n:
            weights["breakout"] = 1.8
        if "EMA" in n or "Trend" in n:
            weights["trend"] = 1.8
        out[str(regime)] = weights
    return out


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)
    model = RegimeModel.load(args.model_json)
    if any(col.startswith(("m15_", "h1_", "h4_")) for col in model.feature_columns):
        feature_engineer = MultiTimeframeRegimeFeatureEngineer()
    else:
        feature_engineer = RegimeFeatureEngineer()
    regime_predictor = KMeansRegimePredictor(model)

    library = _strategy_library(tf)
    strategies = {}
    for strategy_name in set(model.regime_to_strategy.values()):
        if strategy_name in library:
            strategies[strategy_name] = library[strategy_name]
    if not strategies:
        raise SystemExit("No runtime strategies matched model regime_to_strategy mapping.")

    default_strategy = next(iter(strategies.keys()))
    router_cfg = {
        "name": "RegimeSwitchRouter",
        "class": RegimeSwitchRouter,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4],
        "regime_to_strategy": model.regime_to_strategy,
        "default_strategy": default_strategy,
        "strategies": strategies,
    }
    ensemble_cfg = {
        "name": "RegimeEnsemble",
        "class": RegimeEnsembleDecisionStrategy,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4],
        "decision_threshold": 0.25,
        "min_votes": 1,
        "regime_style_weights": _regime_style_weights(model.regime_to_strategy),
        "modules": {
            "trend_ema_pullback": {
                "name": "trend_ema_pullback",
                "class": EMATrendPullback,
                "timeframes": [tf],
                "style": "trend",
                "weight": 1.0,
                "fast_period": 20,
                "slow_period": 100,
                "pullback_pct": 0.0008,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
            "trend_breakout": {
                "name": "trend_breakout",
                "class": Breakout,
                "timeframes": [tf],
                "style": "trend",
                "weight": 0.9,
                "lookback": 40,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.008,
                "min_breakout_pct": 0.0002,
                "quantity": 10000,
            },
            "range_mean_reversion": {
                "name": "range_mean_reversion",
                "class": MeanReversion,
                "timeframes": [tf],
                "style": "mean_reversion",
                "weight": 1.0,
                "sma_period": 20,
                "deviation_pct": 0.003,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.003,
                "quantity": 10000,
            },
            "range_rsi_reversion": {
                "name": "range_rsi_reversion",
                "class": RSIBollingerReversion,
                "timeframes": [tf],
                "style": "mean_reversion",
                "weight": 0.8,
                "window": 20,
                "std_mult": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.003,
                "quantity": 10000,
            },
            "vol_breakout": {
                "name": "vol_breakout",
                "class": ATRBreakout,
                "timeframes": [tf],
                "style": "breakout",
                "weight": 1.0,
                "lookback": 20,
                "atr_period": 14,
                "atr_mult": 1.2,
                "stop_loss_pct": 0.0045,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
            "vol_compression": {
                "name": "vol_compression",
                "class": VolatilityCompressionBreakout,
                "timeframes": [tf],
                "style": "breakout",
                "weight": 0.9,
                "range_lookback": 20,
                "atr_period": 14,
                "compression_window": 40,
                "compression_ratio": 0.75,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.01,
                "quantity": 10000,
            },
        },
    }
    selected_strategy_cfg = ensemble_cfg if args.decision_mode == "ensemble" else router_cfg

    ctx = {
        "data": {
            "instrument": args.instrument,
            "base_timeframe": tf,
            "start_date": dt.datetime.fromisoformat(args.start),
            "end_date": dt.datetime.fromisoformat(args.end),
        },
        "strategy": selected_strategy_cfg,
        "execution": {
            "initial_capital": float(args.initial_capital),
            "fill_mode": args.fill_mode,
            "slippage_pips": 0.2,
            "pricing_model": "oanda_core",
            "spreads_pips": {
                "EUR_USD": 1.4,
                "GBP_USD": 2.0,
                "USD_JPY": 1.4,
                "XAU_USD": 20.0,
            },
            "core_commission_per_10k_units": 1.0,
        },
    }

    result = Backtester(
        context=ctx,
        feature_engineer=feature_engineer,
        regime_predictor=regime_predictor,
    ).run()

    print(f"Trades: {result.total_trades}")
    print(f"Final equity: {result.final_equity:.2f}")
    print(f"Net PnL: {result.final_equity - args.initial_capital:.2f}")
    print(f"Win rate: {result.win_rate:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Fees: {result.total_fees_paid:.2f}")
    print(f"Regime counts: {regime_predictor.regime_counts}")
    print(f"Regime->strategy: {model.regime_to_strategy}")
    print(f"Decision mode: {args.decision_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
