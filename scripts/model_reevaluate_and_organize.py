#!/usr/bin/env python3
"""Model re-evaluation, risk adjustment, and organization pipeline.

Runs regime runtime backtests over multiple out-of-sample periods, computes
comprehensive metrics, and organizes reports under models/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from oanda_bot.backtesting.core.backtester import Backtester
from oanda_bot.backtesting.core.regime_runtime import (
    KMeansRegimePredictor,
    MultiTimeframeRegimeFeatureEngineer,
    RegimeFeatureEngineer,
    RegimeModel,
)
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.warehouse import DataWarehouse
from oanda_bot.backtesting.strategy.examples.atr_breakout import ATRBreakout
from oanda_bot.backtesting.strategy.examples.breakout import Breakout
from oanda_bot.backtesting.strategy.examples.ema_pullback import EMATrendPullback
from oanda_bot.backtesting.strategy.examples.mean_reversion import MeanReversion
from oanda_bot.backtesting.strategy.examples.regime_ensemble_decision import RegimeEnsembleDecisionStrategy
from oanda_bot.backtesting.strategy.examples.regime_switch_router import RegimeSwitchRouter
from oanda_bot.backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from oanda_bot.backtesting.strategy.examples.volatility_compression_breakout import VolatilityCompressionBreakout

import scripts.run_regime_runtime_backtest as runtime_bt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EvalPeriod:
    name: str
    start: str
    end: str
    tag: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-evaluate models across OOS periods and organize reports.")
    p.add_argument("--tf", default="M15")
    p.add_argument("--decision-mode", default="ensemble", choices=["ensemble", "router"])
    p.add_argument("--gpu", default="off", choices=["off", "auto", "on"])
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--risk-per-trade-pct", type=float, default=0.0075)
    p.add_argument("--max-notional-exposure-pct", type=float, default=0.85)
    p.add_argument("--min-quantity", type=int, default=1)
    p.add_argument("--max-quantity", type=int, default=100000)
    p.add_argument("--max-drawdown-stop-pct", type=float, default=0.16)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.04)
    p.add_argument("--financing", default="on", choices=["on", "off"])
    p.add_argument("--default-financing-long-rate", type=float, default=0.03)
    p.add_argument("--default-financing-short-rate", type=float, default=0.03)
    p.add_argument("--rollover-hour-utc", type=int, default=22)
    p.add_argument("--symbols", default="EUR_USD,GBP_USD,USD_JPY,USD_CAD,GBP_JPY,XAU_USD")
    p.add_argument("--output-root", default="models")
    p.add_argument("--qual-dir", default="qualified")
    p.add_argument("--run-name", default=f"reeval_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--min-runs", type=int, default=2)
    p.add_argument("--require-positive-aggregate", action="store_true", default=True)
    p.add_argument("--max-worst-dd", type=float, default=0.25)
    p.add_argument("--max-median-dd", type=float, default=0.18)
    p.add_argument("--min-profitable-run-ratio", type=float, default=0.5)
    return p.parse_args()


def _candidate_model_paths() -> List[Path]:
    roots = [
        PROJECT_ROOT / "models" / "active",
        PROJECT_ROOT / "data" / "research",
        PROJECT_ROOT / "data" / "research" / "archive" / "unusable_models_20260228_160721",
    ]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(sorted(root.rglob("multiframe_regime_model*.json")))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p.resolve())
    return uniq


def _coverage_by_pair(tf: str, symbols: List[str]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    wh = DataWarehouse(PROJECT_ROOT / "data" / "backtesting")
    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for s in symbols:
        try:
            t = Timeframe.from_oanda_granularity(tf)
        except Exception:
            t = Timeframe.M15
        df = wh.load(s, t)
        if df is None or df.empty:
            # Fallback for unsupported enum tfs like M5.
            p_parquet = PROJECT_ROOT / "data" / "backtesting" / s / f"{tf}.parquet"
            p_csv = PROJECT_ROOT / "data" / "backtesting" / s / f"{tf}.csv"
            if p_parquet.exists():
                df = pd.read_parquet(p_parquet)
            elif p_csv.exists():
                df = pd.read_csv(p_csv, index_col=0, parse_dates=True)
            else:
                continue
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        out[s] = (idx.min(), idx.max())
    return out


def _periods() -> List[EvalPeriod]:
    return [
        EvalPeriod("post_train_6m_a", "2024-03-01", "2024-08-31", "post_train"),
        EvalPeriod("post_train_6m_b", "2024-09-01", "2025-02-28", "post_train"),
        EvalPeriod("post_train_6m_c", "2025-03-01", "2025-08-31", "post_train"),
        EvalPeriod("post_train_6m_d", "2025-09-01", "2026-02-27", "post_train"),
        EvalPeriod("pre_train_6m_a", "2013-03-01", "2013-08-31", "pre_train"),
        EvalPeriod("pre_train_6m_b", "2013-09-01", "2014-02-28", "pre_train"),
    ]


def _period_in_coverage(period: EvalPeriod, cov: Tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    s = pd.Timestamp(period.start, tz="UTC")
    e = pd.Timestamp(period.end, tz="UTC")
    return s >= cov[0] and e <= cov[1]


def _strategy_library(tf: Timeframe) -> Dict[str, Dict[str, Any]]:
    return {
        "Breakout": runtime_bt._strategy_library(tf)["Breakout"],
        "MeanReversion": runtime_bt._strategy_library(tf)["MeanReversion"],
        "EMATrendPullback": runtime_bt._strategy_library(tf)["EMATrendPullback"],
        "ATRBreakout": runtime_bt._strategy_library(tf)["ATRBreakout"],
        "RSIBollingerReversion": runtime_bt._strategy_library(tf)["RSIBollingerReversion"],
        "VolatilityCompressionBreakout": runtime_bt._strategy_library(tf)["VolatilityCompressionBreakout"],
    }


def _build_strategy_cfg(
    model: RegimeModel,
    tf: Timeframe,
    args: argparse.Namespace,
    instrument: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
) -> Tuple[Dict[str, Any], Dict[str, int], float, str]:
    library = _strategy_library(tf)
    assigned_quantities, reference_price, ref_source = runtime_bt._apply_runtime_risk_sizing(
        instrument=instrument,
        tf=tf,
        start=start_dt,
        end=end_dt,
        initial_capital=float(args.initial_capital),
        risk_per_trade_pct=float(args.risk_per_trade_pct),
        max_notional_exposure_pct=float(args.max_notional_exposure_pct),
        min_quantity=int(args.min_quantity),
        max_quantity=int(args.max_quantity),
        strategy_cfgs=library,
    )
    strategies = {}
    for strategy_name in set(model.regime_to_strategy.values()):
        if strategy_name in library:
            strategies[strategy_name] = library[strategy_name]
    if not strategies:
        raise RuntimeError("No runtime strategies matched regime_to_strategy mapping")

    default_strategy = next(iter(strategies.keys()))
    router_cfg = {
        "name": "RegimeSwitchRouter",
        "class": RegimeSwitchRouter,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "regime_to_strategy": model.regime_to_strategy,
        "default_strategy": default_strategy,
        "strategies": strategies,
    }
    ensemble_cfg = {
        "name": "RegimeEnsemble",
        "class": RegimeEnsembleDecisionStrategy,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "decision_threshold": 0.25,
        "min_votes": 1,
        "regime_style_weights": runtime_bt._regime_style_weights(model.regime_to_strategy),
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
    module_strategy_map = {
        "trend_ema_pullback": "EMATrendPullback",
        "trend_breakout": "Breakout",
        "range_mean_reversion": "MeanReversion",
        "range_rsi_reversion": "RSIBollingerReversion",
        "vol_breakout": "ATRBreakout",
        "vol_compression": "VolatilityCompressionBreakout",
    }
    for module_name, strategy_name in module_strategy_map.items():
        if module_name in ensemble_cfg["modules"] and strategy_name in assigned_quantities:
            ensemble_cfg["modules"][module_name]["quantity"] = int(assigned_quantities[strategy_name])
    selected = ensemble_cfg if args.decision_mode == "ensemble" else router_cfg
    return selected, assigned_quantities, reference_price, ref_source


def _equity_metrics(equity_curve: List[float], initial_capital: float, start: str, end: str) -> Dict[str, float]:
    e = pd.Series(equity_curve, dtype=float)
    if e.empty:
        e = pd.Series([initial_capital], dtype=float)
    r = e.pct_change().fillna(0.0)
    vol = float(r.std(ddof=0))
    downside = r[r < 0]
    down_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    ann = math.sqrt(252.0)
    ann_vol = float(vol * ann)
    ann_ret = float((e.iloc[-1] / max(e.iloc[0], 1e-9)) ** (252.0 / max(len(e), 1)) - 1.0)
    sharpe = float((r.mean() / vol) * ann) if vol > 1e-12 else 0.0
    sortino = float((r.mean() / down_std) * ann) if down_std > 1e-12 else 0.0
    peak = e.cummax()
    dd = ((peak - e) / peak.replace(0, np.nan)).fillna(0.0)
    max_dd = float(dd.max())
    max_dd_idx = int(dd.idxmax()) if not dd.empty else 0
    calmar = float(ann_ret / max(max_dd, 1e-9)) if max_dd > 0 else float("inf")
    total_return = float((e.iloc[-1] / max(initial_capital, 1e-9)) - 1.0)
    return {
        "total_return_pct": total_return * 100.0,
        "annualized_return_pct": ann_ret * 100.0,
        "annualized_volatility_pct": ann_vol * 100.0,
        "sharpe_from_equity": sharpe,
        "sortino_from_equity": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_from_equity_pct": max_dd * 100.0,
        "max_drawdown_index": float(max_dd_idx),
    }


def _streaks(pnls: Iterable[float]) -> Tuple[int, int]:
    max_win, max_loss = 0, 0
    cur_win, cur_loss = 0, 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _trade_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df.empty:
        return {
            "total_trades": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "trade_pnl_std": 0.0,
            "trade_pnl_skew": 0.0,
            "trade_pnl_kurtosis": 0.0,
            "max_win_streak": 0.0,
            "max_loss_streak": 0.0,
        }
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff = float(abs(avg_win / avg_loss)) if avg_loss < 0 else 0.0
    max_win_streak, max_loss_streak = _streaks(pnl.tolist())
    return {
        "total_trades": float(len(pnl)),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": pf,
        "expectancy_pnl": float(pnl.mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "trade_pnl_std": float(pnl.std(ddof=0)),
        "trade_pnl_skew": float(pnl.skew()) if len(pnl) > 2 else 0.0,
        "trade_pnl_kurtosis": float(pnl.kurtosis()) if len(pnl) > 3 else 0.0,
        "max_win_streak": float(max_win_streak),
        "max_loss_streak": float(max_loss_streak),
    }


def _stability_metrics(equity_curve: List[float]) -> Dict[str, float]:
    e = pd.Series(equity_curve, dtype=float)
    if len(e) < 10:
        return {
            "rolling_50_sharpe_mean": 0.0,
            "rolling_50_sharpe_min": 0.0,
            "rolling_50_return_mean_pct": 0.0,
            "rolling_50_return_min_pct": 0.0,
        }
    r = e.pct_change().fillna(0.0)
    roll_mean = r.rolling(50, min_periods=20).mean()
    roll_std = r.rolling(50, min_periods=20).std(ddof=0)
    roll_sh = (roll_mean / roll_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    roll_ret = ((e / e.shift(50)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "rolling_50_sharpe_mean": float(roll_sh.mean()),
        "rolling_50_sharpe_min": float(roll_sh.min()),
        "rolling_50_return_mean_pct": float(roll_ret.mean() * 100.0),
        "rolling_50_return_min_pct": float(roll_ret.min() * 100.0),
    }


def _compute_metrics(
    result,
    initial_capital: float,
    start: str,
    end: str,
    risk_rejections: Dict[str, int],
    regime_counts: Dict[str, int],
    assigned_quantities: Dict[str, int],
    reference_price: float,
    reference_price_source: str,
) -> Dict[str, Any]:
    trades_df = pd.DataFrame(result.trades)
    fills_df = pd.DataFrame(result.filled_orders)
    eq = list(result.equity_curve or [initial_capital])

    base = {
        "final_equity": float(result.final_equity),
        "net_pnl": float(result.final_equity - initial_capital),
        "win_rate_pct_engine": float(result.win_rate * 100.0),
        "sharpe_ratio_engine": float(result.sharpe_ratio),
        "max_drawdown_pct_engine": float(result.max_drawdown * 100.0),
        "total_fees_paid": float(result.total_fees_paid),
        "total_financing_paid": float(result.total_financing_paid),
        "risk_rejections": risk_rejections,
        "regime_counts": regime_counts,
        "assigned_quantities": assigned_quantities,
        "reference_price_used": float(reference_price),
        "reference_price_source": reference_price_source,
        "fills_count": int(len(fills_df)),
    }
    out = {}
    out.update(base)
    out.update(_trade_metrics(trades_df))
    out.update(_equity_metrics(eq, initial_capital, start, end))
    out.update(_stability_metrics(eq))

    out["fees_per_trade"] = float(out["total_fees_paid"] / max(1.0, out["total_trades"]))
    out["financing_per_trade"] = float(out["total_financing_paid"] / max(1.0, out["total_trades"]))
    out["cost_to_net_pnl_ratio"] = float(
        (out["total_fees_paid"] + out["total_financing_paid"]) / (abs(out["net_pnl"]) + 1e-9)
    )
    out["profitable"] = bool((out["net_pnl"] > 0.0) and (out["max_drawdown_pct_engine"] <= 100.0 * 0.25))
    return out


def _run_backtest(
    model_path: Path,
    instrument: str,
    period: EvalPeriod,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model = RegimeModel.load(str(model_path))
    tf = Timeframe.from_oanda_granularity(args.tf)
    use_gpu, runtime_backend = runtime_bt._resolve_gpu_mode(args.gpu)
    if any(col.startswith(("m15_", "h1_", "h4_")) for col in model.feature_columns):
        feature_engineer = MultiTimeframeRegimeFeatureEngineer(use_gpu=use_gpu)
    else:
        feature_engineer = RegimeFeatureEngineer(use_gpu=use_gpu)
    regime_predictor = KMeansRegimePredictor(model, use_gpu=use_gpu)

    start_dt = dt.datetime.fromisoformat(period.start)
    end_dt = dt.datetime.fromisoformat(period.end)
    strategy_cfg, assigned_quantities, ref_price, ref_src = _build_strategy_cfg(
        model=model,
        tf=tf,
        args=args,
        instrument=instrument,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    ctx = {
        "data": {
            "instrument": instrument,
            "base_timeframe": tf,
            "start_date": start_dt,
            "end_date": end_dt,
        },
        "strategy": strategy_cfg,
        "execution": {
            "initial_capital": float(args.initial_capital),
            "fill_mode": "next_open",
            "slippage_pips": 0.2,
            "pricing_model": "oanda_core",
            "spreads_pips": {
                "EUR_USD": 1.4,
                "GBP_USD": 2.0,
                "USD_JPY": 1.4,
                "XAU_USD": 20.0,
            },
            "core_commission_per_10k_units": 1.0,
            "min_quantity": int(args.min_quantity),
            "max_quantity": int(args.max_quantity),
            "financing_enabled": args.financing == "on",
            "default_financing_long_rate": float(args.default_financing_long_rate),
            "default_financing_short_rate": float(args.default_financing_short_rate),
            "rollover_hour_utc": int(args.rollover_hour_utc),
            "wednesday_triple_rollover": True,
            "financing_long_rate_by_instrument": {
                "EUR_USD": 0.025,
                "GBP_USD": 0.03,
                "USD_JPY": 0.015,
                "XAU_USD": 0.10,
            },
            "financing_short_rate_by_instrument": {
                "EUR_USD": 0.02,
                "GBP_USD": 0.025,
                "USD_JPY": 0.015,
                "XAU_USD": 0.08,
            },
        },
    }
    risk_manager = runtime_bt.RuntimeGuardrailRiskManager(
        initial_capital=float(args.initial_capital),
        max_drawdown_stop_pct=float(args.max_drawdown_stop_pct),
        daily_loss_limit_pct=float(args.daily_loss_limit_pct),
    )
    bt = Backtester(
        context=ctx,
        feature_engineer=feature_engineer,
        regime_predictor=regime_predictor,
        risk_manager=risk_manager,
    )
    result = bt.run()
    metrics = _compute_metrics(
        result=result,
        initial_capital=float(args.initial_capital),
        start=period.start,
        end=period.end,
        risk_rejections=risk_manager.rejections,
        regime_counts=getattr(regime_predictor, "regime_counts", {}),
        assigned_quantities=assigned_quantities,
        reference_price=ref_price,
        reference_price_source=ref_src,
    )
    return {
        "runtime_backend": runtime_backend,
        "metrics": metrics,
        "trades_df": pd.DataFrame(result.trades),
        "fills_df": pd.DataFrame(result.filled_orders),
        "equity_df": pd.DataFrame({"equity": pd.Series(result.equity_curve, dtype=float)}),
        "model_base_tf": getattr(model, "base_tf", ""),
        "model_regime_to_strategy": getattr(model, "regime_to_strategy", {}),
    }


def _model_id(path: Path) -> str:
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{h}"


def _qualifies(runs: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    if len(runs) < int(args.min_runs):
        return False, {"reason": "insufficient_runs"}
    pnl = np.array([float(r["metrics"]["net_pnl"]) for r in runs], dtype=float)
    dd = np.array([float(r["metrics"]["max_drawdown_pct_engine"]) for r in runs], dtype=float)
    sharpe = np.array([float(r["metrics"]["sharpe_ratio_engine"]) for r in runs], dtype=float)
    profitable_ratio = float((pnl > 0).mean()) if len(pnl) else 0.0
    agg = {
        "runs": int(len(runs)),
        "net_pnl_sum": float(pnl.sum()),
        "net_pnl_mean": float(pnl.mean()),
        "sharpe_mean": float(sharpe.mean()) if len(sharpe) else 0.0,
        "worst_drawdown_pct": float(dd.max()) if len(dd) else 0.0,
        "median_drawdown_pct": float(np.median(dd)) if len(dd) else 0.0,
        "profitable_run_ratio": profitable_ratio,
    }
    ok = True
    if bool(args.require_positive_aggregate) and agg["net_pnl_sum"] <= 0.0:
        ok = False
    if agg["worst_drawdown_pct"] > float(args.max_worst_dd) * 100.0:
        ok = False
    if agg["median_drawdown_pct"] > float(args.max_median_dd) * 100.0:
        ok = False
    if agg["profitable_run_ratio"] < float(args.min_profitable_run_ratio):
        ok = False
    return ok, agg


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    periods = _periods()
    coverage = _coverage_by_pair(args.tf, symbols)
    candidates = _candidate_model_paths()

    run_root = PROJECT_ROOT / args.output_root / "reevaluations" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    qual_root = PROJECT_ROOT / args.output_root / args.qual_dir
    qual_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": args.run_name,
        "risk_profile": {
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "max_notional_exposure_pct": args.max_notional_exposure_pct,
            "max_drawdown_stop_pct": args.max_drawdown_stop_pct,
            "daily_loss_limit_pct": args.daily_loss_limit_pct,
        },
        "periods": [asdict(p) for p in periods],
        "coverage": {k: [str(v[0]), str(v[1])] for k, v in coverage.items()},
        "candidate_count": len(candidates),
        "models": [],
    }

    all_rows: List[Dict[str, Any]] = []
    for model_path in candidates:
        try:
            obj = json.loads(model_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_pairs = obj.get("instruments", symbols)
        if isinstance(model_pairs, str):
            model_pairs = [model_pairs]
        model_pairs = [p for p in model_pairs if p in coverage]
        if not model_pairs:
            continue

        mid = _model_id(model_path)
        mdir = run_root / mid
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "source_model_path.txt").write_text(str(model_path), encoding="utf-8")
        shutil.copy2(model_path, mdir / model_path.name)

        model_runs: List[Dict[str, Any]] = []
        for pair in model_pairs:
            pair_dir = mdir / "by_pair" / pair
            pair_dir.mkdir(parents=True, exist_ok=True)
            for period in periods:
                if not _period_in_coverage(period, coverage[pair]):
                    continue
                pdir = pair_dir / period.name
                pdir.mkdir(parents=True, exist_ok=True)
                try:
                    out = _run_backtest(model_path=model_path, instrument=pair, period=period, args=args)
                    metrics = out["metrics"]
                    metrics.update(
                        {
                            "model_id": mid,
                            "model_path": str(model_path),
                            "pair": pair,
                            "period": period.name,
                            "period_start": period.start,
                            "period_end": period.end,
                        }
                    )
                    (pdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    out["trades_df"].to_csv(pdir / "trades.csv", index=False)
                    out["fills_df"].to_csv(pdir / "fills.csv", index=False)
                    out["equity_df"].to_csv(pdir / "equity_curve.csv", index=False)
                    raw = {
                        "runtime_backend": out["runtime_backend"],
                        "model_base_tf": out["model_base_tf"],
                        "model_regime_to_strategy": out["model_regime_to_strategy"],
                    }
                    (pdir / "raw_result.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
                    model_runs.append({"pair": pair, "period": period.name, "metrics": metrics})
                    all_rows.append(metrics)
                except Exception as exc:
                    err = {"error": str(exc), "pair": pair, "period": period.name}
                    (pdir / "error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")

        ok, agg = _qualifies(model_runs, args)
        summary = {
            "model_id": mid,
            "source_model_path": str(model_path),
            "pairs_tested": sorted({r["pair"] for r in model_runs}),
            "periods_tested": sorted({r["period"] for r in model_runs}),
            "run_count": len(model_runs),
            "qualification": {"qualified": ok, **agg},
        }
        (mdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if ok:
            qdir = qual_root / mid
            if qdir.exists():
                shutil.rmtree(qdir)
            shutil.copytree(mdir, qdir)
            # keep canonical model copy in qualified dir
            shutil.copy2(model_path, qdir / model_path.name)
        manifest["models"].append(summary)

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(run_root / "all_runs_metrics.csv", index=False)
        model_summary_rows = []
        for m in manifest["models"]:
            q = m["qualification"]
            model_summary_rows.append(
                {
                    "model_id": m["model_id"],
                    "qualified": q.get("qualified", False),
                    "run_count": m["run_count"],
                    "net_pnl_sum": q.get("net_pnl_sum", 0.0),
                    "net_pnl_mean": q.get("net_pnl_mean", 0.0),
                    "profitable_run_ratio": q.get("profitable_run_ratio", 0.0),
                    "worst_drawdown_pct": q.get("worst_drawdown_pct", 0.0),
                    "median_drawdown_pct": q.get("median_drawdown_pct", 0.0),
                    "sharpe_mean": q.get("sharpe_mean", 0.0),
                    "source_model_path": m["source_model_path"],
                }
            )
        pd.DataFrame(model_summary_rows).sort_values(
            by=["qualified", "net_pnl_sum", "sharpe_mean"],
            ascending=[False, False, False],
        ).to_csv(run_root / "model_qualification_summary.csv", index=False)

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "qualified_root": str(qual_root),
                "candidate_models": len(candidates),
                "evaluated_models": len(manifest["models"]),
                "qualified_models": int(sum(1 for m in manifest["models"] if m["qualification"]["qualified"])),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
