#!/usr/bin/env python3
"""Regime-based strategy research with optional GPU clustering."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.analysis.metrics import expectancy_per_trade, profit_factor
from backtesting.core.engine import BacktestEngine
from backtesting.core.timeframe import Timeframe
from backtesting.data.manager import DataManager
from backtesting.strategy.examples.atr_breakout import ATRBreakout
from backtesting.strategy.examples.breakout import Breakout
from backtesting.strategy.examples.ema_pullback import EMATrendPullback
from backtesting.strategy.examples.ensemble_vote import EnsembleVoteStrategy
from backtesting.strategy.examples.mean_reversion import MeanReversion
from backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from backtesting.strategy.examples.volatility_compression_breakout import (
    VolatilityCompressionBreakout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime research with optional GPU KMeans.")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--tf", default="M15")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--kmeans-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--pricing-model", default="oanda_core", choices=["spread_only", "oanda_core"])
    parser.add_argument("--demo-bars", type=int, default=0)
    return parser.parse_args()


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _make_demo_df(start: dt.datetime, tf: Timeframe, bars: int):
    rng = np.random.default_rng(seed=42)
    step = dt.timedelta(seconds=tf.seconds)
    idx = [start + i * step for i in range(bars)]
    price = 1.1
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(bars):
        if i < bars * 0.25:
            drift = rng.normal(0.0, 0.0006)
            vol_scale = 0.8
        elif i < bars * 0.5:
            drift = rng.normal(0.00025, 0.0005)
            vol_scale = 1.0
        elif i < bars * 0.75:
            drift = rng.normal(0.0, 0.0003)
            vol_scale = 0.5
        else:
            drift = rng.normal(-0.0002, 0.0007)
            vol_scale = 1.2
        close = max(0.0001, price + drift)
        span = abs(rng.normal(0.001, 0.0002)) * vol_scale
        high = max(price, close) + span
        low = min(price, close) - span
        opens.append(price)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        vols.append(int(rng.integers(80, 1000)))
        price = close
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=pd.to_datetime(idx, utc=True),
    )


def _build_candidates(tf: Timeframe) -> List[Tuple[str, Dict]]:
    return [
        (
            "Breakout",
            {
                "name": "Breakout_L40",
                "class": Breakout,
                "timeframes": [tf],
                "lookback": 40,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.008,
                "min_breakout_pct": 0.0002,
                "quantity": 10000,
            },
        ),
        (
            "MeanReversion",
            {
                "name": "MeanRev_SMA30_D0.002",
                "class": MeanReversion,
                "timeframes": [tf],
                "sma_period": 30,
                "deviation_pct": 0.002,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.003,
                "quantity": 10000,
            },
        ),
        (
            "EMATrendPullback",
            {
                "name": "EMAPullback_20_100",
                "class": EMATrendPullback,
                "timeframes": [tf],
                "fast_period": 20,
                "slow_period": 100,
                "pullback_pct": 0.0008,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
        ),
        (
            "ATRBreakout",
            {
                "name": "ATRBreakout_L20",
                "class": ATRBreakout,
                "timeframes": [tf],
                "lookback": 20,
                "atr_period": 14,
                "atr_mult": 1.2,
                "stop_loss_pct": 0.0045,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
        ),
        (
            "RSIBollingerReversion",
            {
                "name": "RSIBB_STD2.0",
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
        ),
        (
            "VolatilityCompressionBreakout",
            {
                "name": "VolCompression_0.75",
                "class": VolatilityCompressionBreakout,
                "timeframes": [tf],
                "range_lookback": 20,
                "atr_period": 14,
                "compression_window": 40,
                "compression_ratio": 0.75,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.010,
                "quantity": 10000,
            },
        ),
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
        ),
    ]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = df["close"].pct_change()
    out["ret_4"] = df["close"].pct_change(4)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    out["range_pct"] = (df["high"] - df["low"]) / df["close"]
    out["range_ma_20"] = out["range_pct"].rolling(20).mean()
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    out["trend_strength"] = (ema_fast - ema_slow) / df["close"]

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / df["close"]

    out = out.dropna()
    return out


def _kmeans_numpy(x: np.ndarray, k: int, iters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=k, replace=False)
    centers = x[idx].copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            points = x[labels == j]
            new_centers[j] = points.mean(axis=0) if len(points) else centers[j]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def _kmeans_cupy(x: np.ndarray, k: int, iters: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore

    cp.random.seed(seed)
    xg = cp.asarray(x, dtype=cp.float32)
    idx = cp.random.choice(xg.shape[0], size=k, replace=False)
    centers = xg[idx].copy()
    labels = cp.zeros(xg.shape[0], dtype=cp.int32)
    for _ in range(iters):
        dists = cp.sum((xg[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = cp.argmin(dists, axis=1)
        new_centers = cp.zeros_like(centers)
        for j in range(k):
            points = xg[labels == j]
            if points.shape[0] == 0:
                new_centers[j] = centers[j]
            else:
                new_centers[j] = cp.mean(points, axis=0)
        if cp.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return cp.asnumpy(labels), cp.asnumpy(centers)


def _cluster_regimes(
    features: pd.DataFrame,
    train_ratio: float,
    k: int,
    iters: int,
    seed: int,
    gpu_mode: str,
) -> Tuple[pd.Series, str, np.ndarray, Dict[str, float], Dict[str, float], List[str]]:
    split = int(len(features) * train_ratio)
    train = features.iloc[:split]
    train_mean = train.mean()
    train_std = train.std().replace(0, 1.0)
    all_scaled = (features - train_mean) / train_std
    all_scaled = all_scaled.replace([np.inf, -np.inf], np.nan).dropna()
    train_scaled = all_scaled.iloc[: min(split, len(all_scaled))]

    backend = "cpu"
    labels = None
    centers = None
    x_train = train_scaled.to_numpy(dtype=np.float64)
    x_all = all_scaled.to_numpy(dtype=np.float64)

    if gpu_mode in {"auto", "on"}:
        try:
            train_labels, centers = _kmeans_cupy(x_train, k, iters, seed)
            backend = "gpu(cupy)"
            # Assign all rows to nearest train centers on CPU for compatibility.
            d = ((x_all[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
        except Exception:
            if gpu_mode == "on":
                raise

    if labels is None:
        train_labels, centers = _kmeans_numpy(x_train, k, iters, seed)
        d = ((x_all[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)

    regime_series = pd.Series(labels, index=all_scaled.index, name="regime").astype(int)
    return (
        regime_series,
        backend,
        np.asarray(centers, dtype=np.float64),
        {k: float(v) for k, v in train_mean.items()},
        {k: float(v) for k, v in train_std.items()},
        list(all_scaled.columns),
    )


def _run_engine(
    instrument: str,
    tf: Timeframe,
    df_slice: pd.DataFrame,
    strategy_cfg: Dict,
    execution_cfg: Dict,
):
    cfg = {
        "data": {
            "instrument": instrument,
            "base_timeframe": tf,
            "start_date": df_slice.index.min().to_pydatetime(),
            "end_date": df_slice.index.max().to_pydatetime(),
        },
        "strategy": strategy_cfg,
        "execution": execution_cfg,
        "data_dict": {tf: df_slice},
    }
    result = BacktestEngine(cfg).run()
    return result


def _trade_regime_stats(trades: List[Dict], regimes: pd.Series) -> Dict[int, Dict]:
    stats: Dict[int, Dict] = {}
    if regimes.empty:
        return stats
    for trade in trades:
        ts = pd.Timestamp(trade["timestamp"])
        try:
            regime = int(regimes.loc[:ts].iloc[-1])
        except Exception:
            continue
        if regime not in stats:
            stats[regime] = {"net_pnl": 0.0, "trades": 0, "wins": 0, "pnl_list": []}
        pnl = float(trade["pnl"])
        stats[regime]["net_pnl"] += pnl
        stats[regime]["trades"] += 1
        stats[regime]["wins"] += 1 if pnl > 0 else 0
        stats[regime]["pnl_list"].append(pnl)
    for regime, s in stats.items():
        t = s["trades"]
        s["win_rate"] = (s["wins"] / t) if t else 0.0
        s["expectancy"] = (sum(s["pnl_list"]) / t) if t else 0.0
        s["profit_factor"] = (
            sum(p for p in s["pnl_list"] if p > 0) / abs(sum(p for p in s["pnl_list"] if p < 0))
            if any(p < 0 for p in s["pnl_list"])
            else (float("inf") if any(p > 0 for p in s["pnl_list"]) else 0.0)
        )
    return stats


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if not (0.5 <= args.train_ratio <= 0.9):
        raise SystemExit("--train-ratio must be between 0.5 and 0.9")
    if args.regimes < 2:
        raise SystemExit("--regimes must be >=2")

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
    if len(df) < 500:
        raise SystemExit(f"Not enough bars for regime research: {len(df)}")

    features = _build_features(df)
    regimes, backend, centers, train_mean, train_std, feature_columns = _cluster_regimes(
        features=features,
        train_ratio=args.train_ratio,
        k=args.regimes,
        iters=args.kmeans_iter,
        seed=args.seed,
        gpu_mode=args.gpu,
    )

    split_ts = regimes.index[int(len(regimes) * args.train_ratio)]
    train_regimes = regimes.loc[regimes.index < split_ts]
    test_regimes = regimes.loc[regimes.index >= split_ts]
    df_aligned = df.loc[regimes.index.min() : regimes.index.max()].copy()
    train_df = df_aligned.loc[df_aligned.index < split_ts]
    test_df = df_aligned.loc[df_aligned.index >= split_ts]
    if train_df.empty or test_df.empty:
        raise SystemExit("Train/test split is empty after feature alignment.")

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
            "XAU_USD": 20.0,
        },
        "core_commission_per_10k_units": 1.0,
    }

    candidates = _build_candidates(tf)
    rows: List[Dict] = []
    best_by_regime: Dict[int, Dict] = {}

    for strategy_name, strategy_cfg in candidates:
        train_res = _run_engine(args.instrument, tf, train_df, strategy_cfg, execution_cfg)
        test_res = _run_engine(args.instrument, tf, test_df, strategy_cfg, execution_cfg)
        train_stats = _trade_regime_stats(train_res.trades, train_regimes)
        test_stats = _trade_regime_stats(test_res.trades, test_regimes)

        for regime in range(args.regimes):
            tr = train_stats.get(regime, {"net_pnl": 0.0, "trades": 0, "expectancy": 0.0, "profit_factor": 0.0})
            te = test_stats.get(regime, {"net_pnl": 0.0, "trades": 0, "expectancy": 0.0, "profit_factor": 0.0})
            row = {
                "strategy_name": strategy_name,
                "params": json.dumps(
                    {k: v for k, v in strategy_cfg.items() if k not in {"class", "timeframes"}},
                    default=str,
                ),
                "regime": regime,
                "train_trades": int(tr["trades"]),
                "train_net_pnl": round(float(tr["net_pnl"]), 6),
                "train_expectancy": round(float(tr["expectancy"]), 6),
                "train_profit_factor": round(float(tr["profit_factor"]), 6),
                "test_trades": int(te["trades"]),
                "test_net_pnl": round(float(te["net_pnl"]), 6),
                "test_expectancy": round(float(te["expectancy"]), 6),
                "test_profit_factor": round(float(te["profit_factor"]), 6),
                "train_total_net": round(float(train_res.final_equity - args.initial_capital), 6),
                "test_total_net": round(float(test_res.final_equity - args.initial_capital), 6),
                "test_total_pf": round(float(profit_factor(test_res.trades)), 6),
                "test_total_expectancy": round(float(expectancy_per_trade(test_res.trades)), 6),
            }
            rows.append(row)

            score = float(tr["net_pnl"]) + (5.0 * float(tr["expectancy"]))
            if tr["trades"] < 2:
                score -= 1e9
            prev = best_by_regime.get(regime)
            if prev is None or score > prev["score"]:
                best_by_regime[regime] = {
                    "score": score,
                    "strategy_name": strategy_name,
                    "params": row["params"],
                    "train_net_pnl": float(tr["net_pnl"]),
                    "train_expectancy": float(tr["expectancy"]),
                    "train_trades": int(tr["trades"]),
                    "test_net_pnl": float(te["net_pnl"]),
                    "test_expectancy": float(te["expectancy"]),
                    "test_trades": int(te["trades"]),
                }

    rows.sort(
        key=lambda r: (
            int(r["regime"]),
            float(r["train_net_pnl"]),
            float(r["train_expectancy"]),
            float(r["test_net_pnl"]),
        ),
        reverse=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"regime_research_{args.instrument}_{args.tf}_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    best_csv_path = out_dir / f"{base}_best_by_regime.csv"
    regimes_csv_path = out_dir / f"{base}_bar_regimes.csv"
    model_json_path = out_dir / f"{base}_runtime_model.json"
    md_path = out_dir / f"{base}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "strategy_name",
            "params",
            "regime",
            "train_trades",
            "train_net_pnl",
            "train_expectancy",
            "train_profit_factor",
            "test_trades",
            "test_net_pnl",
            "test_expectancy",
            "test_profit_factor",
            "train_total_net",
            "test_total_net",
            "test_total_pf",
            "test_total_expectancy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with best_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "regime",
            "strategy_name",
            "params",
            "train_trades",
            "train_net_pnl",
            "train_expectancy",
            "test_trades",
            "test_net_pnl",
            "test_expectancy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime in sorted(best_by_regime):
            row = dict(best_by_regime[regime])
            row["regime"] = regime
            row.pop("score", None)
            writer.writerow(row)

    regimes.to_frame().to_csv(regimes_csv_path)

    regime_to_strategy = {
        str(regime): str(best_by_regime[regime]["strategy_name"])
        for regime in sorted(best_by_regime)
    }
    model_obj = {
        "instrument": args.instrument,
        "timeframe": args.tf,
        "backend": backend,
        "feature_columns": feature_columns,
        "train_mean": train_mean,
        "train_std": train_std,
        "centers": centers.tolist(),
        "regime_to_strategy": regime_to_strategy,
    }
    model_json_path.write_text(json.dumps(model_obj, indent=2), encoding="utf-8")

    regime_counts = regimes.value_counts().sort_index().to_dict()
    md_lines = ["# Regime GPU Research", ""]
    md_lines.append(f"Generated: {dt.datetime.utcnow().isoformat()}Z")
    md_lines.append(f"Instrument: {args.instrument} | TF: {args.tf}")
    md_lines.append(f"Clustering backend: {backend}")
    md_lines.append(f"Regimes: {args.regimes}")
    md_lines.append(f"Bars used: {len(df_aligned)} | Train bars: {len(train_df)} | Test bars: {len(test_df)}")
    md_lines.append("")
    md_lines.append("## Regime Distribution")
    md_lines.append("")
    for regime, count in regime_counts.items():
        md_lines.append(f"- Regime {regime}: {count} bars")
    md_lines.append("")
    md_lines.append("## Best Strategy Per Regime (selected on train, shown with test stats)")
    md_lines.append("")
    md_lines.append("| Regime | Strategy | Train Trades | Train Net | Train Expectancy | Test Trades | Test Net | Test Expectancy |")
    md_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for regime in sorted(best_by_regime):
        b = best_by_regime[regime]
        md_lines.append(
            f"| {regime} | {b['strategy_name']} | {b['train_trades']} | {b['train_net_pnl']:.4f} | "
            f"{b['train_expectancy']:.4f} | {b['test_trades']} | {b['test_net_pnl']:.4f} | {b['test_expectancy']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("Note: regime-strategy portfolio is a model-selection proxy; execute a dedicated regime-switching backtest before live use.")
    md_lines.append("")
    md_lines.append(f"Detailed CSV: `{csv_path}`")
    md_lines.append(f"Best-by-regime CSV: `{best_csv_path}`")
    md_lines.append(f"Per-bar regimes CSV: `{regimes_csv_path}`")
    md_lines.append(f"Runtime model JSON: `{model_json_path}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Backend: {backend}")
    print(f"Detailed CSV: {csv_path}")
    print(f"Best-by-regime CSV: {best_csv_path}")
    print(f"Per-bar regimes CSV: {regimes_csv_path}")
    print(f"Runtime model JSON: {model_json_path}")
    print(f"Report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
