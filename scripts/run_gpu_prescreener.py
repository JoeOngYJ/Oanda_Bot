#!/usr/bin/env python3
"""GPU-accelerated strategy parameter pre-screener for faster backtest selection."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast vectorized pre-screen to rank strategy parameter sets. "
            "Use this to reduce full engine backtest workload."
        )
    )
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--tf", default="M15")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument("--pricing-model", default="oanda_core", choices=["spread_only", "oanda_core"])
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--core-commission-per-10k", type=float, default=1.0)
    parser.add_argument("--demo-bars", type=int, default=0)
    return parser.parse_args()


def _parse_dt(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _make_demo_df(start: dt.datetime, tf: Timeframe, bars: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    step = dt.timedelta(seconds=tf.seconds)
    idx = [start + i * step for i in range(bars)]
    price = 1.1000
    rows = []
    for i in range(bars):
        phase = (i // max(1, bars // 4)) % 4
        if phase == 0:
            drift = rng.normal(0.00015, 0.00045)
        elif phase == 1:
            drift = rng.normal(0.0, 0.0003)
        elif phase == 2:
            drift = rng.normal(-0.00012, 0.0004)
        else:
            drift = rng.normal(0.0, 0.0006)
        close = max(0.0001, price + drift)
        span = abs(rng.normal(0.0009, 0.00025))
        high = max(price, close) + span
        low = min(price, close) - span
        rows.append((price, high, low, close, int(rng.integers(100, 2000))))
        price = close
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=pd.to_datetime(idx, utc=True),
    )


def _select_backend(gpu_mode: str):
    if gpu_mode == "off":
        return np, "cpu(numpy)"
    try:
        import cupy as cp  # type: ignore

        _ = cp.cuda.runtime.getDeviceCount()
        return cp, "gpu(cupy)"
    except Exception:
        if gpu_mode == "on":
            raise
        return np, "cpu(numpy)"


def _pips_scale(instrument: str) -> float:
    return 100.0 if instrument.endswith("JPY") else 10000.0


def _estimate_cost_per_position_change(
    instrument: str,
    pricing_model: str,
    slippage_pips: float,
    core_commission_per_10k: float,
) -> float:
    spread_pips_map = {
        "EUR_USD": 1.4,
        "GBP_USD": 2.0,
        "USD_JPY": 1.4,
        "USD_CAD": 2.2,
        "AUD_USD": 1.4,
        "XAU_USD": 20.0,
    }
    spread_pips = float(spread_pips_map.get(instrument, 1.6))
    pips_to_return = 1.0 / _pips_scale(instrument)

    # Approximate return drag for one position change event.
    spread_drag = spread_pips * pips_to_return
    slip_drag = max(0.0, slippage_pips) * pips_to_return
    comm_drag = 0.0
    if pricing_model.lower() == "oanda_core":
        # Approximate commission in return terms per 10k notional.
        comm_drag = core_commission_per_10k / 10000.0
    return spread_drag + slip_drag + comm_drag


def _score_signal_matrix(
    xp,
    signals: np.ndarray,
    returns: np.ndarray,
    tf: Timeframe,
    cost_per_change: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = xp.asarray(signals, dtype=xp.float32)
    ret = xp.asarray(returns, dtype=xp.float32)

    gross = sig[:-1, :] * ret[:, None]
    changes = xp.abs(sig[1:, :] - sig[:-1, :])
    change_events = (changes > 0).astype(xp.float32)
    costs = cost_per_change * change_events.sum(axis=0)
    net = gross.sum(axis=0) - costs
    trade_events = change_events.sum(axis=0)

    mean_r = gross.mean(axis=0)
    std_r = gross.std(axis=0) + 1e-12
    periods_per_year = 365.25 * 24.0 * 3600.0 / float(tf.seconds)
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year)
    hit_rate = (gross > 0).mean(axis=0)

    if xp.__name__ == "cupy":
        import cupy as cp  # type: ignore

        return (
            cp.asnumpy(net),
            cp.asnumpy(sharpe),
            cp.asnumpy(trade_events),
            cp.asnumpy(hit_rate),
        )
    return (
        np.asarray(net),
        np.asarray(sharpe),
        np.asarray(trade_events),
        np.asarray(hit_rate),
    )


def _breakout_grid(df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
    lookbacks = [20, 30, 40, 50, 60, 80]
    breakouts = [0.0, 0.0001, 0.0002, 0.0003]
    n = len(df)
    combos: List[Dict] = []
    signals: List[np.ndarray] = []

    for lb in lookbacks:
        upper = df["high"].rolling(lb).max().shift(1).to_numpy()
        lower = df["low"].rolling(lb).min().shift(1).to_numpy()
        close = df["close"].to_numpy()
        for min_b in breakouts:
            long_mask = close > (upper * (1.0 + min_b))
            short_mask = close < (lower * (1.0 - min_b))
            sig = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0)).astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            signals.append(sig)
            combos.append(
                {
                    "strategy_name": "Breakout",
                    "config_name": f"Breakout_L{lb}_B{min_b}",
                    "params": {"lookback": lb, "min_breakout_pct": min_b},
                }
            )

    if not signals:
        return np.zeros((n, 0), dtype=np.float32), combos
    return np.column_stack(signals), combos


def _mean_reversion_grid(df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
    sma_periods = [20, 30, 50, 80]
    deviations = [0.0015, 0.0020, 0.0025, 0.0030]
    n = len(df)
    combos: List[Dict] = []
    signals: List[np.ndarray] = []
    close = df["close"]

    for p in sma_periods:
        sma = close.rolling(p).mean().to_numpy()
        px = close.to_numpy()
        dev = (px - sma) / np.where(sma == 0, np.nan, sma)
        for d in deviations:
            long_mask = dev < -d
            short_mask = dev > d
            sig = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0)).astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            signals.append(sig)
            combos.append(
                {
                    "strategy_name": "MeanReversion",
                    "config_name": f"MeanRev_SMA{p}_D{d}",
                    "params": {"sma_period": p, "deviation_pct": d},
                }
            )

    if not signals:
        return np.zeros((n, 0), dtype=np.float32), combos
    return np.column_stack(signals), combos


def _ema_pullback_grid(df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
    fast_slow_pairs = [(20, 100), (20, 200), (50, 200), (30, 120)]
    pullbacks = [0.0005, 0.0008, 0.0010, 0.0012]
    n = len(df)
    combos: List[Dict] = []
    signals: List[np.ndarray] = []
    close = df["close"].to_numpy()

    for fast, slow in fast_slow_pairs:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean().to_numpy()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean().to_numpy()
        trend_long = ema_fast > ema_slow
        trend_short = ema_fast < ema_slow
        for pb in pullbacks:
            pullback_long = close < (ema_fast * (1.0 - pb))
            pullback_short = close > (ema_fast * (1.0 + pb))
            long_mask = trend_long & pullback_long
            short_mask = trend_short & pullback_short
            sig = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0)).astype(np.float32)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            signals.append(sig)
            combos.append(
                {
                    "strategy_name": "EMATrendPullback",
                    "config_name": f"EMAPullback_{fast}_{slow}_P{pb}",
                    "params": {"fast_period": fast, "slow_period": slow, "pullback_pct": pb},
                }
            )

    if not signals:
        return np.zeros((n, 0), dtype=np.float32), combos
    return np.column_stack(signals), combos


def _build_matrix_and_combos(df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
    mats: List[np.ndarray] = []
    combos_all: List[Dict] = []
    for grid_fn in (_breakout_grid, _mean_reversion_grid, _ema_pullback_grid):
        mat, combos = grid_fn(df)
        mats.append(mat)
        combos_all.extend(combos)
    if not mats:
        return np.zeros((len(df), 0), dtype=np.float32), combos_all
    return np.concatenate(mats, axis=1), combos_all


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    if not (0.5 <= args.train_ratio <= 0.9):
        raise SystemExit("--train-ratio must be between 0.5 and 0.9")
    if args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")

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
    if len(df) < 400:
        raise SystemExit(f"Not enough bars for pre-screening: {len(df)}")

    split = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split].copy()
    if len(train_df) < 300:
        raise SystemExit("Train split is too short for reliable prescreen.")

    xp, backend = _select_backend(args.gpu)
    matrix, combos = _build_matrix_and_combos(train_df)
    if matrix.shape[1] == 0:
        raise SystemExit("No strategy combos generated.")

    close = train_df["close"].to_numpy(dtype=np.float64)
    returns = (close[1:] / close[:-1]) - 1.0
    cost_per_change = _estimate_cost_per_position_change(
        instrument=args.instrument,
        pricing_model=args.pricing_model,
        slippage_pips=float(args.slippage_pips),
        core_commission_per_10k=float(args.core_commission_per_10k),
    )
    net, sharpe, trades, hit_rate = _score_signal_matrix(
        xp=xp,
        signals=matrix,
        returns=returns,
        tf=tf,
        cost_per_change=cost_per_change,
    )

    score = net + (0.10 * np.maximum(sharpe, 0.0)) + (0.02 * np.minimum(trades, 300.0))
    rows: List[Dict] = []
    for i, combo in enumerate(combos):
        rows.append(
            {
                "instrument": args.instrument,
                "tf": args.tf,
                "strategy_name": combo["strategy_name"],
                "config_name": combo["config_name"],
                "score": float(score[i]),
                "net_proxy": float(net[i]),
                "sharpe_proxy": float(sharpe[i]),
                "trade_events_proxy": int(trades[i]),
                "hit_rate_proxy": float(hit_rate[i]),
                "params": json.dumps(combo["params"]),
            }
        )

    rows.sort(key=lambda r: (r["score"], r["net_proxy"], r["sharpe_proxy"]), reverse=True)
    top_rows = rows[: args.top_n]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    base = f"gpu_prescreener_{args.instrument}_{args.tf}_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    shortlist_path = out_dir / f"{base}_shortlist.csv"
    md_path = out_dir / f"{base}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "instrument",
            "tf",
            "strategy_name",
            "config_name",
            "score",
            "net_proxy",
            "sharpe_proxy",
            "trade_events_proxy",
            "hit_rate_proxy",
            "params",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with shortlist_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "strategy_name",
            "config_name",
            "score",
            "net_proxy",
            "sharpe_proxy",
            "trade_events_proxy",
            "hit_rate_proxy",
            "params",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in top_rows:
            writer.writerow({k: r[k] for k in fields})

    lines = [
        "# GPU Pre-Screener",
        "",
        f"Generated: {_utc_now().isoformat()}",
        f"Instrument: {args.instrument}",
        f"Timeframe: {args.tf}",
        f"Backend: {backend}",
        f"Train bars: {len(train_df)}",
        f"Total combos scored: {len(rows)}",
        f"Top-N kept: {len(top_rows)}",
        "",
        "## Top Candidates",
        "",
        "| Rank | Strategy | Config | Score | Net Proxy | Sharpe Proxy | Trades Proxy | Hit Rate |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(top_rows, 1):
        lines.append(
            f"| {i} | {r['strategy_name']} | {r['config_name']} | {r['score']:.6f} | "
            f"{r['net_proxy']:.6f} | {r['sharpe_proxy']:.3f} | {r['trade_events_proxy']} | {r['hit_rate_proxy']:.2%} |"
        )
    lines.append("")
    lines.append("Use the shortlist CSV to reduce full cost-aware walk-forward backtests.")
    lines.append(f"All combos CSV: `{csv_path}`")
    lines.append(f"Shortlist CSV: `{shortlist_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Backend: {backend}")
    print(f"Combos scored: {len(rows)}")
    print(f"All CSV: {csv_path}")
    print(f"Shortlist CSV: {shortlist_path}")
    print(f"Report: {md_path}")
    if top_rows:
        top = top_rows[0]
        print(
            "Top candidate: "
            f"{top['strategy_name']} / {top['config_name']} "
            f"score={top['score']:.6f} net={top['net_proxy']:.6f} sharpe={top['sharpe_proxy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
