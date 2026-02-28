#!/usr/bin/env python3
"""Train multi-timeframe regime model (GPU-first, CPU fallback)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.core.timeframe import Timeframe
from backtesting.data.manager import DataManager


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-timeframe regime model.")
    p.add_argument("--instruments", default="XAU_USD,EUR_USD,GBP_USD")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--base-tf", default="M15")
    p.add_argument("--htf-1", default="H1")
    p.add_argument("--htf-2", default="H4")
    p.add_argument("--regimes", type=int, default=4)
    p.add_argument("--kmeans-iter", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--output-dir", default="data/research")
    return p.parse_args()


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
            new_centers[j] = cp.mean(points, axis=0) if points.shape[0] else centers[j]
        if cp.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return cp.asnumpy(labels), cp.asnumpy(centers)


def _rolling_mean(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if len(a) < n:
        return out
    c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1 :] = (c[n:] - c[:-n]) / float(n)
    return out


def _rolling_std(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if len(a) < n:
        return out
    for i in range(n - 1, len(a)):
        out[i] = np.std(a[i - n + 1 : i + 1])
    return out


def _ema(a: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = np.zeros_like(a, dtype=np.float64)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def _feature_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr = _rolling_mean(tr, 14)
    r1 = np.where(np.roll(c, 1) == 0, np.nan, c / np.roll(c, 1) - 1.0)
    r4 = np.where(np.roll(c, 4) == 0, np.nan, c / np.roll(c, 4) - 1.0)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    trend = np.where(c == 0, np.nan, (ema20 - ema50) / c)
    sma20 = _rolling_mean(c, 20)
    std20 = _rolling_std(c, 20)
    bbw = np.where(sma20 == 0, np.nan, (2.0 * 2.0 * std20) / sma20)
    out = pd.DataFrame(
        {
            f"{prefix}_ret1": r1,
            f"{prefix}_ret4": r4,
            f"{prefix}_atr_pct": np.where(c == 0, np.nan, atr / c),
            f"{prefix}_trend": trend,
            f"{prefix}_bbw": bbw,
        },
        index=df.index,
    )
    return out


def _merge_multiframe_features(data_dict: Dict[Timeframe, pd.DataFrame], base_tf: Timeframe, h1_tf: Timeframe, h2_tf: Timeframe):
    base = data_dict[base_tf].sort_index()
    h1 = data_dict[h1_tf].sort_index().reindex(base.index, method="ffill")
    h2 = data_dict[h2_tf].sort_index().reindex(base.index, method="ffill")
    f_base = _feature_frame(base, "m15")
    f_h1 = _feature_frame(h1, "h1")
    f_h2 = _feature_frame(h2, "h4")
    feat = pd.concat([f_base, f_h1, f_h2], axis=1).dropna()
    return feat


def _heuristic_regime_strategy_mapping(centers: np.ndarray, feature_columns: List[str]) -> Dict[str, str]:
    idx = {name: i for i, name in enumerate(feature_columns)}
    mapping = {}
    for r in range(centers.shape[0]):
        c = centers[r]
        trend = c[idx.get("h1_trend", 0)] + c[idx.get("h4_trend", 0)]
        vol = c[idx.get("m15_atr_pct", 0)] + c[idx.get("m15_bbw", 0)]
        if trend > 0.5 and vol > 0:
            strat = "EMATrendPullback"
        elif vol > 0.5:
            strat = "Breakout"
        else:
            strat = "MeanReversion"
        mapping[str(r)] = strat
    return mapping


def main() -> int:
    args = parse_args()
    instruments = [x.strip() for x in args.instruments.split(",") if x.strip()]
    base_tf = Timeframe.from_oanda_granularity(args.base_tf)
    h1_tf = Timeframe.from_oanda_granularity(args.htf_1)
    h2_tf = Timeframe.from_oanda_granularity(args.htf_2)
    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)

    dm = DataManager({})
    frames = []
    for inst in instruments:
        data = dm.ensure_data(
            instrument=inst,
            base_timeframe=base_tf,
            start_date=start,
            end_date=end,
            timeframes=[base_tf, h1_tf, h2_tf],
        )
        feat = _merge_multiframe_features(data, base_tf, h1_tf, h2_tf)
        feat["instrument"] = inst
        frames.append(feat)

    all_feat = pd.concat(frames, axis=0).sort_index()
    feature_columns = [c for c in all_feat.columns if c != "instrument"]
    xdf = all_feat[feature_columns]
    mu = xdf.mean()
    sd = xdf.std().replace(0, 1.0)
    x = ((xdf - mu) / sd).to_numpy(dtype=np.float64)

    backend = "cpu"
    if args.gpu in {"auto", "on"}:
        try:
            labels, centers = _kmeans_cupy(x, args.regimes, args.kmeans_iter, args.seed)
            backend = "gpu(cupy)"
        except Exception:
            if args.gpu == "on":
                raise
            labels, centers = _kmeans_numpy(x, args.regimes, args.kmeans_iter, args.seed)
    else:
        labels, centers = _kmeans_numpy(x, args.regimes, args.kmeans_iter, args.seed)

    mapping = _heuristic_regime_strategy_mapping(centers, feature_columns)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"multiframe_regime_model_{stamp}.json"
    labels_csv = out_dir / f"multiframe_regime_labels_{stamp}.csv"

    obj = {
        "backend": backend,
        "instruments": instruments,
        "base_tf": args.base_tf,
        "feature_columns": feature_columns,
        "train_mean": {k: float(v) for k, v in mu.items()},
        "train_std": {k: float(v) for k, v in sd.items()},
        "centers": centers.tolist(),
        "regime_to_strategy": mapping,
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    model_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    out_df = all_feat.copy()
    out_df["regime"] = labels
    out_df.to_csv(labels_csv)

    print(f"Backend: {backend}")
    print(f"Model JSON: {model_path}")
    print(f"Labels CSV: {labels_csv}")
    print(f"Regime->strategy: {mapping}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
