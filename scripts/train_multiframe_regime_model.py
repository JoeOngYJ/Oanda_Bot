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

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-timeframe regime model.")
    p.add_argument("--instruments", default="XAU_USD,EUR_USD,GBP_USD")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--base-tf", default="M15")
    p.add_argument("--htf-1", default="H1")
    p.add_argument("--htf-2", default="H4")
    p.add_argument("--htf-3", default="D1")
    p.add_argument("--regimes", type=int, default=4)
    p.add_argument("--kmeans-iter", type=int, default=40)
    p.add_argument("--kmeans-restarts", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-lag-bars", type=int, default=1)
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--output-dir", default="data/research")
    return p.parse_args()


def _kmeans_plus_plus_init_numpy(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(np.sum(closest))
        if total <= 0:
            idx = int(rng.integers(0, n))
        else:
            probs = closest / total
            idx = int(rng.choice(n, p=probs))
        centers[i] = x[idx]
        d = np.sum((x - centers[i]) ** 2, axis=1)
        closest = np.minimum(closest, d)
    return centers


def _kmeans_numpy_once(
    x: np.ndarray,
    k: int,
    iters: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    centers = _kmeans_plus_plus_init_numpy(x, k, rng)
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
    final_dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = final_dists.argmin(axis=1)
    inertia = float(np.take_along_axis(final_dists, labels[:, None], axis=1).sum())
    return labels, centers, inertia


def _kmeans_numpy(
    x: np.ndarray,
    k: int,
    iters: int,
    seed: int,
    restarts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best = None
    n_runs = max(1, int(restarts))
    for _ in range(n_runs):
        labels, centers, inertia = _kmeans_numpy_once(x, k, iters, rng)
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    assert best is not None
    return best[0], best[1]


def _kmeans_cupy(
    x: np.ndarray,
    k: int,
    iters: int,
    seed: int,
    restarts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore

    cp.random.seed(seed)
    xg = cp.asarray(x, dtype=cp.float32)
    best_labels = None
    best_centers = None
    best_inertia = None
    n_runs = max(1, int(restarts))
    for _ in range(n_runs):
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
        dists = cp.sum((xg[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = cp.argmin(dists, axis=1)
        inertia = float(cp.take_along_axis(dists, labels[:, None], axis=1).sum().get())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    assert best_labels is not None and best_centers is not None
    return cp.asnumpy(best_labels), cp.asnumpy(best_centers)


def _rolling_mean(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if len(a) < n:
        return out
    c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1 :] = (c[n:] - c[:-n]) / float(n)
    return out


def _rolling_std(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) < n:
        return np.full_like(a, np.nan, dtype=np.float64)
    # Vectorized rolling std is far faster than per-step Python loops.
    return pd.Series(a).rolling(n).std(ddof=0).to_numpy(dtype=np.float64)


def _ema(a: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / (n + 1.0)
    out = np.zeros_like(a, dtype=np.float64)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def _feature_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    o = df["open"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)
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
    body_pct = np.where(o == 0, np.nan, (c - o) / o)
    hl_range_pct = np.where(c == 0, np.nan, (h - l) / c)
    vol_ma20 = _rolling_mean(v, 20)
    vol_std20 = _rolling_std(v, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z = (v - vol_ma20) / vol_std20
    vol_z = np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)

    out_cols = {
        f"{prefix}_ret1": r1,
        f"{prefix}_ret4": r4,
        f"{prefix}_atr_pct": np.where(c == 0, np.nan, atr / c),
        f"{prefix}_trend": trend,
        f"{prefix}_bbw": bbw,
        f"{prefix}_body_pct": body_pct,
        f"{prefix}_range_pct": hl_range_pct,
        f"{prefix}_vol_z": vol_z,
    }
    if prefix == "m15":
        # Session behavior features (UTC) for session-aware regime conditioning.
        idx = pd.DatetimeIndex(df.index)
        hour = idx.hour.to_numpy(dtype=np.float64)
        wday = idx.dayofweek.to_numpy(dtype=np.float64)
        asia = ((hour >= 0) & (hour < 7)).astype(np.float64)
        europe = ((hour >= 7) & (hour < 13)).astype(np.float64)
        us = ((hour >= 13) & (hour < 22)).astype(np.float64)
        eu_us_overlap = ((hour >= 13) & (hour < 17)).astype(np.float64)
        out_cols.update(
            {
                f"{prefix}_sess_asia": asia,
                f"{prefix}_sess_europe": europe,
                f"{prefix}_sess_us": us,
                f"{prefix}_sess_eu_us_overlap": eu_us_overlap,
                f"{prefix}_hour_sin": np.sin((2.0 * np.pi * hour) / 24.0),
                f"{prefix}_hour_cos": np.cos((2.0 * np.pi * hour) / 24.0),
                f"{prefix}_wday_sin": np.sin((2.0 * np.pi * wday) / 7.0),
                f"{prefix}_wday_cos": np.cos((2.0 * np.pi * wday) / 7.0),
            }
        )

    out = pd.DataFrame(
        out_cols,
        index=df.index,
    )
    return out


def _merge_multiframe_features(
    data_dict: Dict[Timeframe, pd.DataFrame],
    base_tf: Timeframe,
    h1_tf: Timeframe,
    h2_tf: Timeframe,
    h3_tf: Timeframe,
    lag_bars: int,
):
    base = data_dict[base_tf].sort_index()
    h1 = data_dict[h1_tf].sort_index().reindex(base.index, method="ffill")
    h2 = data_dict[h2_tf].sort_index().reindex(base.index, method="ffill")
    h3 = data_dict[h3_tf].sort_index().reindex(base.index, method="ffill")
    f_base = _feature_frame(base, "m15")
    f_h1 = _feature_frame(h1, "h1")
    f_h2 = _feature_frame(h2, "h4")
    f_h3 = _feature_frame(h3, "d1")
    feat = pd.concat([f_base, f_h1, f_h2, f_h3], axis=1)
    if lag_bars > 0:
        # Enforce closed-bar-only features to avoid forward contamination.
        feat = feat.shift(int(lag_bars))
    feat = feat.dropna()
    return feat


def _heuristic_regime_strategy_mapping(centers: np.ndarray, feature_columns: List[str]) -> Dict[str, str]:
    idx = {name: i for i, name in enumerate(feature_columns)}
    trend_scores = []
    vol_scores = []
    for r in range(centers.shape[0]):
        c = centers[r]
        trend = c[idx.get("h1_trend", 0)] + c[idx.get("h4_trend", 0)] + c[idx.get("d1_trend", 0)]
        vol = c[idx.get("m15_atr_pct", 0)] + c[idx.get("m15_bbw", 0)]
        trend_scores.append((r, float(trend)))
        vol_scores.append((r, float(vol)))

    mapping = {str(r): "MeanReversion" for r in range(centers.shape[0])}
    if trend_scores:
        trend_regime = max(trend_scores, key=lambda x: x[1])[0]
        mapping[str(trend_regime)] = "EMATrendPullback"
    if vol_scores:
        vol_ranked = [r for r, _ in sorted(vol_scores, key=lambda x: x[1], reverse=True)]
        breakout_regime = next((r for r in vol_ranked if mapping[str(r)] == "MeanReversion"), vol_ranked[0])
        mapping[str(breakout_regime)] = "Breakout"
    return mapping


def _regime_stability_metrics(labels: np.ndarray, k: int) -> Dict[str, object]:
    labels = labels.astype(np.int64, copy=False)
    trans = np.zeros((k, k), dtype=np.int64)
    for i in range(1, len(labels)):
        a = int(labels[i - 1])
        b = int(labels[i])
        if 0 <= a < k and 0 <= b < k:
            trans[a, b] += 1

    trans_probs = np.zeros((k, k), dtype=np.float64)
    row_sum = trans.sum(axis=1, keepdims=True)
    nz = row_sum[:, 0] > 0
    trans_probs[nz] = trans[nz] / row_sum[nz]

    durations_by_regime: Dict[str, List[int]] = {str(i): [] for i in range(k)}
    if len(labels):
        run_label = int(labels[0])
        run_len = 1
        for i in range(1, len(labels)):
            cur = int(labels[i])
            if cur == run_label:
                run_len += 1
            else:
                if 0 <= run_label < k:
                    durations_by_regime[str(run_label)].append(run_len)
                run_label = cur
                run_len = 1
        if 0 <= run_label < k:
            durations_by_regime[str(run_label)].append(run_len)

    avg_dur = {
        r: (float(np.mean(v)) if v else 0.0)
        for r, v in durations_by_regime.items()
    }
    counts = {str(i): int((labels == i).sum()) for i in range(k)}
    return {
        "transition_counts": trans.tolist(),
        "transition_probs": trans_probs.tolist(),
        "avg_regime_duration_bars": avg_dur,
        "regime_counts": counts,
    }


def main() -> int:
    args = parse_args()
    instruments = [x.strip() for x in args.instruments.split(",") if x.strip()]
    base_tf = Timeframe.from_oanda_granularity(args.base_tf)
    h1_tf = Timeframe.from_oanda_granularity(args.htf_1)
    h2_tf = Timeframe.from_oanda_granularity(args.htf_2)
    h3_tf = Timeframe.from_oanda_granularity(args.htf_3)
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
            timeframes=[base_tf, h1_tf, h2_tf, h3_tf],
        )
        feat = _merge_multiframe_features(
            data, base_tf, h1_tf, h2_tf, h3_tf, lag_bars=int(args.feature_lag_bars)
        )
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
            labels, centers = _kmeans_cupy(
                x, args.regimes, args.kmeans_iter, args.seed, args.kmeans_restarts
            )
            backend = "gpu(cupy)"
        except Exception:
            if args.gpu == "on":
                raise
            labels, centers = _kmeans_numpy(
                x, args.regimes, args.kmeans_iter, args.seed, args.kmeans_restarts
            )
    else:
        labels, centers = _kmeans_numpy(
            x, args.regimes, args.kmeans_iter, args.seed, args.kmeans_restarts
        )

    mapping = _heuristic_regime_strategy_mapping(centers, feature_columns)
    stability = _regime_stability_metrics(labels, int(args.regimes))
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
        "feature_lag_bars": int(args.feature_lag_bars),
        "kmeans_restarts": int(args.kmeans_restarts),
        "regime_stability": stability,
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
    print(f"Feature lag bars: {int(args.feature_lag_bars)}")
    print(f"KMeans restarts: {int(args.kmeans_restarts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
