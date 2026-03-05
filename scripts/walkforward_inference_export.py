#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "PyTorch is required for walkforward inference export. Install torch in the active environment."
    ) from exc

from oanda_bot.ml.training.two_stage_walkforward import (
    build_samples,
    load_or_ensure_triplet,
    load_step_models,
    make_signals_frame,
    predict_probabilities,
    split_masks_for_window,
    WalkforwardWindow,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export walk-forward TEST probabilities for one step.")
    p.add_argument("--step-id", required=True, help="Step directory name under models/wf (e.g. step_000_20240101)")
    p.add_argument("--models-dir", default="models/wf")
    p.add_argument("--data-dir", default="data/backtesting")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--out", default="", help="Optional output parquet path. Defaults to reports/signals_<step_id>.parquet")
    p.add_argument("--gate", type=float, default=0.60, help="Gate threshold for signal_side generation")
    return p.parse_args()


def _load_window(step_dir: Path) -> WalkforwardWindow:
    meta_path = step_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json for step: {step_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    win = meta["window"]
    return WalkforwardWindow(
        step_id=str(meta["step_id"]),
        train_start=pd.Timestamp(win["train_start"]),
        train_end=pd.Timestamp(win["train_end"]),
        val_start=pd.Timestamp(win["val_start"]),
        val_end=pd.Timestamp(win["val_end"]),
        test_start=pd.Timestamp(win["test_start"]),
        test_end=pd.Timestamp(win["test_end"]),
    )


def main() -> None:
    args = parse_args()

    step_dir = Path(args.models_dir) / args.step_id
    window = _load_window(step_dir)

    meta = json.loads((step_dir / "meta.json").read_text(encoding="utf-8"))
    horizon_bars = int(meta["horizon_bars"])
    no_trade_band = float(meta["no_trade_band"])
    seq_len = int(meta["schema"]["seq_len"])
    seq_features = int(meta["schema"]["seq_features"])
    ctx_dim = int(meta["schema"]["ctx_dim"])
    gate = float(args.gate)

    pad = pd.Timedelta(minutes=seq_len * 15) + pd.Timedelta(days=3)
    data_start = window.train_start - pad

    m15, h1, h4 = load_or_ensure_triplet(
        instrument=args.instrument,
        data_dir=Path(args.data_dir),
        start=data_start,
        end=window.test_end,
    )

    bundle = build_samples(
        m15,
        h1,
        h4,
        seq_len=seq_len,
        horizon_bars=horizon_bars,
        no_trade_band=no_trade_band,
        start=window.train_start,
        end=window.test_end,
    )

    masks = split_masks_for_window(bundle.timestamps, window, horizon_bars=horizon_bars)
    te = masks["test"]
    if int(te.sum()) == 0:
        raise RuntimeError("No TEST samples for this step after purge/embargo.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opp_model, dir_model = load_step_models(step_dir, seq_features=seq_features, ctx_dim=ctx_dim, device=device)
    p_trade, p_long, p_short = predict_probabilities(
        opp_model,
        dir_model,
        bundle.seq[te],
        bundle.ctx[te],
        device=device,
    )

    active = p_trade > gate
    signal_side = np.where(active & (p_long > p_short), 1, np.where(active, -1, 0)).astype(np.int8)

    sig = make_signals_frame(bundle.timestamps[te], p_trade, p_long, p_short)
    sig["signal_side"] = signal_side

    optional_cols = [
        "y_opportunity",
        "y_direction",
        "gross_ret",
        "net_ret",
        "cost_est",
        "close",
        "atr",
    ]
    for col in optional_cols:
        if hasattr(bundle, col):
            arr = np.asarray(getattr(bundle, col))
            if arr.shape[0] == int(te.shape[0]):
                sig[col] = arr[te]

    p_trade_mean = float(np.mean(p_trade))
    p_trade_min = float(np.min(p_trade))
    p_trade_max = float(np.max(p_trade))
    trade_rate = float(np.mean(active))
    mean_p_long = float(np.mean(p_long))
    mean_p_short = float(np.mean(p_short))
    direction_balance = float(np.mean((p_long > p_short) & active))

    print(f"step_id={args.step_id}")
    print(f"rows={len(sig)}")
    print(f"test_range={window.test_start}..{window.test_end}")
    print(f"gate={gate:.4f}")
    print(f"p_trade_mean={p_trade_mean:.6f}")
    print(f"p_trade_min={p_trade_min:.6f}")
    print(f"p_trade_max={p_trade_max:.6f}")
    print(f"trade_rate={trade_rate:.6f}")
    print(f"mean_p_long={mean_p_long:.6f}")
    print(f"mean_p_short={mean_p_short:.6f}")
    print(f"direction_balance={direction_balance:.6f}")

    y_opp = sig["y_opportunity"].to_numpy() if "y_opportunity" in sig.columns else None
    y_dir = sig["y_direction"].to_numpy() if "y_direction" in sig.columns else None

    if y_opp is not None:
        y_opp_num = pd.to_numeric(pd.Series(y_opp), errors="coerce").to_numpy()
        opp_positive_rate = float(np.nanmean(y_opp_num == 1))
        gate_positive_rate = float(np.mean(active))
        print(f"opp_positive_rate={opp_positive_rate:.6f}")
        print(f"gate_positive_rate={gate_positive_rate:.6f}")

    if y_dir is not None:
        y_dir_num = pd.to_numeric(pd.Series(y_dir), errors="coerce")
        valid = y_dir_num.notna().to_numpy()
        valid = valid & (y_dir_num.to_numpy() != -1)
        if np.any(valid):
            y_valid = y_dir_num.to_numpy()[valid].astype(int)
            label_long_rate = float(np.mean(y_valid == 1))
            print(f"label_long_rate={label_long_rate:.6f}")
        else:
            print("label_long_rate=nan")

    if y_opp is not None and y_dir is not None:
        y_dir_num = pd.to_numeric(pd.Series(y_dir), errors="coerce")
        valid_dir = y_dir_num.notna().to_numpy()
        valid_dir = valid_dir & (y_dir_num.to_numpy() != -1)
        gated_valid = active & valid_dir
        if np.any(gated_valid):
            pred_dir = np.where(p_long > p_short, 1, 0).astype(int)
            dir_acc = float(np.mean(pred_dir[gated_valid] == y_dir_num.to_numpy()[gated_valid].astype(int)))
            print(f"dir_acc={dir_acc:.6f}")
        else:
            print("dir_acc=nan")

    out_path = Path(args.out) if args.out else Path("reports") / f"signals_{args.step_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(out_path, index=False)

    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
