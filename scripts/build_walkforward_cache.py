#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from oanda_bot.ml.training.two_stage_walkforward import (
    DEFAULT_END,
    DEFAULT_START,
    build_samples,
    load_or_ensure_triplet,
    save_sample_cache,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cached two-stage walk-forward sample bundle.")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--data-dir", default="data/backtesting")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--horizon-bars", type=int, default=8)
    p.add_argument("--no-trade-band", type=float, default=0.30)
    p.add_argument("--preprocess-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--out", default="data/research/wf_xau_two_stage_cache_20240101_20260301.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start != DEFAULT_START or end != DEFAULT_END:
        print(f"using_range={start}..{end}")

    m15, h1, h4 = load_or_ensure_triplet(
        instrument=args.instrument,
        data_dir=Path(args.data_dir),
        start=start,
        end=end,
    )

    bundle = build_samples(
        m15,
        h1,
        h4,
        seq_len=args.seq_len,
        horizon_bars=args.horizon_bars,
        no_trade_band=args.no_trade_band,
        start=start,
        end=end,
        preprocess_backend=args.preprocess_backend,
    )

    out = Path(args.out)
    save_sample_cache(
        bundle,
        out,
        instrument=args.instrument,
        start=start,
        end=end,
        seq_len=args.seq_len,
        horizon_bars=args.horizon_bars,
        no_trade_band=args.no_trade_band,
    )

    print(f"saved_cache={out}")
    print(f"samples={len(bundle.timestamps)} seq_shape={tuple(bundle.seq.shape)} ctx_shape={tuple(bundle.ctx.shape)}")


if __name__ == "__main__":
    main()
