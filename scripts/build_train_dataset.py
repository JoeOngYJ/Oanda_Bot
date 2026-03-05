#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.two_stage_training_common import (
    HORIZON_BARS,
    INSTRUMENT,
    TEST_END,
    TRAIN_START,
    add_ohlc_fallback_costs,
    build_feature_label_table,
    load_or_ensure_ohlcv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cached train dataset for two-stage M15 models.")
    p.add_argument("--instrument", default=INSTRUMENT)
    p.add_argument("--no-trade-band", type=float, default=0.30)
    p.add_argument("--horizon-bars", type=int, default=HORIZON_BARS)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--out", default="data/research/xau_two_stage_dataset_20240101_20260301.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    m15, h1, h4 = load_or_ensure_ohlcv(args.instrument)
    data = build_feature_label_table(
        m15,
        h1,
        h4,
        instrument=args.instrument,
        horizon_bars=args.horizon_bars,
        no_trade_band=args.no_trade_band,
        seq_len=args.seq_len,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    meta = {
        "instrument": args.instrument,
        "rows": int(len(data["timestamps"])),
        "seq_shape": list(data["seq"].shape),
        "ctx_shape": list(data["ctx"].shape),
        "horizon_bars": args.horizon_bars,
        "no_trade_band": args.no_trade_band,
        "date_range": {"start": str(TRAIN_START), "end": str(TEST_END)},
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved_npz={out}")
    print(f"saved_meta={meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
