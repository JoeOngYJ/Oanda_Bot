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

import numpy as np
import torch

from oanda_bot.ml.models.two_stage import DirectionTCNModel, OpportunityTCNModel
from scripts.two_stage_training_common import HORIZON_BARS, split_masks_with_embargo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test two-stage inference on TEST split.")
    p.add_argument("--dataset", default="data/research/xau_two_stage_dataset_20240101_20260301.npz")
    p.add_argument("--opportunity-model", default="")
    p.add_argument("--direction-model", default="")
    p.add_argument("--gate", type=float, default=0.60)
    p.add_argument("--examples", type=int, default=5)
    return p.parse_args()


def _latest(pattern: str) -> str:
    files = sorted(Path("models").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model files found for pattern: models/{pattern}")
    return str(files[-1])


def main() -> None:
    args = parse_args()
    z = np.load(args.dataset, allow_pickle=True)
    seq = z["seq"].astype(np.float32)
    ctx = z["ctx"].astype(np.float32)
    ts = z["timestamps"]
    masks = split_masks_with_embargo(ts, horizon_bars=HORIZON_BARS)
    te = masks["test"]
    if te.sum() == 0:
        raise RuntimeError("No TEST rows found after split.")

    seq_te = torch.as_tensor(seq[te], dtype=torch.float32)
    ctx_te = torch.as_tensor(ctx[te], dtype=torch.float32)
    ts_te = ts[te]

    opp_path = args.opportunity_model or _latest("opportunity_*.pt")
    dir_path = args.direction_model or _latest("direction_*.pt")

    opp = OpportunityTCNModel(seq_features=seq.shape[2], ctx_dim=ctx.shape[1])
    opp.load_state_dict(torch.load(opp_path, map_location="cpu"))
    opp.eval()

    direc = DirectionTCNModel(seq_features=seq.shape[2], ctx_dim=ctx.shape[1])
    direc.load_state_dict(torch.load(dir_path, map_location="cpu"))
    direc.eval()

    with torch.no_grad():
        p_trade = opp(seq_te, ctx_te).cpu().numpy()
        p_dir = direc(seq_te, ctx_te).cpu().numpy()

    p_long = p_dir[:, 0]
    p_short = p_dir[:, 1]
    active = p_trade >= float(args.gate)
    side = np.where(p_long >= p_short, 1, -1)
    long_active = np.sum((side == 1) & active)
    short_active = np.sum((side == -1) & active)
    direction_balance = float(long_active / max(1, long_active + short_active))

    print(f"opportunity_model={opp_path}")
    print(f"direction_model={dir_path}")
    print(f"test_rows={len(p_trade)}")
    print(f"avg_p_trade={float(np.mean(p_trade)):.6f}")
    print(f"trade_rate_gate_{args.gate:.2f}={float(np.mean(active)):.6f}")
    print(f"direction_balance_long_share={direction_balance:.6f}")
    print("examples:")
    for i in range(min(args.examples, len(p_trade))):
        print(
            f"{ts_te[i]} p_trade={p_trade[i]:.4f} p_long={p_long[i]:.4f} "
            f"p_short={p_short[i]:.4f} side={'BUY' if side[i]>0 else 'SELL'} active={int(active[i])}"
        )


if __name__ == "__main__":
    main()
