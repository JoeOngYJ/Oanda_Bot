#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
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
import torch
from torch.utils.data import DataLoader, TensorDataset

from oanda_bot.ml.models.two_stage import OpportunityTCNModel, eval_opportunity, seed_everything, train_opportunity_epoch
from scripts.two_stage_training_common import HORIZON_BARS, split_masks_with_embargo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train opportunity model (two-stage pipeline).")
    p.add_argument("--dataset", default="data/research/xau_two_stage_dataset_20240101_20260301.npz")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _make_loader(seq: np.ndarray, ctx: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(seq, dtype=torch.float32),
        torch.as_tensor(ctx, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    z = np.load(args.dataset, allow_pickle=True)
    seq = z["seq"].astype(np.float32)
    ctx = z["ctx"].astype(np.float32)
    y = z["y_opportunity"].astype(np.float32)
    ts = z["timestamps"]

    masks = split_masks_with_embargo(ts, horizon_bars=HORIZON_BARS)
    tr, va = masks["train"], masks["val"]
    if tr.sum() == 0 or va.sum() == 0:
        raise RuntimeError("No train/val rows after split+embargo.")

    tr_loader = _make_loader(seq[tr], ctx[tr], y[tr], args.batch_size, shuffle=True)
    va_loader = _make_loader(seq[va], ctx[va], y[va], args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpportunityTCNModel(seq_features=seq.shape[2], ctx_dim=ctx.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_losses = [train_opportunity_epoch(model, b, opt, device, grad_clip=1.0) for b in tr_loader]
        va_losses = [eval_opportunity(model, b, device) for b in va_loader]
        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"epoch={epoch} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("early_stop=1")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"opportunity_{stamp}.pt"
    cfg_path = out_dir / f"opportunity_{stamp}.json"
    torch.save(model.state_dict(), model_path)
    cfg = {
        "dataset": args.dataset,
        "model_path": str(model_path),
        "seq_features": int(seq.shape[2]),
        "ctx_dim": int(ctx.shape[1]),
        "best_val_loss": float(best),
        "history": history,
        "split_counts": {"train": int(tr.sum()), "val": int(va.sum())},
        "horizon_bars": HORIZON_BARS,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"saved_model={model_path}")
    print(f"saved_config={cfg_path}")


if __name__ == "__main__":
    main()
