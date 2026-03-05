#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "PyTorch is required for walkforward training. Install torch in the active environment."
    ) from exc

from oanda_bot.ml.training.two_stage_walkforward import (
    DEFAULT_END,
    DEFAULT_START,
    WalkforwardWindow,
    build_samples,
    calibrate_gate_from_validation,
    evaluate_test_slice,
    generate_windows,
    load_sample_cache,
    load_or_ensure_triplet,
    make_signals_frame,
    predict_probabilities,
    save_json,
    save_sample_cache,
    seed_everything,
    split_masks_for_window,
    train_direction_model,
    train_opportunity_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward train/eval for two-stage XAU_USD models.")
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--data-dir", default="data/backtesting")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--horizon-bars", type=int, default=8)
    p.add_argument("--no-trade-band", type=float, default=0.30)
    p.add_argument("--preprocess-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--val-months", type=int, default=2)
    p.add_argument("--test-months", type=int, default=2)
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gate", type=float, default=0.60)
    p.add_argument("--target-trade-rate", type=float, default=0.20)
    p.add_argument("--opp-band-quantile", type=float, default=0.70)
    p.add_argument("--opp-atr-k", type=float, default=0.30)
    p.add_argument("--disable-vol-aware-opp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", default="data/research/wf_xau_two_stage_cache_20240101_20260301.npz")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--models-dir", default="models/wf")
    p.add_argument("--report-csv", default="reports/walkforward_results.csv")
    p.add_argument("--report-json", default="reports/walkforward_results.json")
    p.add_argument("--max-steps", type=int, default=0, help="0 means all steps.")
    return p.parse_args()


def _window_dict(w: WalkforwardWindow) -> Dict[str, str]:
    return {
        "train_start": str(w.train_start),
        "train_end": str(w.train_end),
        "val_start": str(w.val_start),
        "val_end": str(w.val_end),
        "test_start": str(w.test_start),
        "test_end": str(w.test_end),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start != DEFAULT_START or end != DEFAULT_END:
        print(f"using_range={start}..{end}")
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild_cache:
        print(f"loading_cache={cache_path}")
        bundle = load_sample_cache(cache_path)
    else:
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
        save_sample_cache(
            bundle,
            cache_path,
            instrument=args.instrument,
            start=start,
            end=end,
            seq_len=args.seq_len,
            horizon_bars=args.horizon_bars,
            no_trade_band=args.no_trade_band,
        )
        print(f"saved_cache={cache_path}")

    windows = generate_windows(
        start=start,
        end=end,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    if args.max_steps > 0:
        windows = windows[: args.max_steps]
    if not windows:
        raise RuntimeError("No walk-forward windows generated for provided date range.")

    models_root = Path(args.models_dir)
    reports_csv = Path(args.report_csv)
    reports_json = Path(args.report_json)
    models_root.mkdir(parents=True, exist_ok=True)
    reports_csv.parent.mkdir(parents=True, exist_ok=True)
    reports_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: List[Dict[str, object]] = []
    print("direction_head_order=[long,short] eval_pred_dir=(p_long > p_short -> long=1)")

    for w in windows:
        masks = split_masks_for_window(bundle.timestamps, w, horizon_bars=args.horizon_bars)
        tr, va, te = masks["train"], masks["val"], masks["test"]
        n_tr, n_va, n_te = int(tr.sum()), int(va.sum()), int(te.sum())

        if n_tr < 256 or n_va < 64 or n_te < 64:
            print(f"skip {w.step_id}: insufficient samples train={n_tr} val={n_va} test={n_te}")
            continue

        print(f"step={w.step_id} train={n_tr} val={n_va} test={n_te}")

        # Vol-aware opportunity labeling:
        # y_opp = 1 if abs(net_ret) > max(band_train, k * ATR), with band from train split quantile.
        if np.isnan(bundle.net_ret).all():
            raise RuntimeError(
                "Cached bundle is missing net_ret. Rebuild cache with the latest build_walkforward_cache.py."
            )
        net_train = bundle.net_ret[tr]
        atr_train = bundle.atr[tr]
        abs_net_train = np.abs(net_train[np.isfinite(net_train)])
        if abs_net_train.size == 0:
            band_train = float(args.no_trade_band)
        else:
            q = float(np.clip(args.opp_band_quantile, 0.01, 0.99))
            band_train = float(np.quantile(abs_net_train, q))

        abs_net_all = np.abs(bundle.net_ret)
        atr_all = np.nan_to_num(bundle.atr, nan=0.0, posinf=0.0, neginf=0.0)
        if args.disable_vol_aware_opp:
            thresh_all = np.full_like(abs_net_all, fill_value=max(float(args.no_trade_band), band_train), dtype=np.float32)
        else:
            thresh_all = np.maximum(max(float(args.no_trade_band), band_train), float(args.opp_atr_k) * atr_all)

        y_opp_all = (abs_net_all > thresh_all).astype(np.float32)
        y_dir_all = np.where(y_opp_all == 1.0, (bundle.net_ret > 0.0).astype(np.int64), -1).astype(np.int64)

        def _split_balance(name: str, m: np.ndarray) -> None:
            y_opp = y_opp_all[m]
            y_dir = y_dir_all[m]
            opp_pos = float(np.mean(y_opp == 1.0)) if y_opp.size else float("nan")
            valid = y_dir != -1
            if np.any(valid):
                long_pct = float(np.mean(y_dir[valid] == 1))
                short_pct = float(np.mean(y_dir[valid] == 0))
            else:
                long_pct = float("nan")
                short_pct = float("nan")
            print(
                f"label_balance[{name}] opp_pos={opp_pos:.4f} "
                f"dir_long={long_pct:.4f} dir_short={short_pct:.4f}"
            )

        _split_balance("train", tr)
        _split_balance("val", va)
        _split_balance("test", te)

        opp_model, opp_summary = train_opportunity_model(
            bundle.seq[tr],
            bundle.ctx[tr],
            y_opp_all[tr],
            bundle.seq[va],
            bundle.ctx[va],
            y_opp_all[va],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=device,
        )

        dir_model, dir_summary = train_direction_model(
            bundle.seq[tr],
            bundle.ctx[tr],
            y_dir_all[tr],
            bundle.seq[va],
            bundle.ctx[va],
            y_dir_all[va],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=device,
        )

        p_val_trade, p_val_long, p_val_short = predict_probabilities(
            opp_model,
            dir_model,
            bundle.seq[va],
            bundle.ctx[va],
            device=device,
        )
        gate_calibrated = calibrate_gate_from_validation(
            p_val_trade,
            target_trade_rate=float(args.target_trade_rate),
        )
        val_trade_rate = float(np.mean(p_val_trade > gate_calibrated)) if p_val_trade.size else float("nan")
        print(
            f"gate_calibrated={gate_calibrated:.6f} "
            f"target_trade_rate={float(args.target_trade_rate):.4f} "
            f"val_trade_rate={val_trade_rate:.4f}"
        )

        p_trade, p_long, p_short = predict_probabilities(
            opp_model,
            dir_model,
            bundle.seq[te],
            bundle.ctx[te],
            device=device,
        )

        metrics = evaluate_test_slice(
            p_trade,
            p_long,
            p_short,
            y_opp_all[te],
            y_dir_all[te],
            gate=gate_calibrated,
        )

        step_dir = models_root / w.step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        torch.save(opp_model.state_dict(), step_dir / "opportunity.pt")
        torch.save(dir_model.state_dict(), step_dir / "direction.pt")

        signals = make_signals_frame(bundle.timestamps[te], p_trade, p_long, p_short)
        signals.to_parquet(step_dir / "test_signals.parquet", index=False)

        meta = {
            "step_id": w.step_id,
            "instrument": args.instrument,
            "window": _window_dict(w),
            "horizon_bars": args.horizon_bars,
            "no_trade_band": args.no_trade_band,
            "gate": gate_calibrated,
            "target_trade_rate": float(args.target_trade_rate),
            "opp_band_quantile": float(args.opp_band_quantile),
            "opp_atr_k": float(args.opp_atr_k),
            "vol_aware_opp": bool(not args.disable_vol_aware_opp),
            "train_config": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
            },
            "schema": {
                "seq_len": int(bundle.seq.shape[1]),
                "seq_features": int(bundle.seq.shape[2]),
                "ctx_dim": int(bundle.ctx.shape[1]),
                "feature_columns": bundle.feature_columns,
            },
            "counts": {"train": n_tr, "val": n_va, "test": n_te},
            "losses": {
                "opp_val_loss": float(opp_summary.best_val_loss),
                "dir_val_loss": float(dir_summary.best_val_loss),
                "opp_epochs": int(opp_summary.epochs_ran),
                "dir_epochs": int(dir_summary.epochs_ran),
            },
            "metrics": metrics,
            "label_balance": {
                "train_opp_pos": float(np.mean(y_opp_all[tr] == 1.0)),
                "val_opp_pos": float(np.mean(y_opp_all[va] == 1.0)),
                "test_opp_pos": float(np.mean(y_opp_all[te] == 1.0)),
            },
            "artifacts": {
                "opportunity_model": str(step_dir / "opportunity.pt"),
                "direction_model": str(step_dir / "direction.pt"),
                "test_signals": str(step_dir / "test_signals.parquet"),
            },
            "trading_eval": {
                "status": "placeholder",
                "note": "Saved per-bar probability signals for later backtester integration.",
            },
        }
        save_json(step_dir / "meta.json", meta)

        row: Dict[str, object] = {
            "step_id": w.step_id,
            **_window_dict(w),
            "opp_val_loss": float(opp_summary.best_val_loss),
            "dir_val_loss": float(dir_summary.best_val_loss),
            "test_trade_rate": metrics["test_trade_rate"],
            "test_dir_acc": metrics["test_dir_acc"],
            "test_mean_p_trade": metrics["test_mean_p_trade"],
            "test_direction_balance": metrics["test_direction_balance"],
            "test_opp_auc": metrics["test_opp_auc"],
            "gate_used": gate_calibrated,
            "val_trade_rate_at_gate": val_trade_rate,
            "train_opp_pos_rate": float(np.mean(y_opp_all[tr] == 1.0)),
            "val_opp_pos_rate": float(np.mean(y_opp_all[va] == 1.0)),
            "test_opp_pos_rate": float(np.mean(y_opp_all[te] == 1.0)),
            "test_signal_path": str(step_dir / "test_signals.parquet"),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("All steps were skipped due to insufficient data after split/purge.")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(reports_csv, index=False)

    summary = {
        "instrument": args.instrument,
        "start": str(start),
        "end": str(end),
        "num_steps": int(len(result_df)),
        "models_dir": str(models_root),
        "report_csv": str(reports_csv),
        "rows": rows,
    }
    save_json(reports_json, summary)

    print(f"saved_csv={reports_csv}")
    print(f"saved_json={reports_json}")
    print(f"steps_trained={len(result_df)}")


if __name__ == "__main__":
    main()
