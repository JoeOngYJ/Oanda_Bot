#!/usr/bin/env python3
"""Train deploy-ready multiframe regime models per pair using the existing trainer.

- Calls scripts/train_multiframe_regime_model.py per symbol (single-instrument run)
- Keeps output schema identical to existing runtime model JSON
- Applies pair-specific regime->strategy mapping preferences
- Stores final active models under models/active/<PAIR>/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_multiframe_regime_model.py"


PAIR_PROFILE = {
    # prefer breakout+trend
    "EUR_USD": "breakout_trend",
    "USD_CAD": "breakout_trend",
    "GBP_JPY": "breakout_trend",
    # prefer breakout + range reversal behavior
    "GBP_USD": "breakout_reversal",
    "XAU_USD": "breakout_reversal",
    # prefer Asia range breakout
    "USD_JPY": "asia_breakout",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-pair deployable multiframe regime models.")
    p.add_argument("--pairs", default="EUR_USD,GBP_USD,USD_JPY,USD_CAD,GBP_JPY,XAU_USD")
    p.add_argument("--start", default="2022-01-01", help="Training window start (ISO date).")
    p.add_argument("--end", default="2026-02-28", help="Training window end (ISO date).")
    p.add_argument("--base-tf", default="M15")
    p.add_argument("--htf-1", default="H1")
    p.add_argument("--htf-2", default="H4")
    p.add_argument("--htf-3", default="D1")
    p.add_argument("--regimes", type=int, default=4)
    p.add_argument("--kmeans-iter", type=int, default=40)
    p.add_argument("--kmeans-restarts", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-lag-bars", type=int, default=1)
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="off")
    p.add_argument("--research-out", default="data/research")
    p.add_argument("--active-out", default="models/active")
    return p.parse_args()


def _center_scores(obj: Dict) -> Dict[int, Dict[str, float]]:
    cols: List[str] = list(obj.get("feature_columns", []))
    centers = obj.get("centers", [])
    idx = {c: i for i, c in enumerate(cols)}
    scores: Dict[int, Dict[str, float]] = {}
    for rid, c in enumerate(centers):
        trend = float(c[idx.get("h1_trend", 0)]) + float(c[idx.get("h4_trend", 0)]) + float(c[idx.get("d1_trend", 0)])
        vol = float(c[idx.get("m15_atr_pct", 0)]) + float(c[idx.get("m15_bbw", 0)])
        range_proxy = -abs(trend)
        scores[rid] = {"trend": trend, "vol": vol, "range": range_proxy}
    return scores


def _mapping_for_pair(pair: str, obj: Dict) -> Dict[str, str]:
    k = len(obj.get("centers", []))
    if k == 0:
        return {}
    scores = _center_scores(obj)
    ids = list(range(k))

    trend_id = max(ids, key=lambda r: scores[r]["trend"])
    vol_id = max(ids, key=lambda r: scores[r]["vol"])
    remaining = [r for r in ids if r not in {trend_id, vol_id}]

    profile = PAIR_PROFILE.get(pair, "breakout_trend")
    mapping = {str(r): "MeanReversion" for r in ids}

    if profile == "breakout_trend":
        mapping[str(trend_id)] = "EMATrendPullback"
        mapping[str(vol_id)] = "Breakout"
        for r in remaining:
            # bias one extra regime to breakout if available
            mapping[str(r)] = "Breakout" if r == remaining[0] else "MeanReversion"
    elif profile == "breakout_reversal":
        mapping[str(vol_id)] = "Breakout"
        mapping[str(trend_id)] = "EMATrendPullback"
        for r in remaining:
            mapping[str(r)] = "MeanReversion"
    elif profile == "asia_breakout":
        mapping[str(vol_id)] = "Breakout"
        mapping[str(trend_id)] = "Breakout"
        for r in remaining:
            mapping[str(r)] = "MeanReversion"
    else:
        mapping[str(trend_id)] = "EMATrendPullback"
        mapping[str(vol_id)] = "Breakout"
    return mapping


def _run_train(pair: str, args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--instruments",
        pair,
        "--start",
        args.start,
        "--end",
        args.end,
        "--base-tf",
        args.base_tf,
        "--htf-1",
        args.htf_1,
        "--htf-2",
        args.htf_2,
        "--htf-3",
        args.htf_3,
        "--regimes",
        str(args.regimes),
        "--kmeans-iter",
        str(args.kmeans_iter),
        "--kmeans-restarts",
        str(args.kmeans_restarts),
        "--seed",
        str(args.seed),
        "--feature-lag-bars",
        str(args.feature_lag_bars),
        "--gpu",
        args.gpu,
        "--output-dir",
        args.research_out,
    ]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"train failed for {pair}: {res.stderr or res.stdout}")

    model_path = None
    for line in (res.stdout or "").splitlines():
        if line.startswith("Model JSON:"):
            model_path = line.split(":", 1)[1].strip()
            break
    if not model_path:
        raise RuntimeError(f"Model path not found in trainer output for {pair}")
    return Path(model_path)


def _promote_model(pair: str, model_path: Path, args: argparse.Namespace) -> Path:
    obj = json.loads(model_path.read_text(encoding="utf-8"))
    obj["regime_to_strategy"] = _mapping_for_pair(pair, obj)
    obj["pair_profile"] = PAIR_PROFILE.get(pair, "breakout_trend")
    obj["promoted_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    out_dir = PROJECT_ROOT / args.active_out / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"multiframe_regime_model_{pair}_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    current = out_dir / "CURRENT_MODEL.txt"
    current.write_text(
        f"{out_name}\nsource={model_path}\nprofile={PAIR_PROFILE.get(pair, 'breakout_trend')}\n",
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        raise SystemExit("No pairs provided")

    summary = []
    for pair in pairs:
        try:
            model = _run_train(pair, args)
            active = _promote_model(pair, model, args)
            summary.append({"pair": pair, "status": "ok", "research_model": str(model), "active_model": str(active)})
            print(f"[{pair}] OK -> {active}")
        except Exception as exc:
            summary.append({"pair": pair, "status": "failed", "error": str(exc)})
            print(f"[{pair}] FAILED: {exc}")

    out = PROJECT_ROOT / "data" / "research" / f"multiframe_per_pair_train_summary_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {out}")

    failures = [s for s in summary if s["status"] != "ok"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
