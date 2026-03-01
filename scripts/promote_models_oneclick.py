#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def _copy_artifacts(src_model: Path, src_meta: Path, dst_dir: Path, prefix: str) -> Dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    model_ext = src_model.suffix or ".bin"
    dst_model = dst_dir / f"{prefix}{model_ext}"
    dst_meta = dst_dir / f"{prefix}.meta.json"
    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_meta, dst_meta)
    return {"model": dst_model.name, "meta": dst_meta.name}


def _archive_existing_active(symbol_dir: Path, retired_root: Path) -> None:
    cur = symbol_dir / "CURRENT_MODEL.txt"
    if not cur.exists():
        return
    first = cur.read_text(encoding="utf-8").splitlines()
    if not first:
        return
    old_name = first[0].strip()
    if not old_name:
        return
    old_model = symbol_dir / old_name
    old_meta = symbol_dir / old_name.replace(".pkl", ".meta.json")
    if not old_meta.exists():
        # common existing pattern: .json sidecar with same stem
        old_meta = symbol_dir / f"{Path(old_name).stem}.json"
    arc = retired_root / symbol_dir.name / _ts()
    arc.mkdir(parents=True, exist_ok=True)
    if old_model.exists():
        shutil.copy2(old_model, arc / old_model.name)
    if old_meta.exists():
        shutil.copy2(old_meta, arc / old_meta.name)
    shutil.copy2(cur, arc / "CURRENT_MODEL.txt")


def _write_symbol_manifest(manifest_root: Path, symbol: str, payload: Dict[str, Any]) -> None:
    manifest_root.mkdir(parents=True, exist_ok=True)
    p = manifest_root / f"{symbol}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="One-click model promotion with active/challenger/retired tiers.")
    p.add_argument("--stable-csv", required=True, help="stable_models_only.csv")
    p.add_argument("--all-oos-csv", required=True, help="selected_models_oos_stability.csv")
    p.add_argument("--active-root", default="models/active")
    p.add_argument("--challenger-root", default="models/challengers")
    p.add_argument("--retired-root", default="models/retired")
    p.add_argument("--manifest-root", default="models/manifests")
    p.add_argument("--demote-active-without-stable", action="store_true")
    args = p.parse_args()

    stable_df = pd.read_csv(PROJECT_ROOT / args.stable_csv)
    all_df = pd.read_csv(PROJECT_ROOT / args.all_oos_csv)

    active_root = PROJECT_ROOT / args.active_root
    challenger_root = PROJECT_ROOT / args.challenger_root
    retired_root = PROJECT_ROOT / args.retired_root
    manifest_root = PROJECT_ROOT / args.manifest_root

    promoted = 0
    challenger_set = 0
    demoted = 0

    symbols = sorted(set(all_df["symbol"].astype(str)))
    for symbol in symbols:
        sdf = stable_df[stable_df["symbol"].astype(str) == symbol].copy()
        adf = all_df[all_df["symbol"].astype(str) == symbol].copy()
        symbol_active = active_root / symbol
        symbol_active.mkdir(parents=True, exist_ok=True)
        symbol_chall = challenger_root / symbol
        symbol_chall.mkdir(parents=True, exist_ok=True)

        champ = sdf[sdf["deployment_role"].fillna("") == "champion"].copy()
        if not champ.empty:
            row = champ.iloc[0]
            src_model = PROJECT_ROOT / str(row["artifact_model_path"])
            src_meta = PROJECT_ROOT / str(row["artifact_meta_path"])
            if src_model.exists() and src_meta.exists():
                _archive_existing_active(symbol_active, retired_root)
                prefix = (
                    f"live_{_safe_name(symbol)}_{_safe_name(str(row['strategy']))}_"
                    f"{_safe_name(str(row['ltf_timeframe']))}_{_safe_name(str(row['regime_variant']))}_{_ts()}"
                )
                copied = _copy_artifacts(src_model, src_meta, symbol_active, prefix)
                (symbol_active / "CURRENT_MODEL.txt").write_text(
                    "\n".join(
                        [
                            copied["model"],
                            f"source={row['artifact_model_path']}",
                            "tier=active",
                            "promotion=oneclick",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                promoted += 1
        elif args.demote_active_without_stable:
            cur = symbol_active / "CURRENT_MODEL.txt"
            if cur.exists():
                _archive_existing_active(symbol_active, retired_root)
                cur.rename(symbol_active / f"CURRENT_MODEL.demoted_{_ts()}.txt")
                demoted += 1

        chall = adf[(adf["deployment_role"].fillna("") == "challenger")].copy()
        if not chall.empty:
            row = chall.iloc[0]
            src_model = PROJECT_ROOT / str(row["artifact_model_path"])
            src_meta = PROJECT_ROOT / str(row["artifact_meta_path"])
            if src_model.exists() and src_meta.exists():
                prefix = (
                    f"challenger_{_safe_name(symbol)}_{_safe_name(str(row['strategy']))}_"
                    f"{_safe_name(str(row['ltf_timeframe']))}_{_safe_name(str(row['regime_variant']))}_{_ts()}"
                )
                copied = _copy_artifacts(src_model, src_meta, symbol_chall, prefix)
                (symbol_chall / "CURRENT_CHALLENGER.txt").write_text(
                    "\n".join(
                        [
                            copied["model"],
                            f"source={row['artifact_model_path']}",
                            "status=paper_only",
                            "tier=challenger",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                challenger_set += 1

        # per-symbol manifest
        manifest = {
            "symbol": symbol,
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "active_pointer": str((symbol_active / "CURRENT_MODEL.txt")) if (symbol_active / "CURRENT_MODEL.txt").exists() else "",
            "challenger_pointer": str((symbol_chall / "CURRENT_CHALLENGER.txt"))
            if (symbol_chall / "CURRENT_CHALLENGER.txt").exists()
            else "",
            "stable_rows": int(len(sdf)),
            "oos_rows": int(len(adf)),
        }
        _write_symbol_manifest(manifest_root, symbol, manifest)

    summary = {
        "stable_csv": args.stable_csv,
        "all_oos_csv": args.all_oos_csv,
        "symbols": int(len(symbols)),
        "promoted_active": int(promoted),
        "set_challenger": int(challenger_set),
        "demoted_active": int(demoted),
        "manifest_root": args.manifest_root,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

