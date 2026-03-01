#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_cfg(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("Config must be an object")
    return obj


def _dump_cfg(path: Path, cfg: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return
    except Exception:
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _load_events(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    for c in ["timestamp_utc", "ts_utc", "ts", "time", "timestamp"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
            return [t.isoformat().replace("+00:00", "Z") for t in s]
    raise ValueError("No timestamp column found in calendar CSV (expected one of timestamp_utc/ts_utc/ts/time/timestamp)")


def main() -> int:
    p = argparse.ArgumentParser(description="Wire macro calendar timestamps into strategy research config.")
    p.add_argument("--config-in", required=True)
    p.add_argument("--calendar-csv", required=True)
    p.add_argument("--config-out", required=True)
    p.add_argument("--pair", default="XAU_USD")
    p.add_argument("--event-pre-minutes", type=int, default=60)
    p.add_argument("--event-post-minutes", type=int, default=120)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config_in))
    events = _load_events(Path(args.calendar_csv))

    macro = dict(cfg.get("macro", {}))
    macro["events_utc"] = events
    macro["event_pre_minutes"] = int(args.event_pre_minutes)
    macro["event_post_minutes"] = int(args.event_post_minutes)
    cfg["macro"] = macro

    pair_overrides = dict(cfg.get("pair_overrides", {}))
    pair_cfg = dict(pair_overrides.get(args.pair, {}))
    barrier = dict(pair_cfg.get("barrier", {}))
    barrier["event_schedule_utc"] = events
    barrier["event_pre_minutes"] = int(args.event_pre_minutes)
    barrier["event_post_minutes"] = int(args.event_post_minutes)
    pair_cfg["barrier"] = barrier
    pair_overrides[args.pair] = pair_cfg
    cfg["pair_overrides"] = pair_overrides

    out_path = Path(args.config_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_cfg(out_path, cfg)
    print(
        json.dumps(
            {
                "config_in": args.config_in,
                "calendar_csv": args.calendar_csv,
                "config_out": args.config_out,
                "pair": args.pair,
                "events_count": len(events),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

