#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

import scripts.regime_strategy_research as rs


def _out_with_suffix(path_str: str, suffix: str) -> str:
    p = Path(path_str)
    if p.suffix:
        return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))
    return str(p.with_name(f"{p.name}{suffix}"))


def _safe_json_load(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _extract_rvals(trades: pd.DataFrame) -> pd.Series:
    if "r_multiple" in trades.columns:
        return pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0.0)
    if "pnl" in trades.columns:
        return pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    if "net_pnl" in trades.columns:
        return pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=trades.index, dtype=float)


def _extract_entry_times(trades: pd.DataFrame) -> pd.DatetimeIndex:
    for col in ["entry_time", "timestamp", "open_time"]:
        if col in trades.columns:
            t = pd.to_datetime(trades[col], utc=True, errors="coerce")
            return pd.DatetimeIndex(t.dropna())
    t = pd.to_datetime(trades.index, utc=True, errors="coerce")
    return pd.DatetimeIndex(t.dropna())


def _in_window(hour: int, bounds: List[int]) -> bool:
    s, e = int(bounds[0]), int(bounds[1])
    if s <= e:
        return (hour >= s) and (hour < e)
    return (hour >= s) or (hour < e)


def _session_name(hour: int, sessions: Dict[str, List[int]]) -> str:
    for s in ["tokyo", "london", "newyork", "overlap"]:
        b = sessions.get(s)
        if isinstance(b, list) and len(b) >= 2 and _in_window(hour, b):
            return s
    return "offhours"


def _session_stats_from_trade_log(path: Path, tz_name: str, sessions: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        trades = pd.read_csv(path)
    except Exception:
        return {}
    if trades.empty:
        return {}
    r = _extract_rvals(trades)
    t = _extract_entry_times(trades)
    n = min(len(r), len(t))
    if n <= 0:
        return {}
    r = r.iloc[:n].reset_index(drop=True)
    t = t[:n].tz_convert(tz_name)
    out: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        sess = _session_name(int(t[i].hour), sessions)
        if sess not in out:
            out[sess] = {"trades": 0.0, "r_sum": 0.0, "wins": 0.0}
        rv = float(r.iloc[i])
        out[sess]["trades"] += 1.0
        out[sess]["r_sum"] += rv
        out[sess]["wins"] += 1.0 if rv > 0 else 0.0
    for k, v in out.items():
        tr = max(1.0, float(v["trades"]))
        v["expectancy_r"] = float(v["r_sum"] / tr)
        v["win_rate"] = float(v["wins"] / tr)
    return out


def _learn_profitable_buckets(
    selected: pd.DataFrame,
    tz_name: str,
    sessions: Dict[str, List[int]],
    min_session_trades: int,
    min_style_trades: int,
    min_session_expectancy: float,
    min_style_expectancy: float,
    fallback_min_trades: int,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    sess_agg: Dict[str, Dict[str, float]] = {}
    style_agg: Dict[str, Dict[str, float]] = {}
    notes: List[str] = []

    for _, row in selected.iterrows():
        style = str(row.get("style", ""))
        tr = float(row.get("trades", 0.0) or 0.0)
        ex = float(row.get("expectancy_r", 0.0) or 0.0)
        if style:
            s = style_agg.setdefault(style, {"trades": 0.0, "r_sum": 0.0})
            s["trades"] += tr
            s["r_sum"] += tr * ex

        trade_log_path = str(row.get("artifact_trade_log_path", "") or "")
        sess_stats: Dict[str, Dict[str, float]] = {}
        if trade_log_path:
            sess_stats = _session_stats_from_trade_log(Path(trade_log_path), tz_name, sessions)
        if not sess_stats:
            sess_stats = _safe_json_load(row.get("session_breakdown_json", ""))
            if sess_stats:
                notes.append("fallback_session_breakdown_json_used")
        for sess, vals in sess_stats.items():
            t_s = float(vals.get("trades", 0.0) or 0.0)
            e_s = float(vals.get("expectancy_r", 0.0) or 0.0)
            a = sess_agg.setdefault(sess, {"trades": 0.0, "r_sum": 0.0})
            a["trades"] += t_s
            a["r_sum"] += t_s * e_s

    profitable_sessions: List[str] = []
    for sess, vals in sess_agg.items():
        t_s = float(vals["trades"])
        if t_s < float(min_session_trades):
            continue
        exp_s = float(vals["r_sum"] / max(t_s, 1.0))
        if exp_s >= float(min_session_expectancy):
            profitable_sessions.append(sess)

    profitable_styles: List[str] = []
    for sty, vals in style_agg.items():
        t_sty = float(vals["trades"])
        if t_sty < float(min_style_trades):
            continue
        exp_sty = float(vals["r_sum"] / max(t_sty, 1.0))
        if exp_sty >= float(min_style_expectancy):
            profitable_styles.append(sty)

    # Fallback for sparse samples: keep positive-expectancy buckets with lighter support.
    if not profitable_sessions:
        for sess, vals in sess_agg.items():
            t_s = float(vals["trades"])
            if t_s < float(max(1, fallback_min_trades)):
                continue
            exp_s = float(vals["r_sum"] / max(t_s, 1.0))
            if exp_s > 0:
                profitable_sessions.append(sess)
    if not profitable_styles:
        for sty, vals in style_agg.items():
            t_sty = float(vals["trades"])
            if t_sty < float(max(1, fallback_min_trades)):
                continue
            exp_sty = float(vals["r_sum"] / max(t_sty, 1.0))
            if exp_sty > 0:
                profitable_styles.append(sty)

    details = {
        "session_aggregate": {
            k: {
                "trades": float(v["trades"]),
                "expectancy_r": float(v["r_sum"] / max(v["trades"], 1.0)),
            }
            for k, v in sess_agg.items()
        },
        "style_aggregate": {
            k: {
                "trades": float(v["trades"]),
                "expectancy_r": float(v["r_sum"] / max(v["trades"], 1.0)),
            }
            for k, v in style_agg.items()
        },
        "notes": sorted(set(notes)),
    }
    return sorted(profitable_sessions), sorted(profitable_styles), details


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-pass refinement using selected model trade logs.")
    p.add_argument("--pipeline-module", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--symbol", default="XAU_USD")
    p.add_argument("--min-session-trades", type=int, default=30)
    p.add_argument("--min-style-trades", type=int, default=40)
    p.add_argument("--min-session-expectancy", type=float, default=0.0)
    p.add_argument("--min-style-expectancy", type=float, default=0.0)
    p.add_argument("--fallback-min-trades", type=int, default=8)
    p.add_argument("--report-json", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = rs._load_config(args.config)
    fns = rs._load_pipeline(args.pipeline_module)
    symbol = str(args.symbol)

    cfg_base = copy.deepcopy(cfg)
    cfg_base["symbols"] = [symbol]
    cfg_base.setdefault("output", {})
    cfg_base["output"]["save_selected_trade_logs"] = True

    pass1_cfg = copy.deepcopy(cfg_base)
    pass1_cfg["output"]["ranking_csv"] = _out_with_suffix(
        str(pass1_cfg["output"].get("ranking_csv", "data/research/regime_strategy_ranking.csv")),
        ".pass1",
    )
    pass1_cfg["output"]["manifest_json"] = _out_with_suffix(
        str(pass1_cfg["output"].get("manifest_json", "data/research/regime_strategy_manifest.json")),
        ".pass1",
    )
    pass1_cfg["output"]["model_dir"] = _out_with_suffix(
        str(pass1_cfg["output"].get("model_dir", "models/research/regime_strategy")),
        "_pass1",
    )
    res1 = rs.run_research(pass1_cfg, fns, template_only=False)
    r1 = pd.read_csv(res1["ranking_csv"])
    selected = r1[r1["model_selected"] == True].copy()
    if selected.empty:
        selected = r1[r1["deploy_eligible"] == True].sort_values(
            ["objective_score", "expectancy_r", "trades"], ascending=[False, False, False]
        ).head(3)

    tz_name = str(cfg_base.get("timezone", "Europe/London"))
    sess_map = cfg_base.get("sessions", {})
    prof_sess, prof_styles, details = _learn_profitable_buckets(
        selected=selected,
        tz_name=tz_name,
        sessions=sess_map,
        min_session_trades=int(args.min_session_trades),
        min_style_trades=int(args.min_style_trades),
        min_session_expectancy=float(args.min_session_expectancy),
        min_style_expectancy=float(args.min_style_expectancy),
        fallback_min_trades=int(args.fallback_min_trades),
    )

    pass2_cfg = copy.deepcopy(cfg_base)
    pass2_cfg["disable_session_filters"] = False
    pair = pass2_cfg.setdefault("pair_overrides", {}).setdefault(symbol, {})
    pair["disable_session_filters"] = False
    if prof_sess:
        pair["preferred_sessions"] = prof_sess
    if prof_styles:
        pair["preferred_styles"] = prof_styles

    pass2_cfg["output"]["ranking_csv"] = _out_with_suffix(
        str(pass2_cfg["output"].get("ranking_csv", "data/research/regime_strategy_ranking.csv")),
        ".refined",
    )
    pass2_cfg["output"]["manifest_json"] = _out_with_suffix(
        str(pass2_cfg["output"].get("manifest_json", "data/research/regime_strategy_manifest.json")),
        ".refined",
    )
    pass2_cfg["output"]["model_dir"] = _out_with_suffix(
        str(pass2_cfg["output"].get("model_dir", "models/research/regime_strategy")),
        "_refined",
    )
    pass2_cfg["output"]["save_selected_trade_logs"] = True

    res2 = rs.run_research(pass2_cfg, fns, template_only=False)
    r2 = pd.read_csv(res2["ranking_csv"])
    selected2 = r2[r2["model_selected"] == True].copy()

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": symbol,
        "pass1": res1,
        "pass2": res2,
        "learned_profitable_sessions": prof_sess,
        "learned_profitable_styles": prof_styles,
        "learning_details": details,
        "selected_pass1_rows": int(len(selected)),
        "selected_pass2_rows": int(len(selected2)),
        "pass1_ranking_csv": str(res1["ranking_csv"]),
        "pass2_ranking_csv": str(res2["ranking_csv"]),
    }
    report_path = Path(args.report_json) if args.report_json else Path(
        f"data/research/regime_refine_from_logs_{symbol.lower()}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(report_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
