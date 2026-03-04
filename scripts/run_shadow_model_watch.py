#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import scripts.pipeline_compat_adapter as adapter
import scripts.regime_strategy_research as rs
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.downloader import OandaDownloader
from scripts.promote_and_validate_selected_models import (
    PROJECT_ROOT,
    _build_hybrid_signals,
    _load_yaml_or_json,
    _predict_proba,
    _safe_json_load,
)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _to_tf(tf: str) -> Timeframe:
    t = str(tf).upper()
    mapping = {
        "M1": Timeframe.M1,
        "M15": Timeframe.M15,
        "M30": Timeframe.M30,
        "H1": Timeframe.H1,
        "H4": Timeframe.H4,
        "D1": Timeframe.D1,
    }
    if t not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[t]


def _load_active_models() -> List[Dict[str, Any]]:
    active_root = PROJECT_ROOT / "models" / "active"
    out: List[Dict[str, Any]] = []
    xau_dir = active_root / "XAU_USD"
    xau_candidates = [
        "live_xau_usd_london_open_continuation_m15_rr3_20260301_115356.pkl",
        "XAU_USD_session_range_breakout_xau_as_v03_20260301_120015.pkl",
        "XAU_USD_compression_breakout_v09_20260301_093757.pkl",
    ]
    for name in xau_candidates:
        p = xau_dir / name
        if p.exists():
            out.append({"symbol": "XAU_USD", "model_path": str(p.relative_to(PROJECT_ROOT))})

    for sym in ["EUR_USD", "GBP_USD", "USD_CAD", "GBP_JPY", "USD_JPY"]:
        sym_dir = active_root / sym
        if not sym_dir.exists():
            continue

        model_path: Path | None = None
        current_ptr = sym_dir / "CURRENT_MODEL.txt"
        if current_ptr.exists():
            model_name = current_ptr.read_text(encoding="utf-8").splitlines()[0].strip()
            if model_name.endswith(".pkl"):
                p = sym_dir / model_name
                if p.exists():
                    model_path = p

        if model_path is None:
            candidates = sorted(sym_dir.glob("live_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                model_path = candidates[0]
        if model_path is None:
            candidates = sorted(sym_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                model_path = candidates[0]

        if model_path is not None:
            out.append({"symbol": sym, "model_path": str(model_path.relative_to(PROJECT_ROOT))})
    return out


def _resolve_meta(model_path: Path) -> Path:
    meta = model_path.with_suffix(".meta.json")
    if meta.exists():
        return meta
    alt = model_path.with_suffix(".json")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Meta json not found for model: {model_path}")


def _ensure_fresh_data(symbol: str, tf: str, lookback_days: int) -> pd.DataFrame:
    tf_u = str(tf).upper()
    now_ts = _utc_now()
    cut = now_ts - pd.Timedelta(days=int(lookback_days))

    # Load local cache first (works for M5 via adapter derivation path).
    local_df = rs._ensure_utc_index(adapter.load_ohlcv(symbol, tf_u))

    # Try to refresh right edge from OANDA; fall back silently to local cache.
    token = os.getenv("OANDA_API_TOKEN", "")
    if token:
        try:
            # Skip refresh fast when DNS is unavailable; avoid long retry stalls.
            socket.gethostbyname("api-fxtrade.oanda.com")
            gran = {"D1": "D", "H4": "H4", "H1": "H1", "M30": "M30", "M15": "M15", "M5": "M5", "M1": "M1"}.get(tf_u, tf_u)
            dl = OandaDownloader(
                {
                    "token": token,
                    "environment": os.getenv("OANDA_ENV", os.getenv("TRADING_ENVIRONMENT", "practice")),
                }
            )
            if len(local_df):
                start = pd.DatetimeIndex(local_df.index).max().to_pydatetime() + dt.timedelta(microseconds=1)
            else:
                start = cut.to_pydatetime()
            end = now_ts.to_pydatetime()
            if start < end:
                new_df = dl.download(symbol, granularity=gran, start=start, end=end)
                if new_df is not None and not new_df.empty:
                    new_df = rs._ensure_utc_index(new_df)
                    merged = pd.concat([local_df, new_df]).sort_index()
                    merged = merged[~merged.index.duplicated(keep="last")]
                    local_df = merged
        except Exception:
            pass
    return local_df[local_df.index >= cut]


def _evaluate_window(
    model_path: Path,
    meta_path: Path,
    cfg: Dict[str, Any],
    lookback_days: int,
    window_days: int,
) -> Dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    symbol = str(meta.get("symbol", ""))
    ltf_tf = str(meta.get("ltf_timeframe", "M15"))
    strategy = str(meta.get("strategy", ""))
    feature_list = list(meta.get("feature_list", []))
    regime_rules = meta.get("regime_rules", {}) or {}
    no_trade_cfg = meta.get("no_trade", {}) or {}
    cost_cfg = meta.get("cost_cfg", {}) or {}
    ranking_metrics = meta.get("ranking_metrics", {}) or {}
    strategy_cfg = cfg.get("strategies", {})
    timezone_name = str(cfg.get("timezone", "Europe/London"))

    with model_path.open("rb") as f:
        models = pickle.load(f)

    h1_tf = str(cfg.get("htf_timeframes", {}).get("h1", "H1"))
    d1_tf = str(cfg.get("htf_timeframes", {}).get("d1", "D1"))
    sym_cfg = rs._pair_cfg(symbol, cfg)

    df_ltf = _ensure_fresh_data(symbol, ltf_tf, lookback_days=lookback_days)
    df_h1 = _ensure_fresh_data(symbol, h1_tf, lookback_days=max(lookback_days, 120))
    df_d1 = _ensure_fresh_data(symbol, d1_tf, lookback_days=max(lookback_days, 800))

    ltf_features = adapter.make_features(df_ltf, ltf_tf, sym_cfg)
    ltf_features = rs._ensure_utc_index(ltf_features).reindex(df_ltf.index)
    sess = rs._session_flags(df_ltf.index, timezone_name, sym_cfg.get("sessions", {}))
    base_df = pd.concat([df_ltf, ltf_features, sess], axis=1)
    regime_feat = rs._compute_regime_features(df_ltf, df_h1, df_d1, sym_cfg.get("regime", {}), regime_rules)
    merged = pd.concat([base_df, regime_feat], axis=1)

    if strategy == "hybrid_regime_router":
        regime_map = _safe_json_load(ranking_metrics.get("regime_strategy_map_json"), {})
        sig = _build_hybrid_signals(merged, regime_map, strategy_cfg, ltf_tf)
    else:
        sig = rs._generate_strategy_signals(
            merged, strategy, {"ltf_seconds": rs._timeframe_seconds(ltf_tf), **strategy_cfg.get(strategy, {})}
        )
        allowed = set(rs.STRATEGY_ALLOWED_REGIMES.get(strategy, []))
        if allowed and not sig.empty:
            sig = sig.join(merged[["regime"]], how="left")
            sig = sig[sig["regime"].isin(allowed)]
        sig = rs._apply_pair_filters(sig, merged, symbol, strategy, sym_cfg)
        try:
            sig = rs._refine_low_tf_entries(sig, merged, ltf_tf, sym_cfg)
        except Exception as exc:
            if "tz-naive" not in str(exc).lower() and "tz-aware" not in str(exc).lower():
                raise

    if sig.empty:
        return {"status": "no_signals"}

    idx = sig.index.intersection(merged.index)
    X = merged.loc[idx, :].copy()
    cols = [c for c in feature_list if c in X.columns]
    if not cols:
        return {"status": "no_feature_overlap"}
    X = X[cols].copy()
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            codes, _ = pd.factorize(X[c], sort=True)
            X[c] = pd.Series(codes, index=X.index, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    sig = sig.loc[X.index]
    probs = _predict_proba(models, X)
    bt_signals = rs._build_signals_for_backtest(sig, probs, no_trade_cfg)

    end_ts = pd.DatetimeIndex(df_ltf.index).max()
    start_ts = end_ts - pd.Timedelta(days=int(window_days))
    df_y = df_ltf.loc[(df_ltf.index >= start_ts) & (df_ltf.index <= end_ts)]
    sig_y = bt_signals.loc[(bt_signals.index >= start_ts) & (bt_signals.index <= end_ts)]
    if df_y.empty or sig_y.empty:
        return {"status": "no_window_signals", "start": str(start_ts), "end": str(end_ts)}

    bt_raw = adapter.backtest_from_signals(df_y, sig_y, cost_cfg)
    if isinstance(bt_raw, tuple) and len(bt_raw) >= 2:
        trade_log, equity_curve = bt_raw[0], bt_raw[1]
    else:
        trade_log = bt_raw.get("trade_log", pd.DataFrame())
        equity_curve = bt_raw.get("equity_curve", pd.Series(dtype=float))

    tm = rs._trade_metrics(
        rs._to_dataframe(trade_log),
        equity_curve,
        df_y.index,
        settings=cfg.get("exploration", {}),
        signals=sig_y,
    )
    return {
        "status": "ok",
        "start": str(start_ts),
        "end": str(end_ts),
        "trades": int(tm.get("trades", 0)),
        "expectancy_r": float(tm.get("expectancy_r", 0.0)),
        "net_expectancy_after_cost": float(tm.get("net_expectancy_after_cost", 0.0)),
        "profit_factor": float(tm.get("profit_factor", 0.0)),
        "win_rate": float(tm.get("win_rate", 0.0)),
        "max_dd": float(tm.get("max_dd", 0.0)),
        "sharpe_trade": float(tm.get("sharpe_trade", 0.0)),
        "window_pass_rate": float(tm.get("window_pass_rate", 0.0)),
        "signal_rows": int(len(sig_y)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Shadow multi-model watch runner (paper mode only).")
    p.add_argument("--config", default="", help="Optional JSON config path with {'models':[...]} entries.")
    p.add_argument("--research-config", default="config/regime_strategy_research.local.yaml")
    p.add_argument("--output-dir", default="data/research/shadow_watch")
    p.add_argument("--poll-seconds", type=int, default=900)
    p.add_argument("--lookback-days", type=int, default=120)
    p.add_argument("--window-days", type=int, default=21)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    out_root = PROJECT_ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = _utc_now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"shadow_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_csv = run_dir / "heartbeat_metrics.csv"

    if args.config:
        cfg_obj = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
        models = list(cfg_obj.get("models", []))
    else:
        models = _load_active_models()
    if not models:
        raise SystemExit("No models configured for shadow watch.")

    rs_cfg = _load_yaml_or_json(PROJECT_ROOT / args.research_config)
    (run_dir / "models.json").write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")

    while True:
        ts = _utc_now().isoformat()
        rows: List[Dict[str, Any]] = []
        for m in models:
            model_path = PROJECT_ROOT / str(m["model_path"])
            symbol = str(m.get("symbol", ""))
            row: Dict[str, Any] = {
                "ts_utc": ts,
                "symbol": symbol,
                "model_path": str(m["model_path"]),
            }
            try:
                meta_path = _resolve_meta(model_path)
                res = _evaluate_window(
                    model_path=model_path,
                    meta_path=meta_path,
                    cfg=rs_cfg,
                    lookback_days=int(args.lookback_days),
                    window_days=int(args.window_days),
                )
                row.update(res)
            except Exception as exc:
                row.update({"status": "error", "error": str(exc)})
            rows.append(row)

        df = pd.DataFrame(rows)
        mode = "a" if heartbeat_csv.exists() else "w"
        df.to_csv(heartbeat_csv, mode=mode, header=(mode == "w"), index=False)
        (run_dir / "latest_snapshot.json").write_text(
            json.dumps({"ts_utc": ts, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"[{ts}] rows={len(rows)} written={heartbeat_csv}")
        if args.once:
            break
        time.sleep(max(30, int(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
