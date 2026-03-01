#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import scripts.pipeline_compat_adapter as adapter
import scripts.regime_strategy_research as rs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _load_symbol_gates(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    obj = _load_yaml_or_json(path)
    if not isinstance(obj, dict):
        return {}
    # Supported shapes:
    # 1) {SYMBOL: {thresholds...}}
    # 2) {"symbol_gates": {SYMBOL: {thresholds...}}}
    gates = obj.get("symbol_gates", obj)
    if not isinstance(gates, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in gates.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, float) and not np.isfinite(v):
            return default
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        fv = float(v)
        if not np.isfinite(fv):
            return default
        return fv
    except Exception:
        return default


def _year_windows(index: pd.DatetimeIndex, train_years: int) -> List[Dict[str, Any]]:
    if len(index) == 0:
        return []
    idx = index.sort_values()
    max_ts = idx.max()
    train_start = max_ts - pd.Timedelta(days=365 * max(1, int(train_years)))
    pre = idx[idx < train_start]
    if len(pre) == 0:
        return []
    years = sorted(set(pre.year.tolist()))
    out: List[Dict[str, Any]] = []
    for y in years:
        s = pd.Timestamp(f"{y}-01-01", tz="UTC")
        e = pd.Timestamp(f"{y}-12-31 23:59:59", tz="UTC")
        n = int(((idx >= s) & (idx <= e)).sum())
        if n < 200:
            continue
        out.append({"year": int(y), "start": s, "end": e, "bars": n})
    return out


def _year_market_state(df_ltf_year: pd.DataFrame) -> Dict[str, float]:
    if df_ltf_year.empty:
        return {
            "xau_return": np.nan,
            "ann_vol": np.nan,
            "atr_pct": np.nan,
            "directional_day_frac": np.nan,
            "compression_day_frac": np.nan,
        }
    cols = [c for c in ["open", "high", "low", "close"] if c in df_ltf_year.columns]
    if len(cols) < 4:
        return {
            "xau_return": np.nan,
            "ann_vol": np.nan,
            "atr_pct": np.nan,
            "directional_day_frac": np.nan,
            "compression_day_frac": np.nan,
        }
    x = df_ltf_year[["open", "high", "low", "close"]].copy()
    x.index = pd.DatetimeIndex(df_ltf_year.index)
    d = (
        x.groupby(x.index.floor("D"))
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna(how="any")
    )
    if d.empty:
        return {
            "xau_return": np.nan,
            "ann_vol": np.nan,
            "atr_pct": np.nan,
            "directional_day_frac": np.nan,
            "compression_day_frac": np.nan,
        }
    d["ret"] = d["close"].pct_change().fillna(0.0)
    prev_close = d["close"].shift(1)
    d["tr"] = np.maximum(d["high"] - d["low"], np.maximum((d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()))
    bar_range = (d["high"] - d["low"]).replace(0.0, np.nan)
    d["trend_day"] = ((d["close"] - d["open"]).abs() / bar_range).clip(0.0, 1.0)
    d["tr_pct"] = (d["tr"] / d["close"]).replace([np.inf, -np.inf], np.nan)
    tr_pct = d["tr_pct"].ffill().bfill()
    q30 = tr_pct.rolling(60, min_periods=20).quantile(0.30)
    compression = (tr_pct < q30).astype(float)
    xau_return = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1.0) if len(d) > 1 else 0.0
    ann_vol = float(d["ret"].std(ddof=0) * np.sqrt(252.0)) if len(d) > 2 else np.nan
    atr_pct = float(tr_pct.mean()) if len(d) > 0 else np.nan
    directional_day_frac = float((d["trend_day"] > 0.60).mean()) if len(d) > 0 else np.nan
    compression_day_frac = float(compression.mean()) if len(d) > 0 else np.nan
    return {
        "xau_return": xau_return,
        "ann_vol": ann_vol,
        "atr_pct": atr_pct,
        "directional_day_frac": directional_day_frac,
        "compression_day_frac": compression_day_frac,
    }


def _regime_fit_summary(dfm: pd.DataFrame) -> Dict[str, Any]:
    if dfm.empty:
        return {
            "regime_profitable_years": 0,
            "regime_expectancy_inband": np.nan,
            "regime_expectancy_outband": np.nan,
            "regime_edge_delta": np.nan,
            "regime_inband_ratio": np.nan,
            "regime_fit_ready": False,
        }
    req_cols = ["ann_vol", "directional_day_frac", "compression_day_frac", "expectancy_r", "profit_factor", "net_expectancy_after_cost"]
    if any(c not in dfm.columns for c in req_cols):
        return {
            "regime_profitable_years": 0,
            "regime_expectancy_inband": np.nan,
            "regime_expectancy_outband": np.nan,
            "regime_edge_delta": np.nan,
            "regime_inband_ratio": np.nan,
            "regime_fit_ready": False,
        }
    valid = dfm.dropna(subset=["ann_vol", "directional_day_frac", "compression_day_frac"]).copy()
    if valid.empty:
        return {
            "regime_profitable_years": 0,
            "regime_expectancy_inband": np.nan,
            "regime_expectancy_outband": np.nan,
            "regime_edge_delta": np.nan,
            "regime_inband_ratio": np.nan,
            "regime_fit_ready": False,
        }
    profitable = (valid["expectancy_r"] > 0.0) & (valid["net_expectancy_after_cost"] > 0.0) & (valid["profit_factor"] > 1.0)
    p = valid.loc[profitable].copy()
    n = valid.loc[~profitable].copy()
    if p.empty:
        return {
            "regime_profitable_years": 0,
            "regime_expectancy_inband": np.nan,
            "regime_expectancy_outband": float(valid["expectancy_r"].mean()),
            "regime_edge_delta": np.nan,
            "regime_inband_ratio": 0.0,
            "regime_fit_ready": False,
        }
    def _band(s: pd.Series) -> tuple[float, float]:
        if len(s) <= 1:
            v = float(s.iloc[0])
            return (v, v)
        return (float(s.quantile(0.25)), float(s.quantile(0.75)))

    vol_lo, vol_hi = _band(p["ann_vol"])
    dir_lo, dir_hi = _band(p["directional_day_frac"])
    cmp_lo, cmp_hi = _band(p["compression_day_frac"])
    in_band = (
        valid["ann_vol"].between(vol_lo, vol_hi, inclusive="both")
        & valid["directional_day_frac"].between(dir_lo, dir_hi, inclusive="both")
        & valid["compression_day_frac"].between(cmp_lo, cmp_hi, inclusive="both")
    )
    exp_in = float(valid.loc[in_band, "expectancy_r"].mean()) if bool(in_band.any()) else np.nan
    exp_out = float(valid.loc[~in_band, "expectancy_r"].mean()) if bool((~in_band).any()) else np.nan
    edge_delta = float(exp_in - exp_out) if np.isfinite(exp_in) and np.isfinite(exp_out) else np.nan
    inband_ratio = float(in_band.mean()) if len(valid) else np.nan
    vol_lift = float(p["ann_vol"].mean() - n["ann_vol"].mean()) if (not p.empty and not n.empty) else np.nan
    dir_lift = float(p["directional_day_frac"].mean() - n["directional_day_frac"].mean()) if (not p.empty and not n.empty) else np.nan
    cmp_lift = float(n["compression_day_frac"].mean() - p["compression_day_frac"].mean()) if (not p.empty and not n.empty) else np.nan
    sep_score = (
        float(np.nanmean([vol_lift, dir_lift, cmp_lift]))
        if any(np.isfinite(v) for v in [vol_lift, dir_lift, cmp_lift])
        else np.nan
    )
    return {
        "regime_profitable_years": int(profitable.sum()),
        "regime_expectancy_inband": exp_in,
        "regime_expectancy_outband": exp_out,
        "regime_edge_delta": edge_delta,
        "regime_inband_ratio": inband_ratio,
        "regime_vol_lift": vol_lift,
        "regime_directional_lift": dir_lift,
        "regime_compression_lift": cmp_lift,
        "regime_separation_score": sep_score,
        "regime_band_ann_vol_lo": vol_lo,
        "regime_band_ann_vol_hi": vol_hi,
        "regime_band_directional_lo": dir_lo,
        "regime_band_directional_hi": dir_hi,
        "regime_band_compression_lo": cmp_lo,
        "regime_band_compression_hi": cmp_hi,
        "regime_fit_ready": True,
    }


def _predict_proba(models: Any, X: pd.DataFrame) -> Optional[pd.Series]:
    if X.empty:
        return None
    if not isinstance(models, list):
        models = [models]
    preds: List[pd.Series] = []
    for m in models:
        if not hasattr(m, "predict_proba") or not hasattr(m, "feature_names"):
            continue
        names = list(getattr(m, "feature_names", []))
        if not names:
            continue
        Xm = X.reindex(columns=names).fillna(0.0)
        try:
            p = m.predict_proba(Xm)[:, 1]
            preds.append(pd.Series(p, index=X.index, dtype=float))
        except Exception:
            continue
    if not preds:
        return None
    arr = np.column_stack([s.to_numpy(dtype=float) for s in preds])
    return pd.Series(np.nanmean(arr, axis=1), index=X.index, dtype=float)


def _safe_json_load(x: Any, fallback: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return fallback
    return x if x is not None else fallback


def _build_hybrid_signals(
    merged: pd.DataFrame,
    regime_map: Dict[str, str],
    strategy_cfg: Dict[str, Any],
    ltf_tf: str,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for regime_name, strategy_name in regime_map.items():
        s = rs._generate_strategy_signals(
            merged,
            str(strategy_name),
            {"ltf_seconds": rs._timeframe_seconds(ltf_tf), **strategy_cfg.get(str(strategy_name), {})},
        )
        if s.empty:
            continue
        ridx = merged.index[merged["regime"] == regime_name]
        idx = s.index.intersection(ridx)
        if len(idx) == 0:
            continue
        part = s.loc[idx].copy()
        part["router_regime"] = regime_name
        part["router_strategy"] = str(strategy_name)
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _evaluate_model_oos_years(
    meta_path: Path,
    model_path: Path,
    cfg: Dict[str, Any],
    train_years: int,
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
    timezone = str(cfg.get("timezone", "Europe/London"))

    with model_path.open("rb") as f:
        models = pickle.load(f)

    h1_tf = str(cfg.get("htf_timeframes", {}).get("h1", "H1"))
    d1_tf = str(cfg.get("htf_timeframes", {}).get("d1", "D1"))
    sym_cfg = rs._pair_cfg(symbol, cfg)

    df_ltf = adapter.load_ohlcv(symbol, ltf_tf)
    df_h1 = adapter.load_ohlcv(symbol, h1_tf)
    df_d1 = adapter.load_ohlcv(symbol, d1_tf)
    df_ltf = rs._ensure_utc_index(df_ltf)
    df_h1 = rs._ensure_utc_index(df_h1)
    df_d1 = rs._ensure_utc_index(df_d1)

    windows = _year_windows(pd.DatetimeIndex(df_ltf.index), train_years=train_years)
    if not windows:
        return {"rows": [], "summary": {"stable": False, "reason": "no_untrained_full_years"}}

    ltf_features = adapter.make_features(df_ltf, ltf_tf, sym_cfg)
    ltf_features = rs._ensure_utc_index(ltf_features).reindex(df_ltf.index)
    sess = rs._session_flags(df_ltf.index, timezone, sym_cfg.get("sessions", {}))
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
        sig = rs._refine_low_tf_entries(sig, merged, ltf_tf, sym_cfg)

    if sig.empty:
        return {"rows": [], "summary": {"stable": False, "reason": "no_signals"}}

    idx = sig.index.intersection(merged.index)
    X = merged.loc[idx, :].copy()
    cols = [c for c in feature_list if c in X.columns]
    if not cols:
        return {"rows": [], "summary": {"stable": False, "reason": "no_feature_overlap"}}
    X = X[cols].copy()
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            codes, _ = pd.factorize(X[c], sort=True)
            X[c] = pd.Series(codes, index=X.index, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    sig = sig.loc[X.index]
    probs = _predict_proba(models, X)
    bt_signals = rs._build_signals_for_backtest(sig, probs, no_trade_cfg)
    if bt_signals.empty:
        return {"rows": [], "summary": {"stable": False, "reason": "no_backtest_signals"}}

    rows: List[Dict[str, Any]] = []
    for w in windows:
        s = w["start"]
        e = w["end"]
        df_y = df_ltf.loc[(df_ltf.index >= s) & (df_ltf.index <= e)]
        sig_y = bt_signals.loc[(bt_signals.index >= s) & (bt_signals.index <= e)]
        if df_y.empty or sig_y.empty:
            continue
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
        ms = _year_market_state(df_y)
        rows.append(
            {
                "year": int(w["year"]),
                "start": str(s.date()),
                "end": str(e.date()),
                "bars": int(w["bars"]),
                "trades": int(tm.get("trades", 0)),
                "expectancy_r": float(tm.get("expectancy_r", 0.0)),
                "net_expectancy_after_cost": float(tm.get("net_expectancy_after_cost", 0.0)),
                "profit_factor": float(tm.get("profit_factor", 0.0)),
                "win_rate": float(tm.get("win_rate", 0.0)),
                "max_dd": float(tm.get("max_dd", 0.0)),
                "sharpe_trade": float(tm.get("sharpe_trade", 0.0)),
                "window_pass_rate": float(tm.get("window_pass_rate", 0.0)),
                "xau_return": float(ms.get("xau_return", np.nan)),
                "ann_vol": float(ms.get("ann_vol", np.nan)),
                "atr_pct": float(ms.get("atr_pct", np.nan)),
                "directional_day_frac": float(ms.get("directional_day_frac", np.nan)),
                "compression_day_frac": float(ms.get("compression_day_frac", np.nan)),
            }
        )

    dfm = pd.DataFrame(rows)
    if dfm.empty:
        return {"rows": [], "summary": {"stable": False, "reason": "no_year_metrics"}}

    profitable = (dfm["expectancy_r"] > 0) & (dfm["net_expectancy_after_cost"] > 0) & (dfm["profit_factor"] > 1.0)
    stable = (
        (int((dfm["trades"] >= 20).sum()) >= max(2, len(dfm) // 2))
        and (float(profitable.mean()) >= 0.55)
        and (float(dfm["expectancy_r"].median()) > 0)
        and (float(dfm["max_dd"].max()) <= 0.20)
    )
    summary = {
        "years_tested": int(len(dfm)),
        "years_profitable": int(profitable.sum()),
        "profit_year_ratio": float(profitable.mean()),
        "median_expectancy_r": float(dfm["expectancy_r"].median()),
        "max_drawdown_worst_year": float(dfm["max_dd"].max()),
        "stable": bool(stable),
    }
    summary.update(_regime_fit_summary(dfm))
    return {"rows": rows, "summary": summary}


def main() -> int:
    p = argparse.ArgumentParser(description="Promote champions + year-by-year OOS validation for selected regime-strategy models.")
    p.add_argument("--ranking-csv", default="data/research/regime_strategy_ranking.csv")
    p.add_argument("--config", default="config/regime_strategy_research.local.yaml")
    p.add_argument("--active-out", default="models/active")
    p.add_argument("--oos-out", default="data/research/regime_strategy_oos_yearly")
    p.add_argument("--train-years", type=int, default=10)
    p.add_argument("--min-tested-years", type=int, default=3)
    p.add_argument("--min-profitable-years", type=int, default=1)
    p.add_argument("--min-profit-year-ratio", type=float, default=0.33)
    p.add_argument("--min-median-expectancy-r", type=float, default=0.0)
    p.add_argument("--max-worst-year-dd", type=float, default=0.20)
    p.add_argument("--gate-mode", choices=["yearly", "regime_fit"], default="yearly")
    p.add_argument("--min-regime-profitable-years", type=int, default=2)
    p.add_argument("--min-regime-edge-delta", type=float, default=0.0001)
    p.add_argument("--min-regime-expectancy-inband", type=float, default=0.0)
    p.add_argument("--min-regime-separation-score", type=float, default=0.0)
    p.add_argument("--min-regime-inband-ratio", type=float, default=0.20)
    p.add_argument("--symbol-gates", default="", help="Optional YAML/JSON with per-symbol thresholds.")
    p.add_argument("--promote-stable-only", action="store_true", help="Only promote champions that passed the selected gate.")
    p.add_argument("--prune-active-to-stable", action="store_true")
    args = p.parse_args()

    cfg = _load_yaml_or_json(PROJECT_ROOT / args.config)
    symbol_gates = _load_symbol_gates((PROJECT_ROOT / args.symbol_gates) if args.symbol_gates else None)
    ranking = pd.read_csv(PROJECT_ROOT / args.ranking_csv)
    sel = ranking[ranking["model_selected"] == True].copy()
    if sel.empty:
        print(json.dumps({"selected": 0, "promoted_champions": 0, "stable_models": 0}, indent=2))
        return 0

    oos_root = PROJECT_ROOT / args.oos_out
    oos_root.mkdir(parents=True, exist_ok=True)

    stable_rows: List[Dict[str, Any]] = []
    for _, row in sel.iterrows():
        model_path = PROJECT_ROOT / str(row["artifact_model_path"])
        meta_path = PROJECT_ROOT / str(row["artifact_meta_path"])
        if not model_path.exists() or not meta_path.exists():
            continue
        model_key = f"{row['symbol']}__{row['strategy']}__{row['regime_variant']}__{Path(model_path).stem.split('_')[-2]}_{Path(model_path).stem.split('_')[-1]}"
        out_dir = oos_root / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        res = _evaluate_model_oos_years(meta_path, model_path, cfg, train_years=args.train_years)
        pd.DataFrame(res.get("rows", [])).to_csv(out_dir / "oos_yearly_metrics.csv", index=False)
        (out_dir / "oos_summary.json").write_text(json.dumps(res.get("summary", {}), indent=2), encoding="utf-8")

        s = dict(res.get("summary", {}))
        s.update(
            {
                "symbol": str(row["symbol"]),
                "strategy": str(row["strategy"]),
                "regime_variant": str(row["regime_variant"]),
                "ltf_timeframe": str(row["ltf_timeframe"]),
                "deployment_role": str(row.get("deployment_role", "")),
                "artifact_model_path": str(row["artifact_model_path"]),
                "artifact_meta_path": str(row["artifact_meta_path"]),
            }
        )
        stable_rows.append(s)

    stable_df = pd.DataFrame(stable_rows)
    if stable_df.empty:
        stable_df = pd.DataFrame(
            columns=[
                "symbol",
                "strategy",
                "regime_variant",
                "ltf_timeframe",
                "deployment_role",
                "years_tested",
                "years_profitable",
                "profit_year_ratio",
                "median_expectancy_r",
                "max_drawdown_worst_year",
                "stable",
                "artifact_model_path",
                "artifact_meta_path",
            ]
        )
    if not stable_df.empty:
        stable_flags: List[bool] = []
        gate_applied: List[Dict[str, Any]] = []
        for _, r in stable_df.iterrows():
            sym = str(r.get("symbol", ""))
            sg = symbol_gates.get(sym, {})
            min_tested = int(sg.get("min_tested_years", args.min_tested_years))
            min_profitable = int(sg.get("min_profitable_years", args.min_profitable_years))
            min_ratio = float(sg.get("min_profit_year_ratio", args.min_profit_year_ratio))
            min_med_exp = float(sg.get("min_median_expectancy_r", args.min_median_expectancy_r))
            max_worst_dd = float(sg.get("max_worst_year_dd", args.max_worst_year_dd))
            if args.gate_mode == "regime_fit":
                min_regime_profitable = int(sg.get("min_regime_profitable_years", args.min_regime_profitable_years))
                min_regime_edge_delta = float(sg.get("min_regime_edge_delta", args.min_regime_edge_delta))
                min_regime_exp_in = float(sg.get("min_regime_expectancy_inband", args.min_regime_expectancy_inband))
                min_regime_sep = float(sg.get("min_regime_separation_score", args.min_regime_separation_score))
                min_regime_inband_ratio = float(sg.get("min_regime_inband_ratio", args.min_regime_inband_ratio))
                ok = (
                    (_safe_int(r.get("years_tested", 0), 0) >= min_tested)
                    and (_safe_float(r.get("max_drawdown_worst_year", 1.0), 1.0) <= max_worst_dd)
                    and (_safe_int(r.get("regime_profitable_years", 0), 0) >= min_regime_profitable)
                    and (_safe_float(r.get("regime_expectancy_inband", -1.0), -1.0) >= min_regime_exp_in)
                    and (_safe_float(r.get("regime_edge_delta", -1.0), -1.0) >= min_regime_edge_delta)
                    and (_safe_float(r.get("regime_separation_score", -1.0), -1.0) >= min_regime_sep)
                    and (_safe_float(r.get("regime_inband_ratio", 0.0), 0.0) >= min_regime_inband_ratio)
                )
                gate_applied.append(
                    {
                        "mode": "regime_fit",
                        "min_tested_years": min_tested,
                        "max_worst_year_dd": max_worst_dd,
                        "min_regime_profitable_years": min_regime_profitable,
                        "min_regime_expectancy_inband": min_regime_exp_in,
                        "min_regime_edge_delta": min_regime_edge_delta,
                        "min_regime_separation_score": min_regime_sep,
                        "min_regime_inband_ratio": min_regime_inband_ratio,
                    }
                )
            else:
                ok = (
                    (_safe_int(r.get("years_tested", 0), 0) >= min_tested)
                    and (_safe_int(r.get("years_profitable", 0), 0) >= min_profitable)
                    and (_safe_float(r.get("profit_year_ratio", 0.0), 0.0) >= min_ratio)
                    and (_safe_float(r.get("median_expectancy_r", -1.0), -1.0) >= min_med_exp)
                    and (_safe_float(r.get("max_drawdown_worst_year", 1.0), 1.0) <= max_worst_dd)
                )
                gate_applied.append(
                    {
                        "mode": "yearly",
                        "min_tested_years": min_tested,
                        "min_profitable_years": min_profitable,
                        "min_profit_year_ratio": min_ratio,
                        "min_median_expectancy_r": min_med_exp,
                        "max_worst_year_dd": max_worst_dd,
                    }
                )
            stable_flags.append(bool(ok))
        stable_df["stable"] = pd.Series(stable_flags, index=stable_df.index)
        stable_df["applied_gate_json"] = [json.dumps(g, separators=(",", ":")) for g in gate_applied]
    stable_csv = oos_root / "selected_models_oos_stability.csv"
    stable_df.to_csv(stable_csv, index=False)

    # Promote champions only (or only stable champions if requested).
    active_root = PROJECT_ROOT / args.active_out
    promoted = 0
    champs = sel[sel["deployment_role"].fillna("") == "champion"].copy()
    promote_champs = champs.copy()
    if args.promote_stable_only and (not stable_df.empty):
        stable_key = (
            stable_df.loc[stable_df["stable"] == True, ["symbol", "strategy", "regime_variant", "ltf_timeframe"]]
            .drop_duplicates()
            .assign(_ok=True)
        )
        promote_champs = promote_champs.merge(
            stable_key, on=["symbol", "strategy", "regime_variant", "ltf_timeframe"], how="inner"
        )
    for _, row in champs.iterrows():
        if args.promote_stable_only:
            match = (
                (promote_champs["symbol"] == row["symbol"])
                & (promote_champs["strategy"] == row["strategy"])
                & (promote_champs["regime_variant"] == row["regime_variant"])
                & (promote_champs["ltf_timeframe"] == row["ltf_timeframe"])
            )
            if not bool(match.any()):
                continue
        symbol = str(row["symbol"])
        dst_dir = active_root / symbol
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_model = PROJECT_ROOT / str(row["artifact_model_path"])
        src_meta = PROJECT_ROOT / str(row["artifact_meta_path"])
        if not src_model.exists() or not src_meta.exists():
            continue
        dst_model = dst_dir / src_model.name
        dst_meta = dst_dir / src_meta.name
        shutil.copy2(src_model, dst_model)
        shutil.copy2(src_meta, dst_meta)
        (dst_dir / "CURRENT_MODEL.txt").write_text(f"{dst_model.name}\n", encoding="utf-8")
        promoted += 1

    stable_keep = stable_df[stable_df.get("stable", False) == True].copy() if not stable_df.empty else stable_df
    stable_keep_csv = oos_root / "stable_models_only.csv"
    stable_keep.to_csv(stable_keep_csv, index=False)

    pruned = 0
    if args.prune_active_to_stable:
        stable_champs = stable_keep[stable_keep["deployment_role"].fillna("") == "champion"].copy() if not stable_keep.empty else stable_keep
        stable_by_symbol: Dict[str, str] = {}
        for _, r in stable_champs.iterrows():
            stable_by_symbol[str(r["symbol"])] = Path(str(r["artifact_model_path"])).name

        for _, r in champs.iterrows():
            symbol = str(r["symbol"])
            dst_dir = active_root / symbol
            cur = dst_dir / "CURRENT_MODEL.txt"
            keep_name = stable_by_symbol.get(symbol, "")
            if keep_name:
                cur.write_text(f"{keep_name}\n", encoding="utf-8")
                continue
            if cur.exists():
                ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
                cur.rename(dst_dir / f"CURRENT_MODEL.demoted_{ts}.txt")
                pruned += 1

    thresholds_path = oos_root / "stability_thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "min_tested_years": int(args.min_tested_years),
                "min_profitable_years": int(args.min_profitable_years),
                "min_profit_year_ratio": float(args.min_profit_year_ratio),
                "min_median_expectancy_r": float(args.min_median_expectancy_r),
                "max_worst_year_dd": float(args.max_worst_year_dd),
                "gate_mode": str(args.gate_mode),
                "min_regime_profitable_years": int(args.min_regime_profitable_years),
                "min_regime_edge_delta": float(args.min_regime_edge_delta),
                "min_regime_expectancy_inband": float(args.min_regime_expectancy_inband),
                "min_regime_separation_score": float(args.min_regime_separation_score),
                "min_regime_inband_ratio": float(args.min_regime_inband_ratio),
                "symbol_gates_file": str(args.symbol_gates or ""),
                "symbol_gates_count": int(len(symbol_gates)),
                "symbol_gates": symbol_gates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "selected": int(len(sel)),
                "champions": int(len(champs)),
                "promoted_champions": int(promoted),
                "oos_summary_csv": str(stable_csv.relative_to(PROJECT_ROOT)),
                "stable_only_csv": str(stable_keep_csv.relative_to(PROJECT_ROOT)),
                "stability_thresholds_json": str(thresholds_path.relative_to(PROJECT_ROOT)),
                "stable_models": int(len(stable_keep)),
                "active_demoted": int(pruned),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
