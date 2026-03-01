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
    p.add_argument("--symbol-gates", default="", help="Optional YAML/JSON with per-symbol thresholds.")
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
            ok = (
                (int(r.get("years_tested", 0) or 0) >= min_tested)
                and (int(r.get("years_profitable", 0) or 0) >= min_profitable)
                and (float(r.get("profit_year_ratio", 0.0) or 0.0) >= min_ratio)
                and (float(r.get("median_expectancy_r", -1.0) or -1.0) >= min_med_exp)
                and (float(r.get("max_drawdown_worst_year", 1.0) or 1.0) <= max_worst_dd)
            )
            stable_flags.append(bool(ok))
            gate_applied.append(
                {
                    "min_tested_years": min_tested,
                    "min_profitable_years": min_profitable,
                    "min_profit_year_ratio": min_ratio,
                    "min_median_expectancy_r": min_med_exp,
                    "max_worst_year_dd": max_worst_dd,
                }
            )
        stable_df["stable"] = pd.Series(stable_flags, index=stable_df.index)
        stable_df["applied_gate_json"] = [json.dumps(g, separators=(",", ":")) for g in gate_applied]
    stable_csv = oos_root / "selected_models_oos_stability.csv"
    stable_df.to_csv(stable_csv, index=False)

    # Promote champions only.
    active_root = PROJECT_ROOT / args.active_out
    promoted = 0
    champs = sel[sel["deployment_role"].fillna("") == "champion"].copy()
    for _, row in champs.iterrows():
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
