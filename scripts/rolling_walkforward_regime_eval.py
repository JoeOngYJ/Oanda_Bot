#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import scripts.pipeline_compat_adapter as adapter
import scripts.regime_strategy_research as rs


def _load_cfg(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(txt)
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("Config must be a dict")
    return obj


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def _predict_proba(models: Any, X: pd.DataFrame) -> Optional[pd.Series]:
    if X.empty:
        return None
    if not isinstance(models, list):
        models = [models]
    preds: List[pd.Series] = []
    for m in models:
        if not hasattr(m, "predict_proba") or not hasattr(m, "feature_names"):
            continue
        cols = [c for c in getattr(m, "feature_names", []) if c in X.columns]
        if not cols:
            continue
        Xm = X.reindex(columns=cols).fillna(0.0)
        try:
            p = m.predict_proba(Xm)[:, 1]
            preds.append(pd.Series(p, index=X.index, dtype=float))
        except Exception:
            continue
    if not preds:
        return None
    arr = np.column_stack([p.to_numpy(dtype=float) for p in preds])
    return pd.Series(np.nanmean(arr, axis=1), index=X.index, dtype=float)


def _build_xy(
    merged: pd.DataFrame,
    ltf_features: pd.DataFrame,
    sig: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    sig_idx = sig.index.intersection(y.index)
    if len(sig_idx) < 20:
        return pd.DataFrame(), pd.Series(dtype=int), []
    feature_cols = [c for c in ltf_features.columns if c in merged.columns]
    feature_cols += [
        c
        for c in merged.columns
        if c.startswith("h1_")
        or c.startswith("d1_")
        or c.startswith("regime_")
        or c.startswith("sess_")
        or c.startswith("sig_")
    ]
    feature_cols = list(dict.fromkeys(feature_cols))
    X = merged.loc[sig_idx, feature_cols].copy()
    ys = y.loc[sig_idx].copy()
    for col in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]):
            codes, _ = pd.factorize(X[col], sort=True)
            X[col] = pd.Series(codes, index=X.index, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    ys = ys.reindex(X.index)
    return X, ys, feature_cols


def _score_strategy_on_train(
    fns: rs.PipelineFns,
    cfg: Dict[str, Any],
    sym_cfg: Dict[str, Any],
    symbol: str,
    ltf_tf: str,
    merged_train: pd.DataFrame,
    ltf_features_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    objective_cfg: Dict[str, Any],
    min_trades: int,
) -> Optional[Dict[str, Any]]:
    sig = rs._generate_strategy_signals(
        merged_train,
        strategy,
        {"ltf_seconds": rs._timeframe_seconds(ltf_tf), **sym_cfg.get("strategies", {}).get(strategy, {})},
    )
    if sig.empty:
        return None
    allowed = set(rs.STRATEGY_ALLOWED_REGIMES[strategy])
    sig = sig.join(merged_train[["regime"]], how="left")
    sig = sig[sig["regime"].isin(allowed)]
    sig = rs._apply_pair_filters(sig, merged_train, symbol, strategy, sym_cfg)
    sig = rs._refine_low_tf_entries(sig, merged_train, ltf_tf, sym_cfg)
    if sig.empty:
        return None

    X, ys, feature_cols = _build_xy(merged_train, ltf_features_train, sig, y_train)
    if len(X) < 20:
        return None
    wf_raw = fns.walkforward_train_eval(X, ys, sym_cfg.get("model", {}), sym_cfg.get("splits", {}))
    wf_metrics, wf_models = rs._extract_wf_output(wf_raw)
    probs = rs._extract_probability_series(wf_metrics, X.index)
    no_trade_cfg = sym_cfg.get("no_trade", {})
    threshold_grid = sorted({float(v) for v in no_trade_cfg.get("threshold_grid", [no_trade_cfg.get("probability_threshold", 0.58)])})
    best: Optional[Dict[str, Any]] = None
    for th in threshold_grid:
        cfg_nt = dict(no_trade_cfg)
        cfg_nt["probability_threshold"] = th
        bt_signals = rs._build_signals_for_backtest(sig.loc[X.index], probs, cfg_nt)
        if bt_signals.empty:
            continue
        bt_raw = fns.backtest_from_signals(merged_train[["open", "high", "low", "close", "volume"]], bt_signals, sym_cfg.get("cost", {}))
        if isinstance(bt_raw, tuple) and len(bt_raw) >= 2:
            trade_log, eq = bt_raw[0], bt_raw[1]
        else:
            trade_log = bt_raw.get("trade_log", pd.DataFrame())
            eq = bt_raw.get("equity_curve", pd.Series(dtype=float))
        tm = rs._trade_metrics(rs._to_dataframe(trade_log), eq, merged_train.index, settings=cfg.get("exploration", {}), signals=bt_signals)
        score = rs._score_objective(tm, objective_cfg)
        if (best is None) or (score > float(best["score"])):
            best = {
                "strategy": strategy,
                "score": float(score),
                "tm": tm,
                "models": wf_models,
                "feature_cols": feature_cols,
                "threshold": float(th),
            }
    if best is None:
        return None
    if int(best["tm"].get("trades", 0)) < min_trades:
        return None
    return best


def run_roll(cfg: Dict[str, Any], fns: rs.PipelineFns, symbol: str, ltf_tf: str, train_years: int, start_year: int, end_year: int) -> pd.DataFrame:
    sym_cfg = rs._pair_cfg(symbol, cfg)
    h1_tf = str(cfg.get("htf_timeframes", {}).get("h1", "H1"))
    d1_tf = str(cfg.get("htf_timeframes", {}).get("d1", "D1"))
    tz_name = str(cfg.get("timezone", "Europe/London"))
    regime_cfg = sym_cfg.get("regime", cfg.get("regime", {}))
    variants = cfg.get("exploration", {}).get("regime_variants", [])
    if not variants:
        variants = rs._variant_grid(regime_cfg.get("thresholds", {}), max_variants=int(cfg.get("exploration", {}).get("max_variants", 8)))
    objective_cfg = cfg.get("exploration", {}).get("objective", {})
    min_trades = int(cfg.get("exploration", {}).get("min_trades", 30))

    df_ltf = rs._ensure_utc_index(fns.load_ohlcv(symbol, ltf_tf))
    df_h1 = rs._ensure_utc_index(fns.load_ohlcv(symbol, h1_tf))
    df_d1 = rs._ensure_utc_index(fns.load_ohlcv(symbol, d1_tf))
    ltf_features_all = rs._ensure_utc_index(fns.make_features(df_ltf, ltf_tf, sym_cfg)).reindex(df_ltf.index)
    sess_all = rs._session_flags(df_ltf.index, tz_name, sym_cfg.get("sessions", {}))
    base_all = pd.concat([df_ltf, ltf_features_all, sess_all], axis=1)
    atr_all = rs._prepare_atr_series(df_ltf, ltf_features_all)
    y_raw = fns.make_labels(df_ltf, atr_all, sym_cfg.get("barrier", {}))
    y_all, _ = rs._parse_labels(y_raw, df_ltf.index)

    rows: List[Dict[str, Any]] = []
    for test_year in range(start_year, end_year + 1):
        tr_start = pd.Timestamp(f"{test_year - train_years}-01-01", tz="UTC")
        tr_end = pd.Timestamp(f"{test_year - 1}-12-31 23:59:59", tz="UTC")
        te_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
        te_end = pd.Timestamp(f"{test_year}-12-31 23:59:59", tz="UTC")

        df_train = _slice(df_ltf, tr_start, tr_end)
        df_test = _slice(df_ltf, te_start, te_end)
        if df_train.empty or df_test.empty:
            continue
        lf_train = ltf_features_all.reindex(df_train.index)
        lf_test = ltf_features_all.reindex(df_test.index)
        base_train = base_all.reindex(df_train.index)
        base_test = base_all.reindex(df_test.index)
        y_train = y_all.reindex(df_train.index).dropna()

        best_candidate: Optional[Dict[str, Any]] = None
        for variant in variants:
            regime_all = rs._compute_regime_features(df_ltf, df_h1, df_d1, regime_cfg, variant)
            merged_train = pd.concat([base_train, regime_all.reindex(base_train.index)], axis=1)
            merged_test = pd.concat([base_test, regime_all.reindex(base_test.index)], axis=1)
            best_for_variant: Optional[Dict[str, Any]] = None
            for strategy in rs.STRATEGY_NAMES:
                cand = _score_strategy_on_train(
                    fns=fns,
                    cfg=cfg,
                    sym_cfg=sym_cfg,
                    symbol=symbol,
                    ltf_tf=ltf_tf,
                    merged_train=merged_train,
                    ltf_features_train=lf_train,
                    y_train=y_train,
                    strategy=strategy,
                    objective_cfg=objective_cfg,
                    min_trades=min_trades,
                )
                if cand is None:
                    continue
                cand["variant"] = variant
                cand["merged_test"] = merged_test
                cand["lf_test"] = lf_test
                if (best_for_variant is None) or (float(cand["score"]) > float(best_for_variant["score"])):
                    best_for_variant = cand
            if best_for_variant is not None and ((best_candidate is None) or (float(best_for_variant["score"]) > float(best_candidate["score"]))):
                best_candidate = best_for_variant

        if best_candidate is None:
            rows.append({"test_year": test_year, "status": "no_candidate"})
            continue

        merged_test = best_candidate["merged_test"]
        strategy = str(best_candidate["strategy"])
        sig_test = rs._generate_strategy_signals(
            merged_test,
            strategy,
            {"ltf_seconds": rs._timeframe_seconds(ltf_tf), **sym_cfg.get("strategies", {}).get(strategy, {})},
        )
        if not sig_test.empty:
            allowed = set(rs.STRATEGY_ALLOWED_REGIMES.get(strategy, []))
            if allowed:
                sig_test = sig_test.join(merged_test[["regime"]], how="left")
                sig_test = sig_test[sig_test["regime"].isin(allowed)]
            sig_test = rs._apply_pair_filters(sig_test, merged_test, symbol, strategy, sym_cfg)
            sig_test = rs._refine_low_tf_entries(sig_test, merged_test, ltf_tf, sym_cfg)
        X_test, _, _ = _build_xy(merged_test, best_candidate["lf_test"], sig_test, y_all.reindex(merged_test.index).fillna(0).astype(int))
        if X_test.empty or sig_test.empty:
            rows.append({"test_year": test_year, "status": "no_test_signals", "chosen_strategy": strategy})
            continue
        sig_test = sig_test.loc[X_test.index]
        probs = _predict_proba(best_candidate["models"], X_test)
        nt_cfg = dict(sym_cfg.get("no_trade", {}))
        nt_cfg["probability_threshold"] = float(best_candidate["threshold"])
        bt_signals = rs._build_signals_for_backtest(sig_test, probs, nt_cfg)
        if bt_signals.empty:
            rows.append({"test_year": test_year, "status": "no_bt_signals", "chosen_strategy": strategy})
            continue
        bt_raw = fns.backtest_from_signals(df_test, bt_signals.loc[bt_signals.index.intersection(df_test.index)], sym_cfg.get("cost", {}))
        if isinstance(bt_raw, tuple) and len(bt_raw) >= 2:
            trade_log, eq = bt_raw[0], bt_raw[1]
        else:
            trade_log = bt_raw.get("trade_log", pd.DataFrame())
            eq = bt_raw.get("equity_curve", pd.Series(dtype=float))
        tm = rs._trade_metrics(rs._to_dataframe(trade_log), eq, df_test.index, settings=cfg.get("exploration", {}), signals=bt_signals)
        ret = np.nan
        if isinstance(eq, pd.Series) and len(eq) > 1:
            s0 = float(eq.iloc[0])
            s1 = float(eq.iloc[-1])
            if s0 != 0:
                ret = (s1 / s0) - 1.0
        rows.append(
            {
                "test_year": test_year,
                "status": "ok",
                "train_start": str(tr_start.date()),
                "train_end": str(tr_end.date()),
                "test_start": str(te_start.date()),
                "test_end": str(te_end.date()),
                "chosen_strategy": strategy,
                "chosen_variant": str(best_candidate["variant"].get("id", "v0")),
                "train_score": float(best_candidate["score"]),
                "trades": int(tm.get("trades", 0)),
                "return": float(ret) if pd.notna(ret) else np.nan,
                "expectancy_r": float(tm.get("expectancy_r", 0.0)),
                "net_expectancy_after_cost": float(tm.get("net_expectancy_after_cost", 0.0)),
                "profit_factor": float(tm.get("profit_factor", 0.0)),
                "win_rate": float(tm.get("win_rate", 0.0)),
                "max_dd": float(tm.get("max_dd", 0.0)),
                "sharpe_trade": float(tm.get("sharpe_trade", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Rolling walk-forward (train N years, test next year) regime-strategy evaluation.")
    p.add_argument("--pipeline-module", default="scripts.pipeline_compat_adapter")
    p.add_argument("--config", required=True)
    p.add_argument("--symbol", default="XAU_USD")
    p.add_argument("--ltf", default="M5")
    p.add_argument("--train-years", type=int, default=10)
    p.add_argument("--test-start-year", type=int, default=2017)
    p.add_argument("--test-end-year", type=int, default=2025)
    p.add_argument("--out-csv", default="data/research/rolling_wfo_xau.csv")
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    fns = rs._load_pipeline(args.pipeline_module)
    out = run_roll(
        cfg=cfg,
        fns=fns,
        symbol=args.symbol,
        ltf_tf=args.ltf,
        train_years=int(args.train_years),
        start_year=int(args.test_start_year),
        end_year=int(args.test_end_year),
    )
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(
        json.dumps(
            {
                "out_csv": str(out_path),
                "rows": int(len(out)),
                "ok_rows": int((out.get("status") == "ok").sum()) if not out.empty else 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

