from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.xau_calibration_thresholds import (
    ThresholdConfig,
    apply_session_threshold,
    build_cost_state_keys,
    fit_session_calibrator,
    fit_session_threshold,
    predict_calibrated_prob,
)
from scripts.xau_broker_execution import (
    BrokerExecutionConfig,
    build_broker_setup_snapshot,
    simulate_one_bar_portfolio_step,
)
from scripts.xau_dependence import (
    compute_codrawdown_metrics,
    compute_pnl_dependence,
    compute_signal_dependence,
    compute_trade_overlap,
)
from scripts.xau_feature_engineering import (
    build_feature_registry,
    build_interaction_features,
    build_session_features,
    build_shared_features,
)
from scripts.xau_htf_scaling import align_htf_features, fit_feature_scalers, transform_feature_scalers
from scripts.xau_labeling import LabelConfig, build_session_conditioned_labels
from scripts.xau_portfolio_control import combine_session_outputs
from scripts.xau_session_anchors import (
    build_session_anchors,
    detect_current_session_future_extreme_leakage,
)
from scripts.xau_session_ingestion import default_session_config, load_ohlcv
from scripts.xau_tradability import build_tradable_mask, summarize_tradability
from scripts.xau_validation_monitoring import evaluate_portfolio_metrics, evaluate_sleeve_metrics
from scripts.xau_wfo_models import (
    SplitConfig,
    fit_session_head,
    fit_shared_trunk,
    make_purged_walk_forward_splits,
    predict_session_head,
    transform_shared_trunk,
)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _load_ohlcv_generic(path: str) -> pd.DataFrame:
    """HTF-safe loader (no fixed 15m cadence assumption)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HTF input file not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported HTF file format: {p.suffix}")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    req = ["timestamp", "open", "high", "low", "close"]
    miss = [c for c in req if c not in raw.columns]
    if miss:
        raise ValueError(f"Malformed HTF input: missing columns {miss}")
    ts = pd.to_datetime(raw["timestamp"], errors="coerce", utc=False)
    if bool(ts.isna().any()):
        raise ValueError("Malformed HTF input: invalid timestamp values.")
    if ts.dt.tz is None:
        raise ValueError("HTF timestamp must be timezone-aware.")
    out = raw.copy()
    out["timestamp"] = ts.dt.tz_convert("UTC")
    if bool(out["timestamp"].duplicated().any()):
        raise ValueError("HTF input contains duplicate timestamps.")
    out = out.sort_values("timestamp")
    if not bool(out["timestamp"].is_monotonic_increasing):
        raise ValueError("HTF timestamp must be monotonic increasing.")
    return out.set_index("timestamp")


def _load_main_ohlcv(path: str) -> pd.DataFrame:
    """Load LTF OHLCV for research without enforcing uninterrupted 15m cadence.

    Real broker history contains weekend/holiday/session gaps by design.
    """
    try:
        return load_ohlcv(path)
    except Exception:
        x = _load_ohlcv_generic(path)
        need = [c for c in ["open", "high", "low", "close"] if c in x.columns]
        if len(need) < 4:
            raise
        if "volume" not in x.columns:
            x["volume"] = 0.0
        return x[["open", "high", "low", "close", "volume"]].copy()


def _pairs_to_matrix(pairs: List[Dict[str, Any]], value_key: str) -> pd.DataFrame:
    sleeves = sorted(set([r["a"] for r in pairs] + [r["b"] for r in pairs]))
    if not sleeves:
        return pd.DataFrame()
    m = pd.DataFrame(np.eye(len(sleeves)), index=sleeves, columns=sleeves, dtype=float)
    for r in pairs:
        a = r["a"]
        b = r["b"]
        v = float(r.get(value_key, np.nan))
        m.loc[a, b] = v
        m.loc[b, a] = v
    return m


def _build_dependence_mats_from_history(
    pred_map: Dict[str, pd.Series],
    sig_map: Dict[str, pd.Series],
    pnl_map: Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    dep_signal = compute_signal_dependence(pd.DataFrame(pred_map)) if pred_map else {"pairs": []}
    dep_trigger = compute_trade_overlap(pd.DataFrame(sig_map).fillna(0)) if sig_map else {"pairs": []}
    dep_pnl = compute_pnl_dependence(pd.DataFrame(pnl_map).fillna(0.0)) if pnl_map else {"pairs": []}
    dep_cod = compute_codrawdown_metrics(pd.DataFrame(pnl_map).fillna(0.0)) if pnl_map else {"pairs": []}
    return {
        "score_corr": _pairs_to_matrix(dep_signal.get("pairs", []), "pearson"),
        "trigger_overlap": _pairs_to_matrix(dep_trigger.get("pairs", []), "jaccard"),
        "pnl_corr": _pairs_to_matrix(dep_pnl.get("pairs", []), "trade_pearson"),
        "coloss_freq": _pairs_to_matrix(dep_cod.get("pairs", []), "simultaneous_worst_decile_freq"),
    }


def _spread_state_proxy(
    df: pd.DataFrame,
    default_spread_bps: float,
    spread_unit: str,
) -> pd.Series:
    """Return spread proxy in configured unit for threshold cost-state bucketing."""
    spread_bps = pd.to_numeric(df.get("sf_spread_proxy", pd.Series(index=df.index, data=np.nan)), errors="coerce").fillna(
        float(default_spread_bps)
    )
    unit = str(spread_unit).strip().lower()
    if unit in {"usd", "usd_oz", "dollar", "dollar_per_oz"}:
        close = pd.to_numeric(df.get("close", pd.Series(index=df.index, data=np.nan)), errors="coerce").fillna(0.0)
        return close * (spread_bps * 1e-4)
    return spread_bps


def _tail_series_map(m: Dict[str, pd.Series], max_len: int | None) -> Dict[str, pd.Series]:
    if not max_len or int(max_len) <= 0:
        return m
    k = int(max_len)
    return {s: v.iloc[-k:] for s, v in m.items()}


def _default_label_cfg() -> LabelConfig:
    sessions = ["asia", "pre_london", "london_open", "london_continuation", "ny_open", "ny_overlap_postdata"]
    tp = {
        "asia": 1.2,
        "pre_london": 1.3,
        "london_open": 1.7,
        "london_continuation": 1.5,
        "ny_open": 1.6,
        "ny_overlap_postdata": 1.4,
    }
    sl = {
        "asia": 1.1,
        "pre_london": 1.2,
        "london_open": 1.3,
        "london_continuation": 1.2,
        "ny_open": 1.4,
        "ny_overlap_postdata": 1.2,
    }
    hz = {
        "asia": 6,
        "pre_london": 7,
        "london_open": 10,
        "london_continuation": 9,
        "ny_open": 9,
        "ny_overlap_postdata": 8,
    }
    nb = {
        "asia": 0.00015,
        "pre_london": 0.00015,
        "london_open": 0.00025,
        "london_continuation": 0.0002,
        "ny_open": 0.0002,
        "ny_overlap_postdata": 0.0002,
    }
    return LabelConfig(
        tp_mult_by_session=tp,
        sl_mult_by_session=sl,
        horizon_by_session=hz,
        neutral_band_by_session=nb,
        vol_col="atr14",
        event_mode="include",
        spread_col="spread_proxy_bps",
    )


def _merge_session_map(base: Dict[str, Any], overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(base)
    if overrides:
        for k, v in overrides.items():
            out[str(k)] = v
    return out


def _build_label_cfg(config: Dict[str, Any]) -> LabelConfig:
    base = _default_label_cfg()
    lc = dict(config.get("label_config", {}))
    broker = dict(config.get("broker_execution", {}))
    sessions = ["asia", "pre_london", "london_open", "london_continuation", "ny_open", "ny_overlap_postdata"]
    tp = _merge_session_map(base.tp_mult_by_session, lc.get("tp_mult_by_session"))
    sl = _merge_session_map(base.sl_mult_by_session, lc.get("sl_mult_by_session"))
    hz = _merge_session_map(base.horizon_by_session, lc.get("horizon_by_session"))
    nb = _merge_session_map(base.neutral_band_by_session, lc.get("neutral_band_by_session"))
    for s in sessions:
        if s not in tp or s not in sl or s not in hz or s not in nb:
            raise ValueError(f"Missing session-conditioned label params for '{s}'.")
    return LabelConfig(
        tp_mult_by_session={k: float(v) for k, v in tp.items()},
        sl_mult_by_session={k: float(v) for k, v in sl.items()},
        horizon_by_session={k: int(v) for k, v in hz.items()},
        neutral_band_by_session={k: float(v) for k, v in nb.items()},
        vol_col=str(lc.get("vol_col", base.vol_col)),
        event_col=str(lc.get("event_col", base.event_col)),
        event_mode=str(lc.get("event_mode", base.event_mode)),
        spread_col=lc.get("spread_col", base.spread_col),
        slippage_bps=float(lc.get("slippage_bps", broker.get("slippage_bps", base.slippage_bps))),
        commission_bps_per_side=float(lc.get("commission_bps_per_side", broker.get("commission_bps_per_side", base.commission_bps_per_side))),
        spread_is_bps=bool(lc.get("spread_is_bps", base.spread_is_bps)),
        max_spread_for_exec=lc.get("max_spread_for_exec", base.max_spread_for_exec),
        min_net_edge=float(lc.get("min_net_edge", base.min_net_edge)),
        max_mae_mult=float(lc.get("max_mae_mult", base.max_mae_mult)),
        time_bucket_edges=tuple(lc.get("time_bucket_edges", base.time_bucket_edges)),
    )


def _broker_cfg_for_layer(base: BrokerExecutionConfig, layer: str) -> BrokerExecutionConfig:
    if layer == "gross_pre_cost":
        return BrokerExecutionConfig(
            starting_equity=base.starting_equity,
            leverage=base.leverage,
            stop_out_margin_level_pct=base.stop_out_margin_level_pct,
            max_margin_utilization_pct=base.max_margin_utilization_pct,
            spread_bps=0.0,
            slippage_bps=0.0,
            commission_bps_per_side=0.0,
        )
    if layer == "after_spread":
        return BrokerExecutionConfig(
            starting_equity=base.starting_equity,
            leverage=base.leverage,
            stop_out_margin_level_pct=base.stop_out_margin_level_pct,
            max_margin_utilization_pct=base.max_margin_utilization_pct,
            spread_bps=base.spread_bps,
            slippage_bps=0.0,
            commission_bps_per_side=0.0,
        )
    if layer == "after_spread_slippage":
        return BrokerExecutionConfig(
            starting_equity=base.starting_equity,
            leverage=base.leverage,
            stop_out_margin_level_pct=base.stop_out_margin_level_pct,
            max_margin_utilization_pct=base.max_margin_utilization_pct,
            spread_bps=base.spread_bps,
            slippage_bps=base.slippage_bps,
            commission_bps_per_side=0.0,
        )
    return base


def _prep_features(df: pd.DataFrame, config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sess_cfg = default_session_config(config.get("timezone", "Europe/London"))
    sess_df = build_session_anchors(df, sess_cfg)
    base = df.join(sess_df)
    shared = build_shared_features(base, config=config)
    sess_blocks = build_session_features(base, config=config)
    sess_union = pd.DataFrame(index=base.index)
    for b in sess_blocks.values():
        sess_union = sess_union.combine_first(b)
    feat = base.join(shared).join(sess_union)
    feat = feat.join(build_interaction_features(feat))

    # Optional HTF alignment
    for src in config.get("htf_sources", []):
        p = src["path"]
        cols = list(src.get("cols", []))
        prefix = str(src.get("prefix", "htf"))
        htf = _load_ohlcv_generic(p)
        aligned = align_htf_features(df, htf, cols).add_prefix(f"{prefix}_")
        feat = feat.join(aligned)

    registry = build_feature_registry(shared, sess_blocks, feat[[c for c in feat.columns if c.startswith("if_")]])
    context_diag = {
        "temporal_context_mode": str(config.get("temporal_context_mode", "hybrid")).lower(),
        "short_context_feature_count": int(len([c for c in shared.columns if c.startswith("sf_ctx_")])),
        "slow_regime_feature_count": int(len([c for c in shared.columns if c.startswith("sf_regime_")])),
        "sequence_feature_count": int(len([c for c in shared.columns if c.startswith("sf_seq_")])),
        "session_context_feature_count": int(
            len([c for c in sess_union.columns if c.startswith("ss_ctx_")])
        ),
        "total_shared_feature_count": int(len([c for c in shared.columns if c.startswith("sf_")])),
        "total_session_feature_count": int(len([c for c in sess_union.columns if c.startswith("ss_")])),
    }
    return feat, registry, context_diag


def run_full_research_pipeline(data_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level deterministic XAU multi-session research pipeline orchestrator."""
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    out_dir = Path(config.get("output_dir", "data/research/xau_full_pipeline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_main_ohlcv(data_path)
    feat, registry, context_diag = _prep_features(df, config)

    # Leakage check on session anchors.
    leak_chk = detect_current_session_future_extreme_leakage(df.join(feat[[c for c in feat.columns if c in ["session_id", "current_session_developing_high", "current_session_developing_low"]]]))
    leakage_checks = {
        "session_anchor_future_extreme_leak_rows": int(len(leak_chk)),
        "feature_columns_contain_label_prefix": bool(any(c.startswith("y_") for c in feat.columns)),
    }

    # Labels (strictly separate from feature creation).
    lbl_cfg = _build_label_cfg(config)
    broker_cfg_for_labels = dict(config.get("broker_execution", {}))
    default_spread_bps = float(broker_cfg_for_labels.get("spread_bps", 2.0))
    lab_in = feat[["close", "high", "low", "session_bucket"]].copy()
    lab_in["atr14"] = pd.to_numeric((df["high"] - df["low"]).rolling(14, min_periods=14).mean(), errors="coerce")
    if "sf_spread_proxy" in feat.columns:
        lab_in["spread_proxy_bps"] = pd.to_numeric(feat["sf_spread_proxy"], errors="coerce").fillna(default_spread_bps)
    else:
        lab_in["spread_proxy_bps"] = pd.Series(default_spread_bps, index=feat.index, dtype=float)
    if "event_window_flag" in feat.columns:
        lab_in["event_window_flag"] = feat["event_window_flag"]
    labels = build_session_conditioned_labels(lab_in, lbl_cfg)
    feat["y_true"] = (pd.to_numeric(labels["y_dir"], errors="coerce").fillna(0) > 0).astype(int)
    feat["y_meta_exec"] = pd.to_numeric(labels["y_meta_exec"], errors="coerce").fillna(0).astype(int)
    feat["label_state"] = labels["label_state"].astype(str)
    feat["label_horizon_bars"] = pd.to_numeric(labels["label_horizon_bars"], errors="coerce").fillna(0).astype(int)
    feat["time_to_resolution_bucket"] = labels["time_to_resolution_bucket"].astype(str)
    feat["label_end_ts"] = pd.Series(feat.index, index=feat.index).shift(-int(config.get("max_label_horizon_bars", 8)))
    feat["regime_bucket"] = feat.get("regime_bucket", pd.Series("unknown", index=feat.index))

    # Remove warmup rows to avoid zero-filled early-window artifacts in train/validation.
    warmup_bars = int(config.get("warmup_bars", 200))
    if warmup_bars > 0 and len(feat) > warmup_bars:
        feat = feat.iloc[warmup_bars:].copy()

    trad_cfg = dict(config.get("tradability", {}))
    trad_cfg.setdefault("default_spread_bps", float(config.get("broker_execution", {}).get("spread_bps", 2.0)))
    trad_cfg.setdefault("spread_is_bps", True)
    trad_cols = build_tradable_mask(feat, trad_cfg)
    feat = feat.join(trad_cols)
    tradability_summary_global = summarize_tradability(
        feat[["tradable_mask", "tradability_score"]], session=feat["session_bucket"]
    )

    split_cfg = SplitConfig(
        n_splits=int(config.get("n_splits", 3)),
        val_size=int(config.get("val_size", 180)),
        max_label_horizon_bars=int(config.get("max_label_horizon_bars", 8)),
        embargo_bars=int(config.get("embargo_bars", 0)),
        min_train_size=int(config.get("min_train_size", 240)),
        label_end_col="label_end_ts",
        split_mode=str(config.get("split_mode", "tail")),
        rolling_train_size=(int(config["rolling_train_size"]) if config.get("rolling_train_size") is not None else None),
        anchor_train_start=config.get("anchor_train_start"),
        validation_years=([int(y) for y in config.get("validation_years", [])] if config.get("validation_years") is not None else None),
    )
    splits = make_purged_walk_forward_splits(feat, split_cfg)
    sleeves = list(config.get("sleeves", ["asia", "london_open", "ny_open"]))
    if bool(config.get("force_all_sessions_for_heads", False)) and str(config.get("mode", "prod")).lower() != "test":
        raise ValueError("force_all_sessions_for_heads is test-only and cannot be used in prod mode.")

    fold_results = []
    all_portfolio_pnl = []
    all_portfolio_decisions = []
    all_sleeve_eval = []
    sleeve_realized_contrib_all: Dict[str, List[pd.Series]] = {s: [] for s in sleeves}
    all_threshold_usage_rows: List[Dict[str, Any]] = []

    # Feature groups for scaling.
    shared_cols = [c for c in feat.columns if c.startswith("sf_") or c.startswith("if_") or c.startswith("htf_")]
    session_cols = [c for c in feat.columns if c.startswith("ss_")]
    schema_cols = ["session_bucket"] + shared_cols + session_cols

    for sp in splits:
        fold = int(sp["fold"])
        fdir = out_dir / f"fold_{fold:02d}"
        fdir.mkdir(parents=True, exist_ok=True)

        tr = np.array(sp["train_indices"], dtype=int)
        va = np.array(sp["validation_indices"], dtype=int)
        tr_df = feat.iloc[tr].copy()
        va_df = feat.iloc[va].copy()

        tr_x = tr_df[schema_cols].copy()
        va_x = va_df[schema_cols].copy()
        # Deterministic model matrix hygiene: keep raw df unchanged, but avoid NaN-only scaler/model inputs.
        tr_x[shared_cols + session_cols] = tr_x[shared_cols + session_cols].fillna(0.0)
        va_x[shared_cols + session_cols] = va_x[shared_cols + session_cols].fillna(0.0)
        scaler = fit_feature_scalers(tr_x, {"global": shared_cols, "session_sensitive": session_cols}, by_session=True)
        tr_t = transform_feature_scalers(tr_x, scaler)
        va_t = transform_feature_scalers(va_x, scaler)

        valid_shared = [c for c in shared_cols if c in tr_t.columns and tr_t[c].notna().any()]
        if len(valid_shared) == 0:
            continue

        # Shared trunk on shared features.
        trunk = fit_shared_trunk(tr_t[valid_shared].fillna(0.0), tr_df["y_true"], seed=seed + fold)
        z_tr = transform_shared_trunk(trunk, tr_t[valid_shared].fillna(0.0))
        z_va = transform_shared_trunk(trunk, va_t[valid_shared].fillna(0.0))

        spread_state_unit = str(config.get("threshold_cost_state_spread_unit", "usd_oz"))
        spread_edges = config.get("threshold_cost_state_spread_edges")
        if spread_edges is None and spread_state_unit.strip().lower() in {"usd", "usd_oz", "dollar", "dollar_per_oz"}:
            # OANDA XAU/USD practical default profile:
            # baseline around 0.20, degraded around 0.30, stress around 0.50 USD/oz.
            spread_edges = [0.20, 0.30, 0.50]
        threshold_cfg = ThresholdConfig(
            min_threshold=float(config.get("min_threshold", 0.5)),
            max_threshold=float(config.get("max_threshold", 0.9)),
            step=float(config.get("threshold_step", 0.05)),
            min_trades=int(config.get("min_trades", 50)),
            min_long_trades=int(config.get("min_long_trades", config.get("min_trades", 50))),
            min_short_trades=int(config.get("min_short_trades", config.get("min_trades", 50))),
            min_ev=float(config.get("threshold_min_ev", 0.0)),
            cost_conditioning_enabled=bool(config.get("threshold_cost_conditioning_enabled", True)),
            cost_state_min_samples=int(config.get("threshold_cost_state_min_samples", 80)),
            cost_state_max_adjustment=float(config.get("threshold_cost_state_max_adjustment", 0.06)),
            cost_state_spread_buckets=int(config.get("threshold_cost_state_spread_buckets", 3)),
            cost_state_spread_atr_buckets=int(config.get("threshold_cost_state_spread_atr_buckets", 3)),
            cost_state_use_calibration_bucket=bool(config.get("threshold_cost_state_use_calibration_bucket", False)),
            cost_state_spread_edges=(list(spread_edges) if spread_edges is not None else None),
            cost_state_spread_atr_edges=(
                list(config.get("threshold_cost_state_spread_atr_edges"))
                if config.get("threshold_cost_state_spread_atr_edges") is not None
                else None
            ),
        )

        head_models: Dict[str, Any] = {}
        meta_head_models: Dict[str, Any] = {}
        calibrators: Dict[str, Any] = {}
        meta_calibrators: Dict[str, Any] = {}
        thresholds: Dict[str, Any] = {}
        val_pred_map: Dict[str, pd.Series] = {}
        val_raw_sign_map: Dict[str, pd.Series] = {}
        val_sig_map: Dict[str, pd.Series] = {}
        val_meta_prob_map: Dict[str, pd.Series] = {}
        val_pnl_map: Dict[str, pd.Series] = {}
        sleeve_eval_fold = []
        train_pred_hist: Dict[str, pd.Series] = {}
        train_sig_hist: Dict[str, pd.Series] = {}
        train_pnl_hist: Dict[str, pd.Series] = {}
        threshold_usage_rows: List[Dict[str, Any]] = []

        for s in sleeves:
            if bool(config.get("force_all_sessions_for_heads", False)):
                tr_mask = pd.Series(True, index=tr_df.index)
                va_mask = pd.Series(True, index=va_df.index)
            else:
                tr_mask = tr_df["session_bucket"].astype(str).eq(s)
                va_mask = va_df["session_bucket"].astype(str).eq(s)
            if bool(config.get("use_tradable_mask_for_training", True)):
                tr_mask = tr_mask & tr_df["tradable_mask"].astype(bool)
            if int(tr_mask.sum()) < int(config.get("min_session_train_rows", 40)):
                continue
            valid_session_cols = [c for c in session_cols if c in tr_t.columns]
            xh_tr = tr_t.loc[tr_mask, valid_session_cols].fillna(0.0)
            xh_va = va_t.loc[va_mask, valid_session_cols].fillna(0.0)
            z_tr_s = z_tr.loc[tr_mask]
            z_va_s = z_va.loc[va_mask]
            y_tr_s = tr_df.loc[tr_mask, "y_true"]
            y_meta_tr_s = tr_df.loc[tr_mask, "y_meta_exec"]
            y_va_s = va_df.loc[va_mask, "y_true"]

            head = fit_session_head(s, z_tr_s, xh_tr, y_tr_s, seed=seed + 100 + fold)
            raw_tr = predict_session_head(head, z_tr_s, xh_tr).to_numpy(dtype=float)
            raw_va = predict_session_head(head, z_va_s, xh_va).to_numpy(dtype=float)
            raw_sign_va = np.where(raw_va >= 0.0, 1, -1).astype(int)
            cal = fit_session_calibrator(s, raw_tr, y_tr_s.to_numpy(dtype=float))
            p_tr = predict_calibrated_prob(cal, raw_tr)
            p_va = predict_calibrated_prob(cal, raw_va)
            meta_head = fit_session_head(s, z_tr_s, xh_tr, y_meta_tr_s, seed=seed + 200 + fold)
            raw_meta_tr = predict_session_head(meta_head, z_tr_s, xh_tr).to_numpy(dtype=float)
            raw_meta_va = predict_session_head(meta_head, z_va_s, xh_va).to_numpy(dtype=float)
            meta_cal = fit_session_calibrator(s, raw_meta_tr, y_meta_tr_s.to_numpy(dtype=float))
            p_meta_tr = predict_calibrated_prob(meta_cal, raw_meta_tr)
            p_meta_va = predict_calibrated_prob(meta_cal, raw_meta_va)
            # EV proxy from label direction in train.
            ev_tr = (2.0 * y_tr_s.to_numpy(dtype=float) - 1.0) * 0.01
            fit_mask = np.isfinite(p_tr) & np.isfinite(ev_tr)
            if bool(config.get("use_tradable_mask_for_threshold_fit", True)):
                fit_mask = fit_mask & tr_df.loc[tr_mask, "tradable_mask"].to_numpy(dtype=bool)
            if int(np.sum(fit_mask)) < int(threshold_cfg.min_trades):
                fit_mask = np.isfinite(p_tr) & np.isfinite(ev_tr)
            tr_spread_state = _spread_state_proxy(
                tr_df.loc[tr_mask],
                default_spread_bps=default_spread_bps,
                spread_unit=spread_state_unit,
            )
            tr_cost_state = pd.DataFrame(
                {
                    "session_bucket": tr_df.loc[tr_mask, "session_bucket"].astype(str).to_numpy(),
                    "spread_proxy": pd.to_numeric(tr_spread_state, errors="coerce").fillna(default_spread_bps).to_numpy(),
                    "spread_atr": pd.to_numeric(tr_df.loc[tr_mask, "tradability_spread_atr"], errors="coerce").fillna(0.0).to_numpy(),
                },
                index=y_tr_s.index,
            )
            th = fit_session_threshold(s, p_tr[fit_mask], ev_tr[fit_mask], threshold_cfg, cost_state_df=tr_cost_state.loc[fit_mask])
            va_spread_state = _spread_state_proxy(
                va_df.loc[va_mask],
                default_spread_bps=default_spread_bps,
                spread_unit=spread_state_unit,
            )
            va_cost_state = {
                "session_bucket": va_df.loc[va_mask, "session_bucket"].astype(str).to_numpy(),
                "spread_proxy": pd.to_numeric(va_spread_state, errors="coerce").fillna(default_spread_bps).to_numpy(),
                "spread_atr": pd.to_numeric(va_df.loc[va_mask, "tradability_spread_atr"], errors="coerce").fillna(0.0).to_numpy(),
            }
            sig_va = apply_session_threshold(s, p_va, th, cost_state=va_cost_state)
            meta_min_prob = float(config.get("meta_min_prob", 0.55))
            sig_va = np.where(p_meta_va >= meta_min_prob, sig_va, 0).astype(int)
            pnl_va = pd.Series(sig_va.astype(float), index=y_va_s.index) * ((2.0 * y_va_s.astype(float) - 1.0) * 0.01)
            tr_cost_state_apply = {
                "session_bucket": tr_df.loc[tr_mask, "session_bucket"].astype(str).to_numpy(),
                "spread_proxy": pd.to_numeric(tr_spread_state, errors="coerce").fillna(default_spread_bps).to_numpy(),
                "spread_atr": pd.to_numeric(tr_df.loc[tr_mask, "tradability_spread_atr"], errors="coerce").fillna(0.0).to_numpy(),
            }
            sig_tr = apply_session_threshold(s, p_tr, th, cost_state=tr_cost_state_apply)
            sig_tr = np.where(p_meta_tr >= meta_min_prob, sig_tr, 0).astype(int)
            pnl_tr = pd.Series(sig_tr.astype(float), index=y_tr_s.index) * ((2.0 * y_tr_s.astype(float) - 1.0) * 0.01)
            state_keys_va = build_cost_state_keys(
                threshold_obj=th,
                session_bucket=va_cost_state["session_bucket"],
                spread_proxy=va_cost_state["spread_proxy"],
                spread_atr=va_cost_state["spread_atr"],
            )
            for sk, g in pd.DataFrame(
                {"state_key": state_keys_va, "signal": sig_va, "session_bucket": va_cost_state["session_bucket"]}
            ).groupby(["session_bucket", "state_key"], sort=True):
                threshold_usage_rows.append(
                    {
                        "sleeve": s,
                        "session_bucket": str(sk[0]),
                        "cost_state_key": str(sk[1]),
                        "bars": int(len(g)),
                        "trade_bars": int((pd.to_numeric(g["signal"], errors="coerce").fillna(0).astype(int) != 0).sum()),
                    }
                )

            head_models[s] = head
            meta_head_models[s] = meta_head
            calibrators[s] = cal
            meta_calibrators[s] = meta_cal
            thresholds[s] = th
            val_pred_map[s] = pd.Series(p_va, index=y_va_s.index)
            val_raw_sign_map[s] = pd.Series(raw_sign_va, index=y_va_s.index)
            val_sig_map[s] = pd.Series(sig_va, index=y_va_s.index)
            val_meta_prob_map[s] = pd.Series(p_meta_va, index=y_va_s.index)
            val_pnl_map[s] = pnl_va
            train_pred_hist[s] = pd.Series(p_tr, index=y_tr_s.index)
            train_sig_hist[s] = pd.Series(sig_tr, index=y_tr_s.index)
            train_pnl_hist[s] = pnl_tr

            preds = pd.DataFrame(
                {
                    "prob": p_va,
                    "signal": sig_va,
                    "meta_prob": p_meta_va,
                    "threshold": th["threshold"],
                    "fold": fold,
                },
                index=y_va_s.index,
            )
            lbl = pd.DataFrame({"y_true": y_va_s, "regime_bucket": va_df.loc[va_mask, "regime_bucket"]}, index=y_va_s.index)
            sev = evaluate_sleeve_metrics(preds, lbl, pnl_va)
            sev["sleeve"] = s
            sev["fold"] = fold
            sleeve_eval_fold.append(sev)

        # timestamp-level final decisions using history available up to each timestamp only.
        sleeve_health = pd.DataFrame(
            [
                {
                    "sleeve": s,
                    "recent_brier": float(calibrators[s].get("diagnostics", {}).get("brier", np.nan)),
                }
                for s in calibrators.keys()
            ],
            columns=["sleeve", "recent_brier"],
        )
        portfolio_state = {"drawdown_pct": 0.0}
        alloc_cfg = dict(config.get("alloc_config", {}))
        broker_cfg = BrokerExecutionConfig(**dict(config.get("broker_execution", {})))
        attrib_layers = [
            "gross_pre_cost",
            "after_spread",
            "after_spread_slippage",
            "full_broker_pre_threshold",
            "after_threshold_gating",
            "after_tradability_filter",
        ]
        layer_cfg = {k: _broker_cfg_for_layer(broker_cfg, k) for k in attrib_layers}
        layer_equity = {k: float(layer_cfg[k].starting_equity) for k in attrib_layers}
        layer_pnl = {k: pd.Series(0.0, index=va_df.index) for k in attrib_layers}
        layer_trade_bars = {k: 0 for k in attrib_layers}
        layer_session_pnl = {k: {} for k in attrib_layers}
        equity = float(broker_cfg.starting_equity)
        peak_equity = equity
        portfolio_dec_rows = []
        portfolio_pnl = pd.Series(0.0, index=va_df.index)
        live_pred_hist = {k: v.copy() for k, v in train_pred_hist.items()}
        live_sig_hist = {k: v.copy() for k, v in train_sig_hist.items()}
        live_pnl_hist = {k: v.copy() for k, v in train_pnl_hist.items()}
        # Keep an independent raw-signal history path for apples-to-apples pre-threshold attribution.
        raw_live_sig_hist = {k: v.copy() for k, v in train_sig_hist.items()}
        raw_live_pnl_hist = {k: v.copy() for k, v in train_pnl_hist.items()}
        realized_sleeve_map = {s: pd.Series(0.0, index=va_df.index) for s in val_pred_map.keys()}
        dep_recompute_every = int(config.get("dep_recompute_every", 96))
        dep_history_bars = int(config.get("dep_history_bars", 4000))
        min_tradable_range_abs = float(config.get("min_tradable_range_abs", 1e-8))
        min_tradable_move_abs = float(config.get("min_tradable_move_abs", 1e-8))
        fold_tradability_summary = summarize_tradability(
            va_df[["tradable_mask", "tradability_score"]], session=va_df["session_bucket"]
        )
        dep_mats_cached = _build_dependence_mats_from_history(
            _tail_series_map(live_pred_hist, dep_history_bars),
            _tail_series_map(live_sig_hist, dep_history_bars),
            _tail_series_map(live_pnl_hist, dep_history_bars),
        )
        dep_mats_raw_cached = _build_dependence_mats_from_history(
            _tail_series_map(live_pred_hist, dep_history_bars),
            _tail_series_map(raw_live_sig_hist, dep_history_bars),
            _tail_series_map(raw_live_pnl_hist, dep_history_bars),
        )
        for i, ts in enumerate(va_df.index):
            if i == 0 or (dep_recompute_every > 0 and (i % dep_recompute_every == 0)):
                dep_mats_cached = _build_dependence_mats_from_history(
                    _tail_series_map(live_pred_hist, dep_history_bars),
                    _tail_series_map(live_sig_hist, dep_history_bars),
                    _tail_series_map(live_pnl_hist, dep_history_bars),
                )
                dep_mats_raw_cached = _build_dependence_mats_from_history(
                    _tail_series_map(live_pred_hist, dep_history_bars),
                    _tail_series_map(raw_live_sig_hist, dep_history_bars),
                    _tail_series_map(raw_live_pnl_hist, dep_history_bars),
                )
            rows = []
            gating_audit: Dict[str, Dict[str, Any]] = {}
            for s in val_pred_map:
                pr = val_pred_map[s].reindex([ts]).iloc[0] if ts in val_pred_map[s].index else np.nan
                raw_sg = int(val_raw_sign_map[s].reindex([ts]).fillna(0).iloc[0]) if s in val_raw_sign_map else 0
                sg = int(val_sig_map[s].reindex([ts]).fillna(0).iloc[0]) if s in val_sig_map else 0
                mp = float(val_meta_prob_map[s].reindex([ts]).fillna(0.0).iloc[0]) if s in val_meta_prob_map else 0.0
                ths = thresholds.get(s, {})
                side_disabled = bool((raw_sg > 0 and not bool(ths.get("long_enabled", True))) or (raw_sg < 0 and not bool(ths.get("short_enabled", True))))
                meta_block = bool(raw_sg != 0 and sg == 0 and mp < float(config.get("meta_min_prob", 0.55)))
                threshold_block = bool(raw_sg != 0 and sg == 0 and (not meta_block) and (not side_disabled))
                gating_audit[s] = {
                    "raw_signal": int(raw_sg),
                    "post_threshold_signal": int(sg),
                    "meta_prob": mp,
                    "meta_block": meta_block,
                    "side_disabled": side_disabled,
                    "threshold_block": threshold_block,
                }
                rows.append(
                    {
                        "sleeve": s,
                        "raw_signal": raw_sg,
                        "signal": sg,
                        "meta_prob": mp,
                        "prob": pr if np.isfinite(pr) else 0.5,
                        "payoff_estimate": 0.02,
                        "cost_estimate": 0.001,
                        "cluster": "eu" if "london" in s or "asia" in s else "us",
                    }
                )
            sleeve_outputs = pd.DataFrame(
                rows,
                columns=["sleeve", "raw_signal", "signal", "meta_prob", "prob", "payoff_estimate", "cost_estimate", "cluster"],
            )
            # Confidence proxies for post-cost EV diagnostics:
            # use model probabilities/meta-edge, not size multiplier, to avoid collapsed deciles.
            so_prob = pd.to_numeric(sleeve_outputs.get("prob"), errors="coerce").fillna(0.5)
            so_meta = pd.to_numeric(sleeve_outputs.get("meta_prob"), errors="coerce").fillna(0.0)
            so_raw_sig = pd.to_numeric(sleeve_outputs.get("raw_signal"), errors="coerce").fillna(0).astype(int)
            raw_candidate_score = float(so_prob.loc[so_raw_sig != 0].max()) if bool((so_raw_sig != 0).any()) else float(so_prob.max())
            meta_edge_score = float((so_prob * so_meta).loc[so_raw_sig != 0].max()) if bool((so_raw_sig != 0).any()) else float((so_prob * so_meta).max())
            dec = combine_session_outputs(
                ts,
                sleeve_outputs,
                {
                    "dependence_mats": dep_mats_cached,
                    "alloc_config": alloc_cfg,
                    "sleeve_health": sleeve_health,
                    "portfolio_state": portfolio_state,
                },
            )
            raw_rows = sleeve_outputs.copy()
            raw_rows["signal"] = pd.to_numeric(raw_rows["raw_signal"], errors="coerce").fillna(0).astype(int)
            dec_raw = combine_session_outputs(
                ts,
                raw_rows[["sleeve", "signal", "prob", "payoff_estimate", "cost_estimate", "cluster"]],
                {
                    "dependence_mats": dep_mats_raw_cached,
                    "alloc_config": alloc_cfg,
                    "sleeve_health": sleeve_health,
                    "portfolio_state": portfolio_state,
                },
            )
            selected = dec["selected_sleeves"]
            sel_alloc = dec.get("selected_allocations", {})
            side_map = dec.get("selected_signals", {})
            if len(selected):
                sel_mask = sleeve_outputs["sleeve"].astype(str).isin([str(s) for s in selected])
                selected_prob_score = float(pd.to_numeric(sleeve_outputs.loc[sel_mask, "prob"], errors="coerce").fillna(0.5).max()) if bool(sel_mask.any()) else 0.0
            else:
                selected_prob_score = 0.0
            decision_score = selected_prob_score if len(selected) else raw_candidate_score
            close_t = float(va_df["close"].iloc[i])
            high_t = float(va_df["high"].iloc[i])
            low_t = float(va_df["low"].iloc[i])
            next_close = float(va_df["close"].iloc[i + 1]) if i + 1 < len(va_df) else float(va_df["close"].iloc[i])
            bar_range = abs(high_t - low_t)
            next_move = abs(next_close - close_t)
            tradable_bar = bool(
                va_df["tradable_mask"].iloc[i]
                and (bar_range > min_tradable_range_abs)
                and (next_move > min_tradable_move_abs)
            )
            if not tradable_bar:
                dec["final_action"] = "NO_TRADE"
                dec["selected_sleeves"] = []
                dec["selected_allocations"] = {}
                dec["selected_signals"] = {}
                dec["size_multiplier"] = 0.0
                selected = []
                sel_alloc = {}
                side_map = {}
                reasons = dict(dec.get("suppression_reasons", {}))
                reasons["market_state"] = "non_tradable_flat_bar"
                dec["suppression_reasons"] = reasons
            dec["gating_audit"] = gating_audit
            if dec.get("final_action") == "NO_TRADE":
                if not tradable_bar:
                    dec["abstain_reason"] = "tradability_block"
                else:
                    has_meta_block = any(bool(v.get("meta_block", False)) for v in gating_audit.values())
                    has_side_disabled = any(bool(v.get("side_disabled", False)) for v in gating_audit.values())
                    has_threshold_block = any(bool(v.get("threshold_block", False)) for v in gating_audit.values())
                    if has_meta_block:
                        dec["abstain_reason"] = "meta_edge_block"
                    elif has_side_disabled:
                        dec["abstain_reason"] = "side_disabled"
                    elif has_threshold_block:
                        dec["abstain_reason"] = "cost_threshold_block"
                    else:
                        dec["abstain_reason"] = "no_candidate"
            else:
                dec["abstain_reason"] = ""

            layers_positions = {
                "gross_pre_cost": [
                    {"sleeve": s, "side": int(dec_raw.get("selected_signals", {}).get(s, 0)), "allocation": float(dec_raw.get("selected_allocations", {}).get(s, 0.0))}
                    for s in dec_raw.get("selected_sleeves", [])
                ],
                "after_spread": [
                    {"sleeve": s, "side": int(dec_raw.get("selected_signals", {}).get(s, 0)), "allocation": float(dec_raw.get("selected_allocations", {}).get(s, 0.0))}
                    for s in dec_raw.get("selected_sleeves", [])
                ],
                "after_spread_slippage": [
                    {"sleeve": s, "side": int(dec_raw.get("selected_signals", {}).get(s, 0)), "allocation": float(dec_raw.get("selected_allocations", {}).get(s, 0.0))}
                    for s in dec_raw.get("selected_sleeves", [])
                ],
                "full_broker_pre_threshold": [
                    {"sleeve": s, "side": int(dec_raw.get("selected_signals", {}).get(s, 0)), "allocation": float(dec_raw.get("selected_allocations", {}).get(s, 0.0))}
                    for s in dec_raw.get("selected_sleeves", [])
                ],
                "after_threshold_gating": [
                    {"sleeve": s, "side": int(dec.get("selected_signals", {}).get(s, 0)), "allocation": float(dec.get("selected_allocations", {}).get(s, 0.0))}
                    for s in dec.get("selected_sleeves", [])
                ],
                "after_tradability_filter": [
                    {"sleeve": s, "side": int(side_map.get(s, 0)), "allocation": float(sel_alloc.get(s, 0.0))}
                    for s in selected
                ],
            }
            if not tradable_bar:
                layers_positions["after_tradability_filter"] = []
            raw_layer_contrib: Dict[str, float] = {}
            for lk in attrib_layers:
                if len(layers_positions[lk]) > 0:
                    layer_trade_bars[lk] += 1
                sim_l = simulate_one_bar_portfolio_step(
                    equity_before=float(layer_equity[lk]),
                    close_t=close_t,
                    high_t=high_t,
                    low_t=low_t,
                    close_next=next_close,
                    sleeve_positions=layers_positions[lk],
                    cfg=layer_cfg[lk],
                )
                layer_equity[lk] = float(sim_l["equity_after"])
                layer_pnl[lk].loc[ts] = float(sim_l["pnl_cash"])
                sess_name = str(va_df["session_bucket"].iloc[i])
                sess_map = layer_session_pnl[lk].setdefault(sess_name, [])
                sess_map.append(float(sim_l["pnl_cash"]))
                if lk == "full_broker_pre_threshold":
                    raw_layer_contrib = {str(k): float(v) for k, v in sim_l.get("per_sleeve_pnl_cash", {}).items()}

            sim = simulate_one_bar_portfolio_step(
                equity_before=equity,
                close_t=close_t,
                high_t=high_t,
                low_t=low_t,
                close_next=next_close,
                sleeve_positions=[
                    {"sleeve": s, "side": int(side_map.get(s, 0)), "allocation": float(sel_alloc.get(s, 0.0))}
                    for s in selected
                ],
                cfg=broker_cfg,
            )
            realized = float(sim["pnl_cash"])
            equity = float(sim["equity_after"])
            peak_equity = max(peak_equity, equity)
            dd_pct = 0.0 if peak_equity <= 0 else max(0.0, (peak_equity - equity) / peak_equity)
            portfolio_state["drawdown_pct"] = float(dd_pct)

            for s in val_pred_map:
                contrib = float(sim["per_sleeve_pnl_cash"].get(s, 0.0))
                if s in realized_sleeve_map:
                    realized_sleeve_map[s].loc[ts] = contrib

                # Update online histories using only information available after this bar.
                if s not in live_pred_hist:
                    live_pred_hist[s] = pd.Series(dtype=float)
                if s not in live_sig_hist:
                    live_sig_hist[s] = pd.Series(dtype=float)
                if s not in live_pnl_hist:
                    live_pnl_hist[s] = pd.Series(dtype=float)
                live_pred_hist[s].loc[ts] = float(val_pred_map[s].reindex([ts]).fillna(0.5).iloc[0])
                live_sig_hist[s].loc[ts] = int(val_sig_map[s].reindex([ts]).fillna(0).iloc[0])
                live_pnl_hist[s].loc[ts] = float(contrib)
                raw_live_sig_hist[s].loc[ts] = int(val_raw_sign_map[s].reindex([ts]).fillna(0).iloc[0])
                raw_live_pnl_hist[s].loc[ts] = float(raw_layer_contrib.get(s, 0.0))
            portfolio_pnl.loc[ts] = realized
            portfolio_dec_rows.append(
                {
                    "timestamp": ts,
                    **dec,
                    "realized_pnl": realized,
                    "equity_after": equity,
                    "used_margin": float(sim["used_margin"]),
                    "margin_level_pct": float(sim["margin_level_pct"]),
                    "stopout_triggered": bool(sim["stopout_triggered"]),
                    "tradable_mask": bool(va_df["tradable_mask"].iloc[i]),
                    "tradability_score": float(va_df["tradability_score"].iloc[i]),
                    "raw_selected_sleeves": dec_raw.get("selected_sleeves", []),
                    "raw_selected_signals": dec_raw.get("selected_signals", {}),
                    "raw_selected_allocations": dec_raw.get("selected_allocations", {}),
                    "raw_candidate_score": float(raw_candidate_score),
                    "meta_edge_score": float(meta_edge_score),
                    "selected_prob_score": float(selected_prob_score),
                    "decision_score": float(decision_score),
                }
            )
        all_portfolio_pnl.append(portfolio_pnl)
        all_portfolio_decisions.extend(portfolio_dec_rows)
        all_sleeve_eval.extend(sleeve_eval_fold)
        dec_fold_df = pd.DataFrame(portfolio_dec_rows)
        session_metrics_fold: List[Dict[str, Any]] = []
        if len(dec_fold_df):
            dec_fold_df["session_bucket"] = va_df.reindex(pd.to_datetime(dec_fold_df["timestamp"], utc=True, errors="coerce"))["session_bucket"].to_numpy()
            dec_fold_df["realized_pnl"] = pd.to_numeric(dec_fold_df["realized_pnl"], errors="coerce").fillna(0.0)
            dec_fold_df["is_trade"] = dec_fold_df["final_action"].astype(str).ne("NO_TRADE")
            dec_fold_df["is_abstain"] = ~dec_fold_df["is_trade"]
            eq = pd.to_numeric(dec_fold_df["realized_pnl"], errors="coerce").fillna(0.0).cumsum()
            dd = eq - eq.cummax()
            dec_fold_df["drawdown"] = dd
            for sess, g in dec_fold_df.groupby("session_bucket", sort=True):
                session_metrics_fold.append(
                    {
                        "fold": fold,
                        "session_bucket": str(sess),
                        "bars": int(len(g)),
                        "trade_count": int(g["is_trade"].sum()),
                        "abstain_rate": float(g["is_abstain"].mean()),
                        "expectancy_after_cost": float(g["realized_pnl"].mean()),
                        "pnl_sum": float(g["realized_pnl"].sum()),
                        "drawdown_contribution": float(g.loc[g["drawdown"] < 0, "realized_pnl"].sum()),
                    }
                )
        threshold_usage_df = pd.DataFrame(threshold_usage_rows)
        if len(threshold_usage_df):
            threshold_usage_df = (
                threshold_usage_df.groupby(["sleeve", "session_bucket", "cost_state_key"], as_index=False)
                .agg(bars=("bars", "sum"), trade_bars=("trade_bars", "sum"))
            )
            threshold_usage_df["trade_freq"] = threshold_usage_df["trade_bars"] / threshold_usage_df["bars"].clip(lower=1)
            all_threshold_usage_rows.extend(threshold_usage_df.to_dict(orient="records"))

        # Persist fold artifacts.
        _save_json(fdir / "feature_schema.json", {"column_order": schema_cols})
        _save_json(fdir / "feature_registry.json", registry.to_dict(orient="records"))
        _save_json(fdir / "scalers.json", scaler)
        _save_json(fdir / "shared_trunk.json", trunk)
        _save_json(fdir / "session_heads.json", head_models)
        _save_json(fdir / "meta_session_heads.json", meta_head_models)
        _save_json(fdir / "calibrators.json", calibrators)
        _save_json(fdir / "meta_calibrators.json", meta_calibrators)
        _save_json(fdir / "thresholds.json", thresholds)
        layer_summary = {
            k: {
                "pnl_sum": float(layer_pnl[k].sum()),
                "ending_equity": float(layer_equity[k]),
                "trade_bars": int(layer_trade_bars[k]),
                "session_pnl": {sn: float(np.sum(vals)) for sn, vals in layer_session_pnl[k].items()},
            }
            for k in attrib_layers
        }
        _save_json(fdir / "failure_decomposition.json", layer_summary)
        pd.DataFrame({k: v for k, v in layer_pnl.items()}).to_csv(fdir / "failure_decomposition_layers.csv", index=True)
        _save_json(fdir / "tradability_summary.json", fold_tradability_summary)
        dep_mats_final = _build_dependence_mats_from_history(
            _tail_series_map(live_pred_hist, dep_history_bars),
            _tail_series_map(live_sig_hist, dep_history_bars),
            _tail_series_map(live_pnl_hist, dep_history_bars),
        )
        for k, m in dep_mats_final.items():
            m.to_csv(fdir / f"{k}.csv")
        _save_json(fdir / "portfolio_controller_state.json", {"alloc_config": alloc_cfg, "sleeve_health": sleeve_health.to_dict(orient="records")})
        _save_json(fdir / "sleeve_eval.json", sleeve_eval_fold)
        _save_json(fdir / "session_validation_summary.json", session_metrics_fold)
        if len(threshold_usage_df):
            threshold_usage_df.to_csv(fdir / "threshold_usage_by_session_cost_bucket.csv", index=False)
        pd.DataFrame(portfolio_dec_rows).to_json(fdir / "portfolio_decisions.json", orient="records", date_format="iso")
        fold_results.append({"fold": fold, "n_sleeves": int(len(val_pred_map)), "n_decisions": int(len(portfolio_dec_rows))})
        for s, ser in realized_sleeve_map.items():
            sleeve_realized_contrib_all.setdefault(s, []).append(ser)

    if len(all_portfolio_pnl) == 0:
        return {"folds": 0, "output_dir": str(out_dir), "leakage_checks": leakage_checks}

    port_all = pd.concat(all_portfolio_pnl).sort_index()
    sleeve_pnl_map_all = {
        s: pd.concat(v).sort_index()
        for s, v in sleeve_realized_contrib_all.items()
        if len(v)
    }
    portfolio_summary = evaluate_portfolio_metrics(port_all, sleeve_pnl_map_all)
    dec_df = pd.DataFrame(all_portfolio_decisions)
    if len(dec_df):
        decision_score = pd.to_numeric(dec_df.get("decision_score", pd.Series(index=dec_df.index, data=np.nan)), errors="coerce")
        raw_candidate_score = pd.to_numeric(dec_df.get("raw_candidate_score", pd.Series(index=dec_df.index, data=np.nan)), errors="coerce")
        size_proxy = pd.to_numeric(dec_df.get("size_multiplier", pd.Series(index=dec_df.index, data=0.0)), errors="coerce").fillna(0.0)
        prob_proxy = decision_score.where(np.isfinite(decision_score), raw_candidate_score)
        prob_proxy = prob_proxy.where(np.isfinite(prob_proxy), size_proxy).fillna(0.0)
        net = pd.to_numeric(dec_df.get("realized_pnl", pd.Series(index=dec_df.index, data=0.0)), errors="coerce").fillna(0.0)
        d = pd.DataFrame({"p": prob_proxy.clip(0.0, 1.0), "net": net})
        if d["p"].nunique() <= 1:
            ev_deciles = []
        else:
            d["decile"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
            ev_deciles = [
                {
                    "decile": int(k),
                    "mean_proxy_score": float(g["p"].mean()),
                    "mean_net_pnl": float(g["net"].mean()),
                    "count": int(len(g)),
                }
                for k, g in d.groupby("decile", sort=True)
            ]
        retained = dec_df[dec_df["tradable_mask"].astype(bool)]
        excluded = dec_df[~dec_df["tradable_mask"].astype(bool)]
        retained_vs_excluded = {
            "retained_count": int(len(retained)),
            "excluded_count": int(len(excluded)),
            "retained_mean_pnl": float(pd.to_numeric(retained.get("realized_pnl", 0.0), errors="coerce").fillna(0.0).mean()) if len(retained) else 0.0,
            "excluded_mean_pnl": float(pd.to_numeric(excluded.get("realized_pnl", 0.0), errors="coerce").fillna(0.0).mean()) if len(excluded) else 0.0,
        }
        dec_ts = pd.to_datetime(dec_df.get("timestamp"), utc=True, errors="coerce")
        bucket_map = feat.reindex(dec_ts)["time_to_resolution_bucket"] if len(feat) else pd.Series(index=dec_df.index, dtype=object)
        sess_map = feat.reindex(dec_ts)["session_bucket"] if len(feat) else pd.Series(index=dec_df.index, dtype=object)
        bt = pd.DataFrame({"bucket": bucket_map.astype(str), "net": net})
        time_bucket_perf = [
            {"bucket": str(k), "mean_net_pnl": float(g["net"].mean()), "count": int(len(g))}
            for k, g in bt.groupby("bucket", sort=True)
            if str(k) not in {"nan", "None"}
        ]
        dec_df["session_bucket"] = sess_map.astype(str).to_numpy()
        dec_df["is_trade"] = dec_df["final_action"].astype(str).ne("NO_TRADE")
        dec_df["is_abstain"] = ~dec_df["is_trade"]
        session_validation = [
            {
                "session_bucket": str(s),
                "bars": int(len(g)),
                "trade_count": int(g["is_trade"].sum()),
                "abstain_rate": float(g["is_abstain"].mean()),
                "expectancy_after_cost": float(pd.to_numeric(g["realized_pnl"], errors="coerce").fillna(0.0).mean()),
                "pnl_sum": float(pd.to_numeric(g["realized_pnl"], errors="coerce").fillna(0.0).sum()),
            }
            for s, g in dec_df.groupby("session_bucket", sort=True)
        ]
    else:
        ev_deciles = []
        retained_vs_excluded = {"retained_count": 0, "excluded_count": 0, "retained_mean_pnl": 0.0, "excluded_mean_pnl": 0.0}
        time_bucket_perf = []
        session_validation = []
    threshold_usage_summary = []
    if len(all_threshold_usage_rows):
        tu = pd.DataFrame(all_threshold_usage_rows)
        tu = tu.groupby(["sleeve", "session_bucket", "cost_state_key"], as_index=False).agg(
            bars=("bars", "sum"), trade_bars=("trade_bars", "sum")
        )
        tu["trade_freq"] = tu["trade_bars"] / tu["bars"].clip(lower=1)
        threshold_usage_summary = tu.sort_values(["sleeve", "session_bucket", "cost_state_key"]).to_dict(orient="records")
    calib_by_session = []
    if len(all_sleeve_eval):
        sev = pd.DataFrame(all_sleeve_eval)
        req_cols = {"sleeve", "brier_score", "ece", "calibration_slope", "calibration_intercept"}
        if req_cols.issubset(sev.columns):
            calib_by_session = [
                {
                    "session_bucket": str(s),
                    "brier_mean": float(g["brier_score"].mean()),
                    "ece_mean": float(g["ece"].mean()),
                    "cal_slope_mean": float(g["calibration_slope"].mean()),
                    "cal_intercept_mean": float(g["calibration_intercept"].mean()),
                }
                for s, g in sev.groupby("sleeve", sort=True)
            ]

    broker_cfg_summary = BrokerExecutionConfig(**dict(config.get("broker_execution", {})))
    broker_setup_snapshot = build_broker_setup_snapshot(
        broker_cfg_summary,
        ref_price=float(config.get("broker_setup_ref_price", 2000.0)),
        stress_move_pct=float(config.get("broker_setup_stress_move_pct", 0.01)),
    )

    summary = {
        "folds": len(fold_results),
        "fold_results": fold_results,
        "output_dir": str(out_dir),
        "leakage_checks": leakage_checks,
        "broker_setup_snapshot": broker_setup_snapshot,
        "tradability_summary": tradability_summary_global,
        "portfolio_summary": portfolio_summary,
        "failure_decomposition": {
            "ev_by_score_decile_after_cost": ev_deciles,
            "retained_vs_excluded_tradability": retained_vs_excluded,
            "time_to_resolution_bucket_perf": time_bucket_perf,
        },
        "session_validation": {
            "per_session": session_validation,
            "calibration_quality_by_session": calib_by_session,
            "threshold_usage_by_session_cost_bucket": threshold_usage_summary,
        },
        "sleeve_summary": all_sleeve_eval,
        "portfolio_decision_count": int(len(all_portfolio_decisions)),
        "temporal_context_diagnostics": context_diag,
    }
    _save_json(out_dir / "run_summary.json", summary)
    return summary
    broker = dict(config.get("broker_execution", {}))
