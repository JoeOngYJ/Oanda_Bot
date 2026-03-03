#!/usr/bin/env python3
"""Regime -> strategy routing research module (non-leaky HTF->LTF alignment).

This module is intentionally pipeline-adaptive: it expects these existing functions
from a user-provided pipeline module path:
- load_ohlcv(symbol, timeframe) -> DataFrame indexed by timestamp
- make_features(df, timeframe, config) -> DataFrame
- make_labels(df_ltf, atr_series, barrier_cfg) -> labels + trade metadata
- walkforward_train_eval(X, y, model_cfg, splits_cfg) -> metrics + models
- backtest_from_signals(df, signals, cost_cfg) -> trade_log + equity_curve
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


REGIME_NAMES = ["TREND", "RANGE", "TRANSITION", "EXHAUSTION"]
STRATEGY_NAMES = [
    "session_range_breakout",
    "compression_breakout",
    "htf_trend_pullback_continuation",
    "trend_breakout_continuation",
    "liquidity_sweep_reversal",
    "failed_breakout_reversal",
    "london_open_continuation",
    "session_vwap_reversion",
]

STRATEGY_ALLOWED_REGIMES: Dict[str, List[str]] = {
    "session_range_breakout": ["TREND", "RANGE", "TRANSITION"],
    "compression_breakout": ["TRANSITION", "TREND"],
    "htf_trend_pullback_continuation": ["TREND"],
    "trend_breakout_continuation": ["TREND", "TRANSITION"],
    "liquidity_sweep_reversal": ["RANGE", "EXHAUSTION"],
    "failed_breakout_reversal": ["RANGE", "EXHAUSTION", "TRANSITION"],
    "london_open_continuation": ["TREND", "TRANSITION"],
    "session_vwap_reversion": ["RANGE", "EXHAUSTION"],
}

STRATEGY_STYLE: Dict[str, str] = {
    "session_range_breakout": "breakout",
    "compression_breakout": "breakout",
    "htf_trend_pullback_continuation": "trend",
    "trend_breakout_continuation": "trend",
    "liquidity_sweep_reversal": "reversal",
    "failed_breakout_reversal": "reversal",
    "london_open_continuation": "trend",
    "session_vwap_reversion": "reversal",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "symbols": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "GBP_JPY", "XAU_USD"],
    "ltf_timeframe": "M15",
    "entry_timeframes": [],
    "htf_timeframes": {"h1": "H1", "d1": "D1"},
    "timezone": "Europe/London",
    "exploration": {
        "max_variants": 16,
        "top_k": 3,
        "hybrid_min_per_symbol": 1,
        "min_trades": 30,
        "regime_router_min_regime_trades": 20,
        "regime_router_min_mapped_regimes": 2,
        "window_min_trades": 12,
        "min_window_pass_rate": 0.40,
        "window_dd_cap": 0.15,
        "objective": {
            "expectancy_weight": 1.0,
            "max_dd_penalty": 0.50,
            "cost_penalty": 0.20,
            "sharpe_weight": 0.05,
            "window_pass_weight": 0.20,
        },
        "regime_variants": [],
    },
    "training": {
        "years": 10,
        "symbol_batch_size": 1,
    },
    "data_quality": {
        "max_missing_bar_ratio": 0.03,
        "max_time_gap_multiple": 6.0,
        "warn_only": True,
    },
    "regime": {
        "drop_last_htf_bar": True,
        "strict_no_lookahead": True,
        "hysteresis_enabled": True,
        "hysteresis_margin": 0.02,
        "hysteresis_min_confidence": 0.52,
        "min_regime_bars": 1,
        "weights": {"h1": 0.6, "d1": 0.4},
        "thresholds": {
            "slope_lookback": 5,
            "swing_window": 20,
            "swing_break_rate_window": 24,
            "trend_min": 0.55,
            "range_min": 0.55,
            "transition_min": 0.50,
            "exhaustion_min": 0.55,
            "regime_margin": 0.03,
            "transition_contraction_ratio": 0.85,
            "transition_expansion_ratio": 1.10,
            "exhaustion_extension_atr": 2.0,
            "impulse_body_atr": 1.25,
        },
    },
    "sessions": {
        "tokyo": [0, 9],
        "london": [8, 17],
        "newyork": [13, 22],
        "overlap": [13, 17],
    },
    "pair_overrides": {
        "EUR_USD": {
            "preferred_sessions": ["london", "overlap"],
            "preferred_styles": ["breakout", "trend"],
            "entry_timeframes": ["M5"],
            "barrier": {"horizon_bars": 12, "up_atr_mult": 0.75, "down_atr_mult": 0.75, "three_class_labels": True},
            "hard_filters": {"expectancy_min": 0.0, "net_expectancy_min": 0.0, "window_pass_min": 0.5},
        },
        "GBP_USD": {
            "preferred_sessions": ["london", "overlap"],
            "preferred_styles": ["breakout", "reversal"],
            "entry_timeframes": ["M5"],
            "barrier": {"horizon_bars": 14, "up_atr_mult": 0.8, "down_atr_mult": 0.8, "three_class_labels": True},
            "hard_filters": {"expectancy_min": 0.0, "net_expectancy_min": 0.0, "window_pass_min": 0.5},
        },
        "USD_JPY": {
            "preferred_sessions": ["tokyo", "overlap"],
            "preferred_styles": ["breakout"],
            "entry_timeframes": ["M15"],
            "barrier": {"horizon_bars": 10, "up_atr_mult": 0.7, "down_atr_mult": 0.7, "three_class_labels": True},
            "hard_filters": {"expectancy_min": 0.0, "net_expectancy_min": 0.0, "window_pass_min": 0.5},
        },
        "USD_CAD": {
            "preferred_sessions": ["newyork", "overlap"],
            "preferred_styles": ["breakout", "trend"],
            "entry_timeframes": ["M5"],
            "barrier": {"horizon_bars": 12, "up_atr_mult": 0.8, "down_atr_mult": 0.8, "three_class_labels": True},
            "hard_filters": {"expectancy_min": 0.0, "net_expectancy_min": 0.0, "window_pass_min": 0.5},
        },
        "GBP_JPY": {
            "preferred_sessions": ["london", "overlap", "tokyo"],
            "preferred_styles": ["breakout", "trend"],
            "entry_timeframes": ["M15"],
            "barrier": {"horizon_bars": 12, "up_atr_mult": 0.85, "down_atr_mult": 0.85, "three_class_labels": True},
            "hard_filters": {"expectancy_min": 0.0, "net_expectancy_min": 0.0, "window_pass_min": 0.5},
            "stricter_filters": True,
        },
        "XAU_USD": {
            "preferred_sessions": ["london", "overlap"],
            "preferred_styles": ["breakout", "reversal"],
            "entry_timeframes": ["M15"],
            "barrier": {
                "horizon_bars": 12,
                "up_atr_mult": 0.95,
                "down_atr_mult": 0.95,
                "three_class_labels": True,
                "neutral_atr_mult": 0.35,
            },
            "signal_quality": {
                "allowed_sessions": ["london", "overlap"],
                "min_regime_confidence": 0.60,
                "min_regime_score_margin": 0.05,
                "max_atr_norm": 2.0,
                "max_body_atr": 1.8,
            },
            "hard_filters": {"expectancy_min": 0.0002, "net_expectancy_min": 0.0001, "window_pass_min": 0.55},
            "entry_refinement": {
                "enabled": True,
                "apply_timeframes": ["M15", "M5"],
                "min_regime_confidence": 0.60,
                "min_regime_score_margin": 0.04,
                "confirmation_bars": 2,
                "min_body_atr": 0.10,
                "max_atr_norm": 2.0,
            },
            "regime": {
                "hysteresis_enabled": True,
                "hysteresis_margin": 0.04,
                "hysteresis_min_confidence": 0.60,
                "min_regime_bars": 4,
            },
        },
    },
    "no_trade": {
        "enabled": True,
        "probability_threshold": 0.58,
        "calibrate_threshold": True,
        "threshold_grid": [0.52, 0.55, 0.58, 0.62, 0.66],
        "objective": {
            "expectancy_weight": 1.0,
            "max_dd_penalty": 0.50,
            "cost_penalty": 0.20,
            "sharpe_weight": 0.05,
            "window_pass_weight": 0.20,
        },
        "min_trades": 30,
        "max_dd_cap": 0.20,
        "ece_bins": 10,
    },
    "entry_refinement": {
        "enabled": True,
        "apply_timeframes": ["M5", "M1"],
        "min_regime_confidence": 0.56,
        "min_regime_score_margin": 0.03,
        "confirmation_bars": 2,
        "min_body_atr": 0.08,
        "max_atr_norm": 2.5,
    },
    "risk_controls": {
        "kill_switch_expectancy_max": 0.0,
        "kill_switch_sharpe_max": 0.0,
        "kill_switch_window_pass_max": 0.35,
    },
    "deployment_policy": {
        "enabled": True,
        "champion_count": 1,
        "challenger_count": 1,
        "rolling_monitoring_windows": 4,
        "disable_on_negative_windows": 3,
        "disable_on_dd_breach_pct": 18.0,
    },
    "model": {},
    "splits": {},
    "barrier": {},
    "cost": {},
    "output": {
        "ranking_csv": "data/research/regime_strategy_ranking.csv",
        "model_dir": "models/research/regime_strategy",
        "manifest_json": "data/research/regime_strategy_manifest.json",
    },
}


@dataclass
class PipelineFns:
    load_ohlcv: Callable[[str, str], pd.DataFrame]
    make_features: Callable[[pd.DataFrame, str, Dict[str, Any]], pd.DataFrame]
    make_labels: Callable[[pd.DataFrame, pd.Series, Dict[str, Any]], Any]
    walkforward_train_eval: Callable[[pd.DataFrame, pd.Series, Dict[str, Any], Dict[str, Any]], Any]
    backtest_from_signals: Callable[[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Any]
    select_and_transform_features: Optional[Callable[[pd.DataFrame, List[str]], Tuple[pd.DataFrame, Dict[str, Any]]]] = None
    generate_purged_walkforward_splits: Optional[
        Callable[[pd.Index, pd.Series, int, int, int], List[Tuple[np.ndarray, np.ndarray]]]
    ] = None
    fit_probability_calibrator: Optional[Callable[..., Any]] = None
    apply_probability_calibrator: Optional[Callable[..., Any]] = None
    compute_fold_diagnostics: Optional[Callable[..., Dict[str, Any]]] = None
    compute_expected_value: Optional[Callable[[pd.Series, pd.Series, pd.Series, pd.Series], pd.Series]] = None
    apply_trade_gating: Optional[
        Callable[[pd.DataFrame, str, str, float, float, Optional[str]], pd.Series]
    ] = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not available but YAML config requested.")
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    return _deep_merge(DEFAULT_CONFIG, cfg)


def _load_pipeline(module_path: str) -> PipelineFns:
    mod = importlib.import_module(module_path)
    required = [
        "load_ohlcv",
        "make_features",
        "make_labels",
        "walkforward_train_eval",
        "backtest_from_signals",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"Pipeline module missing required functions: {missing}")
    return PipelineFns(
        load_ohlcv=getattr(mod, "load_ohlcv"),
        make_features=getattr(mod, "make_features"),
        make_labels=getattr(mod, "make_labels"),
        walkforward_train_eval=getattr(mod, "walkforward_train_eval"),
        backtest_from_signals=getattr(mod, "backtest_from_signals"),
        select_and_transform_features=getattr(mod, "select_and_transform_features", None),
        generate_purged_walkforward_splits=getattr(mod, "generate_purged_walkforward_splits", None),
        fit_probability_calibrator=getattr(mod, "fit_probability_calibrator", None),
        apply_probability_calibrator=getattr(mod, "apply_probability_calibrator", None),
        compute_fold_diagnostics=getattr(mod, "compute_fold_diagnostics", None),
        compute_expected_value=getattr(mod, "compute_expected_value", None),
        apply_trade_gating=getattr(mod, "apply_trade_gating", None),
    )


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out.sort_index()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _clip01(x: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(x)
    return s.clip(lower=0.0, upper=1.0)


def _norm_ratio(x: pd.Series, scale: float = 1.0) -> pd.Series:
    z = x / max(scale, 1e-9)
    return _clip01(0.5 + 0.5 * np.tanh(z))


def _timeframe_seconds(tf: str) -> int:
    tf = tf.upper()
    if tf.startswith("M") and tf[1:].isdigit():
        return int(tf[1:]) * 60
    if tf.startswith("H") and tf[1:].isdigit():
        return int(tf[1:]) * 3600
    if tf == "D1" or tf == "D":
        return 86400
    return 3600


def _session_flags(index_utc: pd.DatetimeIndex, tz_name: str, sessions: Dict[str, List[int]]) -> pd.DataFrame:
    local_idx = index_utc.tz_convert(tz_name)
    h = local_idx.hour
    out = pd.DataFrame(index=index_utc)

    def _in_window(hours: np.ndarray, start_end: Iterable[int]) -> np.ndarray:
        s, e = [int(v) for v in start_end]
        if s <= e:
            return (hours >= s) & (hours < e)
        return (hours >= s) | (hours < e)

    out["sess_tokyo"] = _in_window(h, sessions.get("tokyo", [0, 9])).astype(float)
    out["sess_london"] = _in_window(h, sessions.get("london", [8, 17])).astype(float)
    out["sess_newyork"] = _in_window(h, sessions.get("newyork", [13, 22])).astype(float)
    out["sess_overlap"] = _in_window(h, sessions.get("overlap", [13, 17])).astype(float)
    return out


def _compute_htf_scores(df: pd.DataFrame, th: Dict[str, Any]) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    close = x["close"].astype(float)
    open_ = x["open"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    atr14 = _atr(x, 14).replace(0, np.nan)

    slope_lb = int(th.get("slope_lookback", 5))
    swing_win = int(th.get("swing_window", 20))
    break_win = int(th.get("swing_break_rate_window", 24))

    ema50_slope = (ema50 - ema50.shift(slope_lb)) / max(slope_lb, 1)
    slope_atr = (ema50_slope.abs() / (atr14 + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rolling_high = high.rolling(swing_win, min_periods=swing_win).max().shift(1)
    rolling_low = low.rolling(swing_win, min_periods=swing_win).min().shift(1)
    break_up = (close > rolling_high).astype(float)
    break_dn = (close < rolling_low).astype(float)
    swing_break_rate = (break_up + break_dn).rolling(break_win, min_periods=max(5, break_win // 3)).mean().fillna(0.0)

    price_side = np.sign(close - ema200).replace(0, 1.0)
    side_strength = (close - ema200).abs() / (atr14 + 1e-9)

    trend_strength = _clip01(
        0.45 * _norm_ratio(slope_atr, scale=1.0)
        + 0.25 * _norm_ratio(side_strength, scale=2.0)
        + 0.30 * swing_break_rate
    )

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    bb_up = bb_mid + (2.0 * bb_std)
    bb_dn = bb_mid - (2.0 * bb_std)

    dist_mid = close - bb_mid
    cross = (np.sign(dist_mid) != np.sign(dist_mid.shift(1))).astype(float)
    cross_freq = cross.rolling(20, min_periods=10).mean().fillna(0.0)

    flat_slope = 1.0 - _norm_ratio(slope_atr, scale=1.0)
    bounded_hl = 1.0 - swing_break_rate

    range_score = _clip01(0.40 * cross_freq + 0.35 * flat_slope + 0.25 * bounded_hl)

    atr_short = atr14.rolling(5, min_periods=5).mean()
    atr_long = atr14.rolling(30, min_periods=15).mean()
    contraction = (atr_short / (atr_long + 1e-9))
    contract_flag = (contraction < float(th.get("transition_contraction_ratio", 0.85))).astype(float)
    expansion_strength = _norm_ratio((contraction - float(th.get("transition_expansion_ratio", 1.10))) * 3.0, scale=1.0)
    recent_contract = contract_flag.rolling(8, min_periods=1).max().shift(1).fillna(0.0)
    transition_score = _clip01(0.55 * recent_contract * expansion_strength + 0.45 * (1.0 - range_score))

    ext20 = (close - ema20).abs() / (atr14 + 1e-9)
    ext50 = (close - ema50).abs() / (atr14 + 1e-9)
    body = (close - open_).abs()
    impulse = (body / (atr14 + 1e-9) > float(th.get("impulse_body_atr", 1.25))).astype(float)
    impulse_freq = impulse.rolling(10, min_periods=4).mean().fillna(0.0)

    exhaustion_score = _clip01(
        0.55
        * _norm_ratio((0.5 * ext20 + 0.5 * ext50) - float(th.get("exhaustion_extension_atr", 2.0)), scale=1.0)
        + 0.45 * impulse_freq
    )

    out = pd.DataFrame(
        {
            "trend_strength": trend_strength,
            "range_score": range_score,
            "transition_score": transition_score,
            "exhaustion_score": exhaustion_score,
            "price_side": price_side.astype(float),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "atr14": atr14,
            "bb_up": bb_up,
            "bb_dn": bb_dn,
            "rolling_high": rolling_high,
            "rolling_low": rolling_low,
        },
        index=x.index,
    )
    return out


def _classify_regime(scores: pd.DataFrame, th: Dict[str, Any]) -> pd.DataFrame:
    x = scores.copy()
    cols = ["trend_strength", "range_score", "transition_score", "exhaustion_score"]
    arr = x[cols].to_numpy(dtype=float)
    order = np.argsort(arr, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2]

    best = arr[np.arange(len(arr)), best_idx]
    second = arr[np.arange(len(arr)), second_idx]
    margin = float(th.get("regime_margin", 0.03))

    min_map = {
        0: float(th.get("trend_min", 0.55)),
        1: float(th.get("range_min", 0.55)),
        2: float(th.get("transition_min", 0.50)),
        3: float(th.get("exhaustion_min", 0.55)),
    }

    regime_idx = best_idx.copy()
    low_conf = np.array([best[i] < min_map[int(best_idx[i])] for i in range(len(best_idx))], dtype=bool)
    small_margin = (best - second) < margin
    # Prefer TRANSITION when confidence is weak / ambiguous.
    regime_idx[low_conf | small_margin] = 2

    regime = pd.Series([REGIME_NAMES[int(i)] for i in regime_idx], index=x.index, dtype="object")
    conf = pd.Series(best, index=x.index).clip(lower=0.0, upper=1.0)

    out = x.copy()
    out["regime"] = regime
    out["regime_confidence"] = conf
    out["trend_bias"] = np.sign(x["price_side"]).replace(0, 1.0)
    return out


def _asof_merge_ltf(ltf_index: pd.DatetimeIndex, htf_df: pd.DataFrame, prefix: str, allow_exact: bool) -> pd.DataFrame:
    left = pd.DataFrame({"timestamp": ltf_index}).sort_values("timestamp")
    right = htf_df.copy().reset_index().rename(columns={htf_df.index.name or "index": "timestamp"})
    right = right.sort_values("timestamp")
    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        allow_exact_matches=allow_exact,
    )
    merged = merged.set_index("timestamp").reindex(ltf_index)
    merged = merged.add_prefix(f"{prefix}_")
    return merged


def _compute_regime_features(
    df_ltf: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_d1: pd.DataFrame,
    regime_cfg: Dict[str, Any],
    variant: Dict[str, Any],
) -> pd.DataFrame:
    th = _deep_merge(regime_cfg.get("thresholds", {}), variant)

    h1 = _ensure_utc_index(df_h1)
    d1 = _ensure_utc_index(df_d1)
    if bool(regime_cfg.get("drop_last_htf_bar", True)):
        if len(h1) > 0:
            h1 = h1.iloc[:-1]
        if len(d1) > 0:
            d1 = d1.iloc[:-1]

    h1_scores = _classify_regime(_compute_htf_scores(h1, th), th)
    d1_scores = _classify_regime(_compute_htf_scores(d1, th), th)

    allow_exact = not bool(regime_cfg.get("strict_no_lookahead", True))
    ltf_idx = _ensure_utc_index(df_ltf).index

    h1_m = _asof_merge_ltf(ltf_idx, h1_scores, prefix="h1", allow_exact=allow_exact)
    d1_m = _asof_merge_ltf(ltf_idx, d1_scores, prefix="d1", allow_exact=allow_exact)

    out = pd.concat([h1_m, d1_m], axis=1)
    w_h1 = float(regime_cfg.get("weights", {}).get("h1", 0.6))
    w_d1 = float(regime_cfg.get("weights", {}).get("d1", 0.4))

    out["regime_score_trend"] = (w_h1 * out["h1_trend_strength"].fillna(0.0)) + (w_d1 * out["d1_trend_strength"].fillna(0.0))
    out["regime_score_range"] = (w_h1 * out["h1_range_score"].fillna(0.0)) + (w_d1 * out["d1_range_score"].fillna(0.0))
    out["regime_score_transition"] = (w_h1 * out["h1_transition_score"].fillna(0.0)) + (
        w_d1 * out["d1_transition_score"].fillna(0.0)
    )
    out["regime_score_exhaustion"] = (w_h1 * out["h1_exhaustion_score"].fillna(0.0)) + (
        w_d1 * out["d1_exhaustion_score"].fillna(0.0)
    )
    out["regime_trend_bias"] = np.sign((w_h1 * out["h1_trend_bias"].fillna(0.0)) + (w_d1 * out["d1_trend_bias"].fillna(0.0)))

    score_cols = [
        "regime_score_trend",
        "regime_score_range",
        "regime_score_transition",
        "regime_score_exhaustion",
    ]
    score_arr = out[score_cols].to_numpy(dtype=float)
    regime_idx = np.argmax(score_arr, axis=1)
    regime_conf = np.max(score_arr, axis=1)
    out["regime"] = [REGIME_NAMES[int(i)] for i in regime_idx]
    out["regime_confidence"] = regime_conf
    sorted_scores = np.sort(score_arr, axis=1)
    out["regime_score_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    if bool(regime_cfg.get("hysteresis_enabled", True)):
        h_margin = float(regime_cfg.get("hysteresis_margin", 0.02))
        h_min_conf = float(regime_cfg.get("hysteresis_min_confidence", 0.52))
        rg = out["regime"].astype("object").to_numpy(copy=True)
        mg = out["regime_score_margin"].to_numpy(dtype=float)
        cf = out["regime_confidence"].to_numpy(dtype=float)
        for i in range(1, len(rg)):
            if (mg[i] < h_margin) and (cf[i - 1] >= h_min_conf):
                rg[i] = rg[i - 1]
        out["regime"] = rg
    min_regime_bars = max(1, int(regime_cfg.get("min_regime_bars", 1)))
    if min_regime_bars > 1 and len(out) > 1:
        rg = out["regime"].astype("object").to_numpy(copy=True)
        last = rg[0]
        run_len = 1
        for i in range(1, len(rg)):
            if rg[i] == last:
                run_len += 1
                continue
            if run_len < min_regime_bars:
                rg[i] = last
                run_len += 1
                continue
            last = rg[i]
            run_len = 1
        out["regime"] = rg
    out["regime_variant_id"] = str(variant.get("id", "v0"))
    return out


def _candles(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        df["open"].astype(float),
        df["high"].astype(float),
        df["low"].astype(float),
        df["close"].astype(float),
    )


def _gen_session_range_breakout(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    o, h, l, c = _candles(df)
    lookback_hours = int(cfg.get("lookback_hours", 6))
    sec = int(cfg.get("ltf_seconds", _timeframe_seconds("M15")))
    bars = max(2, int((lookback_hours * 3600) / max(sec, 1)))

    prev_high = h.rolling(bars, min_periods=bars).max().shift(1)
    prev_low = l.rolling(bars, min_periods=bars).min().shift(1)

    long_sig = (c > prev_high) & ((df.get("sess_london", 0) > 0) | (df.get("sess_overlap", 0) > 0) | (df.get("sess_tokyo", 0) > 0))
    short_sig = (c < prev_low) & ((df.get("sess_london", 0) > 0) | (df.get("sess_overlap", 0) > 0) | (df.get("sess_tokyo", 0) > 0))

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_range_break_dist"] = np.where(out["direction"] > 0, (c - prev_high), np.where(out["direction"] < 0, (prev_low - c), 0.0))
    out["sig_strategy"] = "session_range_breakout"
    return out[out["direction"] != 0]


def _gen_compression_breakout(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    _, h, l, c = _candles(df)
    atr_fast = _atr(df, 14)
    atr_slow = _atr(df, 50)
    ma = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    bbw = (4.0 * sd) / (ma.abs() + 1e-9)

    squeeze = (atr_fast / (atr_slow + 1e-9) < float(cfg.get("squeeze_atr_ratio", 0.8))) & (
        bbw < float(cfg.get("squeeze_bbw", 0.02))
    )
    recent_squeeze = squeeze.rolling(6, min_periods=1).max().shift(1).fillna(0).astype(bool)

    n = int(cfg.get("break_lookback", 20))
    high_n = h.rolling(n, min_periods=n).max().shift(1)
    low_n = l.rolling(n, min_periods=n).min().shift(1)
    long_sig = recent_squeeze & (c > high_n)
    short_sig = recent_squeeze & (c < low_n)

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_compression_ratio"] = (atr_fast / (atr_slow + 1e-9)).fillna(1.0)
    out["sig_strategy"] = "compression_breakout"
    return out[out["direction"] != 0]


def _gen_htf_trend_pullback_continuation(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    _, _, _, c = _candles(df)
    ema20 = _ema(c, 20)
    bias = np.sign(df.get("regime_trend_bias", pd.Series(index=df.index, data=0.0))).replace(0, 1)

    long_pullback = (bias > 0) & (c.shift(1) < ema20.shift(1)) & (c > ema20)
    short_pullback = (bias < 0) & (c.shift(1) > ema20.shift(1)) & (c < ema20)

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_pullback, 1, np.where(short_pullback, -1, 0))
    out["sig_pullback_dist"] = ((c - ema20).abs()).fillna(0.0)
    out["sig_strategy"] = "htf_trend_pullback_continuation"
    return out[out["direction"] != 0]


def _gen_trend_breakout_continuation(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    _, h, l, c = _candles(df)
    n = int(cfg.get("break_lookback", 30))
    high_n = h.rolling(n, min_periods=n).max().shift(1)
    low_n = l.rolling(n, min_periods=n).min().shift(1)
    bias = np.sign(df.get("regime_trend_bias", pd.Series(index=df.index, data=0.0))).replace(0, 1)

    long_sig = (bias > 0) & (c > high_n)
    short_sig = (bias < 0) & (c < low_n)

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_breakout_dist"] = np.where(out["direction"] > 0, c - high_n, np.where(out["direction"] < 0, low_n - c, 0.0))
    out["sig_strategy"] = "trend_breakout_continuation"
    return out[out["direction"] != 0]


def _gen_liquidity_sweep_reversal(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    o, h, l, c = _candles(df)
    n = int(cfg.get("sweep_lookback", 20))
    low_n = l.rolling(n, min_periods=n).min().shift(1)
    high_n = h.rolling(n, min_periods=n).max().shift(1)
    atr14 = _atr(df, 14)

    lower_wick = (np.minimum(o, c) - l).clip(lower=0)
    upper_wick = (h - np.maximum(o, c)).clip(lower=0)

    long_sig = (l < low_n) & (c > low_n) & (lower_wick / (atr14 + 1e-9) > float(cfg.get("min_wick_atr", 0.25)))
    short_sig = (h > high_n) & (c < high_n) & (upper_wick / (atr14 + 1e-9) > float(cfg.get("min_wick_atr", 0.25)))

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_wick_atr"] = np.where(out["direction"] > 0, lower_wick / (atr14 + 1e-9), np.where(out["direction"] < 0, upper_wick / (atr14 + 1e-9), 0.0))
    out["sig_strategy"] = "liquidity_sweep_reversal"
    return out[out["direction"] != 0]


def _gen_failed_breakout_reversal(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    _, h, l, c = _candles(df)
    n = int(cfg.get("break_lookback", 20))
    high_n = h.rolling(n, min_periods=n).max().shift(1)
    low_n = l.rolling(n, min_periods=n).min().shift(1)

    prev_break_up = c.shift(1) > high_n.shift(1)
    prev_break_dn = c.shift(1) < low_n.shift(1)

    short_sig = prev_break_up & (c < high_n)
    long_sig = prev_break_dn & (c > low_n)

    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_fail_break_dist"] = np.where(out["direction"] > 0, c - low_n, np.where(out["direction"] < 0, high_n - c, 0.0))
    out["sig_strategy"] = "failed_breakout_reversal"
    return out[out["direction"] != 0]


def _gen_london_open_continuation(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    o, h, l, c = _candles(df)
    open_hours = int(cfg.get("open_window_hours", 2))
    sec = int(cfg.get("ltf_seconds", _timeframe_seconds("M15")))
    n = max(2, int((open_hours * 3600) / max(sec, 1)))
    london = pd.to_numeric(df.get("sess_london", pd.Series(index=df.index, data=0.0)), errors="coerce").fillna(0.0)
    atr14 = _atr(df, 14)
    rng = (h - l).rolling(n, min_periods=n).max().shift(1)
    long_sig = (london > 0) & (c > o.rolling(n, min_periods=n).mean()) & (rng / (atr14 + 1e-9) > 0.8)
    short_sig = (london > 0) & (c < o.rolling(n, min_periods=n).mean()) & (rng / (atr14 + 1e-9) > 0.8)
    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_london_open_range_atr"] = (rng / (atr14 + 1e-9)).fillna(0.0)
    out["sig_strategy"] = "london_open_continuation"
    return out[out["direction"] != 0]


def _gen_session_vwap_reversion(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    o, h, l, c = _candles(df)
    tp = (h + l + c) / 3.0
    vol = pd.to_numeric(df.get("volume", pd.Series(index=df.index, data=1.0)), errors="coerce").fillna(1.0)
    day_key = pd.DatetimeIndex(df.index).tz_convert("UTC").normalize()
    num = (tp * vol).groupby(day_key).cumsum()
    den = vol.groupby(day_key).cumsum().replace(0, np.nan)
    vwap = (num / den).replace([np.inf, -np.inf], np.nan).ffill()
    atr14 = _atr(df, 14)
    dist = (c - vwap) / (atr14 + 1e-9)
    z = float(cfg.get("vwap_dist_atr", 1.2))
    overlap = pd.to_numeric(df.get("sess_overlap", pd.Series(index=df.index, data=0.0)), errors="coerce").fillna(0.0)
    long_sig = (dist < -z) & (overlap > 0)
    short_sig = (dist > z) & (overlap > 0)
    out = pd.DataFrame(index=df.index)
    out["direction"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    out["sig_vwap_dist_atr"] = dist.fillna(0.0)
    out["sig_strategy"] = "session_vwap_reversion"
    return out[out["direction"] != 0]


def _generate_strategy_signals(df: pd.DataFrame, strategy: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    if strategy == "session_range_breakout":
        return _gen_session_range_breakout(df, cfg)
    if strategy == "compression_breakout":
        return _gen_compression_breakout(df, cfg)
    if strategy == "htf_trend_pullback_continuation":
        return _gen_htf_trend_pullback_continuation(df, cfg)
    if strategy == "trend_breakout_continuation":
        return _gen_trend_breakout_continuation(df, cfg)
    if strategy == "liquidity_sweep_reversal":
        return _gen_liquidity_sweep_reversal(df, cfg)
    if strategy == "failed_breakout_reversal":
        return _gen_failed_breakout_reversal(df, cfg)
    if strategy == "london_open_continuation":
        return _gen_london_open_continuation(df, cfg)
    if strategy == "session_vwap_reversion":
        return _gen_session_vwap_reversion(df, cfg)
    raise ValueError(f"Unknown strategy: {strategy}")


def _parse_labels(raw: Any, index: pd.DatetimeIndex) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    y: Optional[pd.Series] = None

    if isinstance(raw, tuple) and len(raw) >= 1:
        y_raw = raw[0]
        if len(raw) > 1 and isinstance(raw[1], dict):
            meta = raw[1]
    else:
        y_raw = raw

    if isinstance(y_raw, pd.Series):
        y = y_raw.reindex(index)
    elif isinstance(y_raw, pd.DataFrame):
        if "label" in y_raw.columns:
            y = y_raw["label"].reindex(index)
        else:
            y = y_raw.iloc[:, 0].reindex(index)
    else:
        arr = np.asarray(y_raw)
        if len(arr) == len(index):
            y = pd.Series(arr, index=index)

    if y is None:
        raise ValueError("Unable to parse labels from make_labels output.")
    return y, meta


def _extract_wf_output(raw: Any) -> Tuple[Dict[str, Any], Any]:
    metrics: Dict[str, Any] = {}
    models: Any = None
    if isinstance(raw, tuple) and len(raw) >= 2:
        metrics = raw[0] if isinstance(raw[0], dict) else {"raw_metrics": raw[0]}
        models = raw[1]
    elif isinstance(raw, dict):
        metrics = raw
        models = raw.get("models")
    else:
        metrics = {"raw_metrics": raw}
    return metrics, models


def _extract_probability_series(metrics: Dict[str, Any], index: pd.DatetimeIndex) -> Optional[pd.Series]:
    for key in ["oof_proba", "pred_proba", "proba", "probability"]:
        if key not in metrics:
            continue
        v = metrics[key]
        if isinstance(v, pd.Series):
            return v.reindex(index)
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, 1]
        if arr.ndim == 1 and len(arr) == len(index):
            return pd.Series(arr, index=index)
    return None


def _estimate_label_end_ts(
    y_index: pd.DatetimeIndex,
    full_index: pd.DatetimeIndex,
    horizon_bars: int,
) -> pd.Series:
    full = pd.DatetimeIndex(full_index)
    pos = pd.Series(np.arange(len(full), dtype=int), index=full)
    yp = pos.reindex(y_index).fillna(-1).astype(int)
    hb = max(1, int(horizon_bars))
    end_pos = np.clip(yp.to_numpy(dtype=int) + hb, 0, max(0, len(full) - 1))
    end_ts = pd.Series(full[end_pos], index=y_index, dtype="datetime64[ns, UTC]")
    return end_ts


def _prepare_training_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    fns: PipelineFns,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    Xi = X.copy()
    yi = y.copy()
    prep_meta: Dict[str, Any] = {}
    if callable(fns.select_and_transform_features):
        try:
            Xi_t, stats = fns.select_and_transform_features(Xi, list(Xi.columns))
            Xi = Xi_t
            prep_meta["feature_transform_stats"] = stats
        except Exception as exc:
            prep_meta["feature_transform_error"] = str(exc)

    for col in list(Xi.columns):
        if not pd.api.types.is_numeric_dtype(Xi[col]):
            codes, _ = pd.factorize(Xi[col], sort=True)
            Xi[col] = pd.Series(codes, index=Xi.index, dtype=float)
    Xi = Xi.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    yi = yi.reindex(Xi.index)
    return Xi, yi, prep_meta


def _build_custom_purged_splits(
    X_index: pd.DatetimeIndex,
    label_end_ts: pd.Series,
    splits_cfg: Dict[str, Any],
    fns: PipelineFns,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not callable(fns.generate_purged_walkforward_splits):
        return []
    n_splits = int(splits_cfg.get("folds", 5))
    n = len(X_index)
    if n < 40:
        return []
    test_size = int(splits_cfg.get("test_size", max(10, n // max(4, n_splits + 1))))
    embargo = int(splits_cfg.get("embargo_bars", max(1, test_size // 10)))
    try:
        return fns.generate_purged_walkforward_splits(X_index, label_end_ts.reindex(X_index), n_splits, test_size, embargo)
    except Exception:
        return []


def _to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if obj is None:
        return pd.DataFrame()
    return pd.DataFrame(obj)


def _compute_max_drawdown(equity_curve: pd.Series | pd.DataFrame) -> float:
    if isinstance(equity_curve, pd.DataFrame):
        if "equity" in equity_curve.columns:
            e = equity_curve["equity"].astype(float)
        else:
            e = equity_curve.iloc[:, 0].astype(float)
    else:
        e = pd.Series(equity_curve, dtype=float)
    if e.empty:
        return 0.0
    peak = e.cummax()
    dd = (peak - e) / peak.replace(0, np.nan)
    return float(dd.fillna(0.0).max())


def _trade_metrics(
    trade_log: pd.DataFrame,
    equity_curve: Any,
    ltf_index: pd.DatetimeIndex,
    settings: Optional[Dict[str, Any]] = None,
    signals: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    settings = settings or {}
    window_min_trades = max(5, int(settings.get("window_min_trades", 12)))
    window_dd_cap = float(settings.get("window_dd_cap", 0.15))
    trades = _to_dataframe(trade_log)
    if trades.empty:
        return {
            "expectancy_r": 0.0,
            "net_expectancy_after_cost": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "sharpe_trade": 0.0,
            "max_dd": _compute_max_drawdown(pd.Series(index=ltf_index, data=1.0)),
            "trades": 0,
            "trades_per_month": 0.0,
            "trades_per_day": 0.0,
            "cost_per_trade": 0.0,
            "financing_per_trade": 0.0,
            "cost_to_gross_pnl_ratio": 0.0,
            "avg_hold_hours": 0.0,
            "overnight_trade_pct": 0.0,
            "signal_flip_rate": 0.0,
            "long_expectancy_r": 0.0,
            "short_expectancy_r": 0.0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "rolling_3m_expectancy": 0.0,
            "rolling_3m_sharpe": 0.0,
            "worst_3m_drawdown": 0.0,
            "window_pass_rate": 0.0,
        }

    if "r_multiple" in trades.columns:
        rvals = pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0.0)
    elif "pnl" in trades.columns:
        rvals = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    else:
        rvals = pd.Series(np.zeros(len(trades)), index=trades.index)

    rvals = pd.to_numeric(rvals, errors="coerce").fillna(0.0)
    wins = int((rvals > 0).sum())
    losses = int((rvals < 0).sum())
    gross_profit = float(rvals[rvals > 0].sum())
    gross_loss = float(-rvals[rvals < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    if "entry_time" in trades.columns:
        t_idx = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    elif "timestamp" in trades.columns:
        t_idx = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    else:
        t_idx = pd.to_datetime(trades.index, utc=True, errors="coerce")

    t_ser = pd.Series(t_idx, index=trades.index)
    valid_t = t_ser.notna()
    t_clean = pd.DatetimeIndex(t_ser.loc[valid_t])
    rvals_time = rvals.loc[valid_t] if len(rvals) == len(valid_t) else rvals.iloc[: len(t_clean)]

    t_period = t_clean.tz_localize(None).to_period("M")
    months = max(1, int(t_period.nunique()))
    trades_per_month = float(len(trades) / months)
    trade_days = max(1, int(t_clean.normalize().nunique())) if len(t_clean) else 1
    trades_per_day = float(len(trades) / trade_days)

    if len(rvals) > 1 and float(rvals.std(ddof=0)) > 1e-12:
        sharpe_trade = float((rvals.mean() / rvals.std(ddof=0)) * np.sqrt(min(len(rvals), 252)))
    else:
        sharpe_trade = 0.0

    fee_cols = [c for c in trades.columns if any(k in str(c).lower() for k in ["fee", "commission", "spread_cost", "slippage_cost"])]
    financing_cols = [c for c in trades.columns if any(k in str(c).lower() for k in ["financing", "swap", "rollover"])]
    net_pnl_col = next((c for c in ["net_pnl", "pnl", "profit"] if c in trades.columns), None)
    gross_pnl_col = next((c for c in ["gross_pnl", "pnl_gross", "pnl_before_cost", "raw_pnl"] if c in trades.columns), None)

    fee_total = float(trades[fee_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum().sum()) if fee_cols else 0.0
    financing_total = (
        float(trades[financing_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum().sum()) if financing_cols else 0.0
    )
    total_cost = float(fee_total + financing_total)
    cost_per_trade = float(total_cost / max(1, len(trades)))
    financing_per_trade = float(financing_total / max(1, len(trades)))

    if gross_pnl_col:
        gross_pnl = float(pd.to_numeric(trades[gross_pnl_col], errors="coerce").fillna(0.0).sum())
    elif net_pnl_col:
        net_tmp = pd.to_numeric(trades[net_pnl_col], errors="coerce").fillna(0.0)
        gross_pnl = float(net_tmp.sum() + total_cost)
    else:
        gross_pnl = float(rvals.sum() + total_cost)

    if net_pnl_col:
        net_pnl_series = pd.to_numeric(trades[net_pnl_col], errors="coerce").fillna(0.0)
        net_expectancy = float(net_pnl_series.mean())
    else:
        net_expectancy = float(rvals.mean() - cost_per_trade)

    cost_to_gross_ratio = float(total_cost / (abs(gross_pnl) + 1e-9))

    direction_col = next((c for c in ["direction", "side", "signal"] if c in trades.columns), None)
    if direction_col:
        d = pd.to_numeric(trades[direction_col], errors="coerce").fillna(0.0)
        if direction_col == "side":
            d = d.map({"long": 1, "short": -1, "buy": 1, "sell": -1}).fillna(d)
    else:
        d = pd.Series(np.zeros(len(trades)), index=trades.index)

    long_mask = d > 0
    short_mask = d < 0
    long_r = rvals.loc[long_mask] if long_mask.any() else pd.Series(dtype=float)
    short_r = rvals.loc[short_mask] if short_mask.any() else pd.Series(dtype=float)

    entry_col = next((c for c in ["entry_time", "open_time"] if c in trades.columns), None)
    exit_col = next((c for c in ["exit_time", "close_time"] if c in trades.columns), None)
    avg_hold_hours = 0.0
    overnight_trade_pct = 0.0
    if entry_col and exit_col:
        ent = pd.to_datetime(trades[entry_col], utc=True, errors="coerce")
        exi = pd.to_datetime(trades[exit_col], utc=True, errors="coerce")
        hold = (exi - ent).dt.total_seconds() / 3600.0
        hold = pd.to_numeric(hold, errors="coerce").dropna()
        if not hold.empty:
            avg_hold_hours = float(hold.mean())
        ovn = ((exi.dt.normalize() > ent.dt.normalize()) | (hold >= 24.0)).fillna(False)
        overnight_trade_pct = float(ovn.mean()) if len(ovn) else 0.0

    signal_flip_rate = 0.0
    if signals is not None and not signals.empty and "direction" in signals.columns:
        sig_d = pd.to_numeric(signals["direction"], errors="coerce").fillna(0.0)
        sig_d = sig_d[sig_d != 0.0]
        if len(sig_d) > 1:
            flips = (sig_d * sig_d.shift(1) < 0).astype(float)
            signal_flip_rate = float(flips.mean())

    rolling_3m_expectancy = 0.0
    rolling_3m_sharpe = 0.0
    worst_3m_drawdown = 0.0
    window_pass_rate = 0.0
    if len(t_clean) >= window_min_trades:
        r_by_time = pd.Series(rvals_time.to_numpy(), index=t_clean).sort_index()
        month_ends = pd.date_range(r_by_time.index.min().normalize(), r_by_time.index.max().normalize(), freq="M", tz="UTC")
        win_scores: List[float] = []
        win_sharpes: List[float] = []
        win_dds: List[float] = []
        passes = 0
        windows = 0
        eq = _to_dataframe(equity_curve)
        eq_s: Optional[pd.Series] = None
        if not eq.empty:
            if "equity" in eq.columns:
                eq_s = pd.to_numeric(eq["equity"], errors="coerce")
            else:
                eq_s = pd.to_numeric(eq.iloc[:, 0], errors="coerce")
            eq_s.index = pd.to_datetime(eq.index, utc=True, errors="coerce")
            eq_s = eq_s.dropna().sort_index()
        for mend in month_ends:
            mstart = mend - pd.DateOffset(months=3)
            rv = r_by_time.loc[(r_by_time.index > mstart) & (r_by_time.index <= mend)]
            if len(rv) < window_min_trades:
                continue
            windows += 1
            exp = float(rv.mean())
            std = float(rv.std(ddof=0))
            shp = float((exp / std) * np.sqrt(min(len(rv), 252))) if std > 1e-12 else 0.0
            gp = float(rv[rv > 0].sum())
            gl = float(-rv[rv < 0].sum())
            pf_w = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            dd_w = 0.0
            if eq_s is not None and not eq_s.empty:
                eq_w = eq_s.loc[(eq_s.index > mstart) & (eq_s.index <= mend)]
                dd_w = _compute_max_drawdown(eq_w) if len(eq_w) > 1 else 0.0
            win_scores.append(exp)
            win_sharpes.append(shp)
            win_dds.append(dd_w)
            if (exp > 0.0) and (pf_w >= 1.0) and (dd_w <= window_dd_cap):
                passes += 1
        if windows > 0:
            rolling_3m_expectancy = float(win_scores[-1])
            rolling_3m_sharpe = float(win_sharpes[-1])
            worst_3m_drawdown = float(max(win_dds)) if win_dds else 0.0
            window_pass_rate = float(passes / windows)

    return {
        "expectancy_r": float(rvals.mean()),
        "net_expectancy_after_cost": net_expectancy,
        "profit_factor": float(pf),
        "win_rate": float(wins / max(1, wins + losses)),
        "sharpe_trade": sharpe_trade,
        "max_dd": float(_compute_max_drawdown(equity_curve)),
        "trades": int(len(trades)),
        "trades_per_month": trades_per_month,
        "trades_per_day": trades_per_day,
        "cost_per_trade": cost_per_trade,
        "financing_per_trade": financing_per_trade,
        "cost_to_gross_pnl_ratio": cost_to_gross_ratio,
        "avg_hold_hours": avg_hold_hours,
        "overnight_trade_pct": overnight_trade_pct,
        "signal_flip_rate": signal_flip_rate,
        "long_expectancy_r": float(long_r.mean()) if len(long_r) else 0.0,
        "short_expectancy_r": float(short_r.mean()) if len(short_r) else 0.0,
        "long_win_rate": float((long_r > 0).mean()) if len(long_r) else 0.0,
        "short_win_rate": float((short_r > 0).mean()) if len(short_r) else 0.0,
        "rolling_3m_expectancy": rolling_3m_expectancy,
        "rolling_3m_sharpe": rolling_3m_sharpe,
        "worst_3m_drawdown": worst_3m_drawdown,
        "window_pass_rate": window_pass_rate,
    }


def _session_breakdown(trade_log: pd.DataFrame, tz_name: str, sessions: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    trades = _to_dataframe(trade_log)
    if trades.empty:
        return {}

    if "entry_time" in trades.columns:
        t_idx = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    elif "timestamp" in trades.columns:
        t_idx = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    else:
        t_idx = pd.to_datetime(trades.index, utc=True, errors="coerce")

    valid_mask = ~pd.isna(t_idx)
    t_valid = pd.DatetimeIndex(pd.Series(t_idx)[valid_mask])
    sf = _session_flags(t_valid, tz_name, sessions)
    if sf.empty:
        return {}

    trades_valid = trades.loc[valid_mask].copy()
    if "r_multiple" in trades.columns:
        rvals = pd.to_numeric(trades_valid["r_multiple"], errors="coerce").fillna(0.0)
    elif "pnl" in trades.columns:
        rvals = pd.to_numeric(trades_valid["pnl"], errors="coerce").fillna(0.0)
    else:
        rvals = pd.Series(np.zeros(len(trades_valid)), index=trades_valid.index)
    rvals = pd.Series(rvals.to_numpy(), index=t_valid)

    out: Dict[str, Dict[str, float]] = {}
    for sess_col in ["sess_tokyo", "sess_london", "sess_newyork", "sess_overlap"]:
        mask = sf[sess_col] > 0
        if mask.sum() == 0:
            continue
        rv = rvals.loc[mask.index][mask]
        if rv.empty:
            continue
        out[sess_col] = {
            "trades": float(len(rv)),
            "expectancy_r": float(rv.mean()),
            "win_rate": float((rv > 0).mean()),
        }
    return out


def _regime_breakdown(trade_log: pd.DataFrame, regime_series: pd.Series) -> Dict[str, Dict[str, float]]:
    trades = _to_dataframe(trade_log)
    if trades.empty:
        return {}

    if "entry_time" in trades.columns:
        t_idx = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    elif "timestamp" in trades.columns:
        t_idx = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    else:
        t_idx = pd.to_datetime(trades.index, utc=True, errors="coerce")

    if "r_multiple" in trades.columns:
        rvals = pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0.0)
    elif "pnl" in trades.columns:
        rvals = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    else:
        rvals = pd.Series(np.zeros(len(trades)), index=trades.index)

    ts = pd.Series(t_idx, index=trades.index)
    valid = ts.notna()
    if valid.sum() == 0:
        return {}

    t_valid = pd.DatetimeIndex(ts.loc[valid])
    rv = pd.Series(rvals.loc[valid].to_numpy(), index=t_valid)
    rg = regime_series.reindex(t_valid).fillna("UNKNOWN")

    out: Dict[str, Dict[str, float]] = {}
    for regime in sorted(pd.unique(rg)):
        mask = rg == regime
        r = rv.loc[mask]
        if r.empty:
            continue
        out[str(regime)] = {
            "trades": float(len(r)),
            "expectancy_r": float(r.mean()),
            "win_rate": float((r > 0).mean()),
        }
    return out


def _select_regime_strategy_map(
    strategy_records: List[Dict[str, Any]],
    objective_cfg: Dict[str, Any],
    min_regime_trades: int = 20,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    if not strategy_records:
        return {}, {}

    fallback = max(strategy_records, key=lambda r: float(r.get("objective_score", -1e9)))
    mapping: Dict[str, str] = {}
    perf: Dict[str, Any] = {}

    for regime in REGIME_NAMES:
        best = None
        best_score = -1e9
        for rec in strategy_records:
            rb = rec.get("regime_breakdown", {})
            m = rb.get(regime, {})
            trades = float(m.get("trades", 0.0))
            if trades < float(min_regime_trades):
                continue
            exp = float(m.get("expectancy_r", 0.0))
            wr = float(m.get("win_rate", 0.0))
            support = min(1.0, trades / 30.0)
            score = (exp * support) + 0.05 * (wr - 0.5)
            if score > best_score:
                best_score = score
                best = rec
        if best is None:
            best = fallback
            best_score = float(best.get("objective_score", 0.0)) * 0.5
            fb_m = best.get("regime_breakdown", {}).get(regime, {})
            fb_trades = float(fb_m.get("trades", 0.0))
            if fb_trades < float(min_regime_trades):
                perf[regime] = {
                    "strategy": "",
                    "score": float(best_score),
                    "regime_metrics": fb_m,
                    "fallback": True,
                    "blocked_low_support": True,
                }
                continue
        mapping[regime] = str(best["strategy"])
        perf[regime] = {
            "strategy": str(best["strategy"]),
            "score": float(best_score),
            "regime_metrics": best.get("regime_breakdown", {}).get(regime, {}),
            "fallback": bool(best is fallback),
        }
    return mapping, perf


def _variant_grid(base_thresholds: Dict[str, Any], max_variants: int) -> List[Dict[str, Any]]:
    trend_mins = [0.50, 0.55, 0.60]
    range_mins = [0.50, 0.55]
    trans_mins = [0.45, 0.50]
    exh_ext = [1.8, 2.0, 2.2]

    variants: List[Dict[str, Any]] = []
    i = 0
    for tm in trend_mins:
        for rm in range_mins:
            for trm in trans_mins:
                for ee in exh_ext:
                    i += 1
                    v = {
                        "id": f"v{i:02d}",
                        "trend_min": tm,
                        "range_min": rm,
                        "transition_min": trm,
                        "exhaustion_extension_atr": ee,
                    }
                    variants.append(_deep_merge(base_thresholds, v))
                    if len(variants) >= max_variants:
                        return variants
    return variants


def _pair_prefs(symbol: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("pair_overrides", {}).get(symbol, {})


def _pair_cfg(symbol: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    pair = _pair_prefs(symbol, cfg)
    out = dict(cfg)
    for k in ["barrier", "model", "splits", "no_trade", "cost", "entry_refinement", "regime"]:
        if k in pair and isinstance(pair[k], dict):
            out[k] = _deep_merge(cfg.get(k, {}), pair.get(k, {}))
    if "entry_timeframes" in pair:
        out["entry_timeframes"] = list(pair.get("entry_timeframes", []))
    return out


def _data_quality_report(df: pd.DataFrame, tf: str, cfg: Dict[str, Any]) -> Dict[str, float]:
    qcfg = cfg.get("data_quality", {})
    x = _ensure_utc_index(df)
    if x.empty:
        return {"rows": 0.0, "missing_ratio": 1.0, "max_gap_bars": 0.0, "warn": 1.0}
    exp_sec = max(_timeframe_seconds(tf), 1)
    d = pd.Series(pd.DatetimeIndex(x.index).view("int64")).diff().dropna() / 1e9
    d = pd.to_numeric(d, errors="coerce").dropna()
    if d.empty:
        return {"rows": float(len(x)), "missing_ratio": 0.0, "max_gap_bars": 1.0, "warn": 0.0}
    max_gap_bars = float(d.max() / exp_sec)
    missing_ratio = float((d > (1.5 * exp_sec)).mean())
    max_gap_multiple = float(qcfg.get("max_time_gap_multiple", 6.0))
    max_missing = float(qcfg.get("max_missing_bar_ratio", 0.03))
    warn = float((max_gap_bars > max_gap_multiple) or (missing_ratio > max_missing))
    return {
        "rows": float(len(x)),
        "missing_ratio": missing_ratio,
        "max_gap_bars": max_gap_bars,
        "warn": warn,
    }


def _apply_pair_filters(signals: pd.DataFrame, base_df: pd.DataFrame, symbol: str, strategy: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    if signals.empty:
        return signals
    prefs = _pair_prefs(symbol, cfg)
    style = STRATEGY_STYLE[strategy]
    pref_styles = set(prefs.get("preferred_styles", []))
    pref_sessions = set(prefs.get("preferred_sessions", []))
    disable_session_filters = bool(cfg.get("disable_session_filters", False) or prefs.get("disable_session_filters", False))

    out = signals.copy()
    if pref_styles and style not in pref_styles:
        # Not forbidden, but stricter evidence required.
        col = next((c for c in out.columns if c.startswith("sig_")), None)
        if col:
            out = out[pd.to_numeric(out[col], errors="coerce").fillna(0.0) > 0]

    if pref_sessions and not disable_session_filters:
        mask = pd.Series(False, index=out.index)
        for s in pref_sessions:
            col = f"sess_{s}"
            if col in base_df.columns:
                mask = mask | (base_df.loc[out.index, col] > 0)
        out = out[mask]

    if bool(prefs.get("stricter_filters", False)):
        out = out[out["direction"].notna()]
        if "regime_confidence" in base_df.columns:
            out = out[base_df.loc[out.index, "regime_confidence"].fillna(0.0) >= 0.58]
        if "regime_score_margin" in base_df.columns:
            out = out[base_df.loc[out.index, "regime_score_margin"].fillna(0.0) >= 0.04]
    quality = prefs.get("signal_quality", {})
    if isinstance(quality, dict) and len(quality) > 0 and not out.empty:
        allowed_sessions = set(quality.get("allowed_sessions", []))
        if allowed_sessions and not disable_session_filters:
            qmask = pd.Series(False, index=out.index)
            for s in allowed_sessions:
                col = f"sess_{s}"
                if col in base_df.columns:
                    qmask = qmask | (base_df.loc[out.index, col] > 0)
            out = out[qmask]
        min_conf = float(quality.get("min_regime_confidence", 0.0))
        if min_conf > 0 and "regime_confidence" in base_df.columns:
            out = out[base_df.loc[out.index, "regime_confidence"].fillna(0.0) >= min_conf]
        min_margin = float(quality.get("min_regime_score_margin", 0.0))
        if min_margin > 0 and "regime_score_margin" in base_df.columns:
            out = out[base_df.loc[out.index, "regime_score_margin"].fillna(0.0) >= min_margin]
        max_atr_norm = float(quality.get("max_atr_norm", 0.0))
        if max_atr_norm > 0:
            atr = pd.to_numeric(base_df.get("atr14", _atr(base_df, 14)), errors="coerce")
            med = atr.rolling(200, min_periods=50).median()
            atr_norm = (atr / (med + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            out = out[atr_norm.reindex(out.index).fillna(1.0) <= max_atr_norm]
        max_body_atr = float(quality.get("max_body_atr", 0.0))
        if max_body_atr > 0:
            close = pd.to_numeric(base_df["close"], errors="coerce")
            open_ = pd.to_numeric(base_df["open"], errors="coerce")
            atr = pd.to_numeric(base_df.get("atr14", _atr(base_df, 14)), errors="coerce")
            body_atr = ((close - open_).abs() / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out = out[body_atr.reindex(out.index).fillna(0.0) <= max_body_atr]
    return out


def _refine_low_tf_entries(signals: pd.DataFrame, base_df: pd.DataFrame, ltf_tf: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    rules = cfg.get("entry_refinement", {})
    if not bool(rules.get("enabled", True)):
        return signals
    apply_tfs = {str(v).upper() for v in rules.get("apply_timeframes", ["M5", "M1"])}
    if str(ltf_tf).upper() not in apply_tfs:
        return signals
    if signals.empty:
        return signals

    out = signals.copy()
    min_conf = float(rules.get("min_regime_confidence", 0.56))
    min_margin = float(rules.get("min_regime_score_margin", 0.03))
    conf_bars = max(1, int(rules.get("confirmation_bars", 2)))
    min_body_atr = float(rules.get("min_body_atr", 0.08))
    max_atr_norm = float(rules.get("max_atr_norm", 2.5))

    if "regime_confidence" in base_df.columns:
        out = out[base_df.loc[out.index, "regime_confidence"].fillna(0.0) >= min_conf]
    if "regime_score_margin" in base_df.columns:
        out = out[base_df.loc[out.index, "regime_score_margin"].fillna(0.0) >= min_margin]
    if out.empty:
        return out

    close = pd.to_numeric(base_df["close"], errors="coerce")
    open_ = pd.to_numeric(base_df["open"], errors="coerce")
    if "atr14" in base_df.columns:
        atr = pd.to_numeric(base_df["atr14"], errors="coerce")
    else:
        atr = _atr(base_df, 14)
    body_atr = ((close - open_).abs() / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    med_atr = atr.rolling(200, min_periods=50).median()
    atr_norm = (atr / (med_atr + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    momentum = close.diff(conf_bars)
    long_confirm = momentum > 0
    short_confirm = momentum < 0
    confirm = np.where(out["direction"] > 0, long_confirm.reindex(out.index).fillna(False), short_confirm.reindex(out.index).fillna(False))

    out = out[pd.Series(confirm, index=out.index)]
    if out.empty:
        return out
    out = out[body_atr.reindex(out.index).fillna(0.0) >= min_body_atr]
    out = out[atr_norm.reindex(out.index).fillna(1.0) <= max_atr_norm]
    return out


def _build_signals_for_backtest(candidates: pd.DataFrame, probs: Optional[pd.Series], no_trade_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = candidates.copy()
    if probs is not None:
        out["model_probability"] = probs.reindex(out.index)

    if bool(no_trade_cfg.get("enabled", True)):
        th = float(no_trade_cfg.get("probability_threshold", 0.58))
        if "model_probability" in out.columns:
            out = out[out["model_probability"].fillna(0.0) >= th]

    if "signal" not in out.columns:
        out["signal"] = out["direction"]
    return out


def _compute_brier_ece(probs: Optional[pd.Series], y_true: pd.Series, bins: int = 10) -> Dict[str, float]:
    if probs is None:
        return {"brier_score": float("nan"), "ece": float("nan")}
    p = pd.to_numeric(probs, errors="coerce").clip(0.0, 1.0)
    yb = (pd.to_numeric(y_true, errors="coerce").fillna(0.0) > 0).astype(float)
    z = pd.concat([p.rename("p"), yb.rename("y")], axis=1).dropna(how="any")
    if z.empty:
        return {"brier_score": float("nan"), "ece": float("nan")}
    brier = float(((z["p"] - z["y"]) ** 2).mean())
    bins = max(4, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            m = (z["p"] >= lo) & (z["p"] <= hi)
        else:
            m = (z["p"] >= lo) & (z["p"] < hi)
        if not m.any():
            continue
        acc = float(z.loc[m, "y"].mean())
        conf = float(z.loc[m, "p"].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
    return {"brier_score": brier, "ece": float(ece)}


def _score_objective(metrics: Dict[str, Any], objective_cfg: Dict[str, Any]) -> float:
    ew = float(objective_cfg.get("expectancy_weight", 1.0))
    ddp = float(objective_cfg.get("max_dd_penalty", 0.5))
    cp = float(objective_cfg.get("cost_penalty", 0.2))
    sw = float(objective_cfg.get("sharpe_weight", 0.05))
    wp = float(objective_cfg.get("window_pass_weight", 0.2))
    return (
        ew * float(metrics.get("expectancy_r", 0.0))
        - ddp * float(metrics.get("max_dd", 0.0))
        - cp * float(metrics.get("cost_per_trade", 0.0))
        + sw * float(metrics.get("sharpe_trade", 0.0))
        + wp * float(metrics.get("window_pass_rate", 0.0))
    )


def _safe_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    return str(obj)


def _save_model_artifact(model_obj: Any, meta: Dict[str, Any], out_dir: Path, base_name: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{base_name}.pkl"
    meta_path = out_dir / f"{base_name}.json"

    model_saved = False
    try:
        with model_path.open("wb") as f:
            pickle.dump(model_obj, f)
        model_saved = True
    except Exception:
        meta["model_pickle_error"] = "Model object not pickleable; metadata saved only."

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(_safe_jsonable(meta), f, indent=2)

    return {
        "model_path": str(model_path) if model_saved else "",
        "meta_path": str(meta_path),
    }


def _save_trade_log_artifact(trade_log: Any, out_dir: Path, base_name: str) -> str:
    df = _to_dataframe(trade_log)
    if df.empty:
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{base_name}.trades.csv"
    df.to_csv(p, index=False)
    return str(p)


def _prepare_atr_series(df_ltf: pd.DataFrame, ltf_features: pd.DataFrame) -> pd.Series:
    for col in ["atr", "atr14", "m15_atr_pct", "h1_atr_pct"]:
        if col in ltf_features.columns:
            s = pd.to_numeric(ltf_features[col], errors="coerce")
            if "pct" in col:
                close = pd.to_numeric(df_ltf["close"], errors="coerce")
                s = (s * close).replace([np.inf, -np.inf], np.nan)
            return s.ffill()
    return _atr(df_ltf, 14)


def _ranking_headers() -> List[str]:
    return [
        "symbol",
        "ltf_timeframe",
        "strategy",
        "style",
        "hybrid_router",
        "regime_variant",
        "regime_strategy_map_json",
        "regime_router_perf_json",
        "allowed_regimes",
        "objective_score",
        "expectancy_r",
        "net_expectancy_after_cost",
        "profit_factor",
        "win_rate",
        "sharpe_trade",
        "max_dd",
        "rolling_3m_expectancy",
        "rolling_3m_sharpe",
        "worst_3m_drawdown",
        "window_pass_rate",
        "trades",
        "trades_per_month",
        "trades_per_day",
        "data_quality_missing_ratio",
        "data_quality_max_gap_bars",
        "data_quality_warn",
        "cost_per_trade",
        "financing_per_trade",
        "cost_to_gross_pnl_ratio",
        "avg_hold_hours",
        "overnight_trade_pct",
        "signal_flip_rate",
        "long_expectancy_r",
        "short_expectancy_r",
        "long_win_rate",
        "short_win_rate",
        "brier_score",
        "ece",
        "calibrator_type",
        "fold_diagnostics_json",
        "feature_transform_stats_json",
        "calibrated_no_trade_threshold",
        "session_breakdown_json",
        "regime_breakdown_json",
        "no_trade_threshold",
        "kill_switch",
        "deploy_eligible",
        "deployment_role",
        "model_selected",
        "artifact_model_path",
        "artifact_meta_path",
        "artifact_trade_log_path",
    ]


def _slice_last_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    if df.empty or years <= 0:
        return df
    end_ts = pd.DatetimeIndex(df.index).max()
    start_ts = end_ts - pd.Timedelta(days=365 * years)
    return df.loc[df.index >= start_ts]


def run_research(cfg: Dict[str, Any], fns: Optional[PipelineFns], template_only: bool = False) -> Dict[str, Any]:
    out_cfg = cfg.get("output", {})
    ranking_csv = Path(out_cfg.get("ranking_csv", "data/research/regime_strategy_ranking.csv"))
    ranking_csv.parent.mkdir(parents=True, exist_ok=True)

    if template_only:
        pd.DataFrame(columns=_ranking_headers()).to_csv(ranking_csv, index=False)
        return {"ranking_csv": str(ranking_csv), "rows": 0, "selected": 0}
    if fns is None:
        raise ValueError("Pipeline functions are required unless --template-only is used.")

    ltf_tfs_raw = cfg.get("entry_timeframes", []) or [cfg.get("ltf_timeframe", "M15")]
    if isinstance(ltf_tfs_raw, str):
        default_ltf_tfs = [ltf_tfs_raw]
    else:
        default_ltf_tfs = [str(v) for v in ltf_tfs_raw]
    h1_tf = str(cfg.get("htf_timeframes", {}).get("h1", "H1"))
    d1_tf = str(cfg.get("htf_timeframes", {}).get("d1", "D1"))

    symbols = list(cfg.get("symbols", []))
    tz_name = str(cfg.get("timezone", "Europe/London"))
    train_cfg = cfg.get("training", {})
    training_years = int(train_cfg.get("years", 10))
    symbol_batch_size = max(1, int(train_cfg.get("symbol_batch_size", 1)))

    exp_cfg = cfg.get("exploration", {})
    max_variants = int(exp_cfg.get("max_variants", 16))
    top_k = int(exp_cfg.get("top_k", 3))
    hybrid_min_per_symbol = max(0, int(exp_cfg.get("hybrid_min_per_symbol", 1)))
    min_trades = int(exp_cfg.get("min_trades", 30))
    objective_cfg = exp_cfg.get("objective", {})
    risk_ctrl = cfg.get("risk_controls", {})

    regime_cfg = cfg.get("regime", {})
    thresholds = regime_cfg.get("thresholds", {})
    variants = exp_cfg.get("regime_variants", [])
    if variants:
        variants = [dict(v) for v in variants][:max_variants]
        for i, v in enumerate(variants, start=1):
            v.setdefault("id", f"v{i:02d}")
    else:
        variants = _variant_grid(thresholds, max_variants=max_variants)

    rows: List[Dict[str, Any]] = []
    model_candidates: List[Dict[str, Any]] = []

    for b_start in range(0, len(symbols), symbol_batch_size):
        batch_symbols = symbols[b_start : b_start + symbol_batch_size]
        for symbol in batch_symbols:
            sym_cfg = _pair_cfg(symbol, cfg)
            ltf_tfs_raw_sym = sym_cfg.get("entry_timeframes", default_ltf_tfs) or default_ltf_tfs
            if isinstance(ltf_tfs_raw_sym, str):
                ltf_tfs = [ltf_tfs_raw_sym]
            else:
                ltf_tfs = [str(v) for v in ltf_tfs_raw_sym]
            try:
                df_h1 = _ensure_utc_index(fns.load_ohlcv(symbol, h1_tf))
                df_d1 = _ensure_utc_index(fns.load_ohlcv(symbol, d1_tf))
            except Exception as exc:
                print(f"Skipping symbol {symbol}: data load failed ({exc})", flush=True)
                continue

            if training_years > 0:
                df_h1 = _slice_last_years(df_h1, training_years)
                df_d1 = _slice_last_years(df_d1, training_years)
            if df_h1.empty or df_d1.empty:
                continue

            for ltf_tf in ltf_tfs:
                try:
                    df_ltf = _ensure_utc_index(fns.load_ohlcv(symbol, ltf_tf))
                except Exception as exc:
                    print(f"Skipping {symbol} {ltf_tf}: data load failed ({exc})", flush=True)
                    continue
                if training_years > 0:
                    df_ltf = _slice_last_years(df_ltf, training_years)
                if df_ltf.empty:
                    continue

                dq = _data_quality_report(df_ltf, ltf_tf, sym_cfg)

                ltf_features = fns.make_features(df_ltf, ltf_tf, sym_cfg)
                ltf_features = _ensure_utc_index(ltf_features)
                ltf_features = ltf_features.reindex(df_ltf.index)

                sess = _session_flags(df_ltf.index, tz_name, sym_cfg.get("sessions", {}))
                base_df = pd.concat([df_ltf, ltf_features, sess], axis=1)

                atr_series = _prepare_atr_series(df_ltf, ltf_features)
                y_raw = fns.make_labels(df_ltf, atr_series, sym_cfg.get("barrier", {}))
                y, label_meta = _parse_labels(y_raw, df_ltf.index)

                for variant in variants:
                    regime_feat = _compute_regime_features(df_ltf, df_h1, df_d1, sym_cfg.get("regime", regime_cfg), variant)
                    merged = pd.concat([base_df, regime_feat], axis=1)
                    strategy_eval_records: List[Dict[str, Any]] = []

                    for strategy in STRATEGY_NAMES:
                        sig = _generate_strategy_signals(
                            merged,
                            strategy,
                            {
                                "ltf_seconds": _timeframe_seconds(ltf_tf),
                                **sym_cfg.get("strategies", {}).get(strategy, {}),
                            },
                        )
                        if sig.empty:
                            continue

                        allowed = set(STRATEGY_ALLOWED_REGIMES[strategy])
                        sig = sig.join(merged[["regime"]], how="left")
                        sig = sig[sig["regime"].isin(allowed)]
                        if sig.empty:
                            continue

                        sig = _apply_pair_filters(sig, merged, symbol, strategy, sym_cfg)
                        sig = _refine_low_tf_entries(sig, merged, ltf_tf, sym_cfg)
                        if sig.empty:
                            continue

                        sig_idx = sig.index.intersection(y.index)
                        if len(sig_idx) < 20:
                            continue

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

                        X_raw = merged.loc[sig_idx, feature_cols].copy()
                        y_sig = y.loc[sig_idx].copy()
                        X, y_sig, prep_meta = _prepare_training_matrix(X_raw, y_sig, fns)
                        if len(X) < 20:
                            continue

                        horizon = int(sym_cfg.get("barrier", {}).get("horizon_bars", 8))
                        label_end_ts = _estimate_label_end_ts(pd.DatetimeIndex(X.index), pd.DatetimeIndex(df_ltf.index), horizon)
                        splits_cfg_local = dict(sym_cfg.get("splits", {}))
                        custom_splits = _build_custom_purged_splits(pd.DatetimeIndex(X.index), label_end_ts, splits_cfg_local, fns)
                        if custom_splits:
                            splits_cfg_local["custom_splits"] = [(tr.tolist(), te.tolist()) for tr, te in custom_splits]

                        wf_raw = fns.walkforward_train_eval(X, y_sig, sym_cfg.get("model", {}), splits_cfg_local)
                        wf_metrics, wf_models = _extract_wf_output(wf_raw)
                        probs = _extract_probability_series(wf_metrics, X.index)
                        fold_diagnostics: List[Dict[str, Any]] = []
                        if probs is not None and callable(fns.compute_fold_diagnostics) and custom_splits:
                            exp_cost = pd.to_numeric(merged.get("expected_cost_proxy_bps", pd.Series(index=X.index, data=0.0)), errors="coerce")
                            exp_cost = exp_cost.reindex(X.index).fillna(0.0) / 10000.0
                            for fid, (_, te) in enumerate(custom_splits, start=1):
                                tei = np.asarray(te, dtype=int)
                                if len(tei) == 0:
                                    continue
                                y_te = y_sig.iloc[tei]
                                p_te = probs.iloc[tei]
                                cinfo = {
                                    "expected_cost": exp_cost.iloc[tei],
                                    "threshold": float(sym_cfg.get("no_trade", {}).get("probability_threshold", 0.58)),
                                }
                                try:
                                    fd = fns.compute_fold_diagnostics(y_te, p_te, cinfo, fid)
                                    if isinstance(fd, dict):
                                        fold_diagnostics.append(fd)
                                except Exception:
                                    pass

                        calibrator = None
                        probs_cal = probs
                        calibrator_type = ""
                        if probs is not None and callable(fns.fit_probability_calibrator) and callable(fns.apply_probability_calibrator):
                            try:
                                reg_conf = pd.to_numeric(merged.get("regime_confidence", pd.Series(index=X.index, data=np.nan)), errors="coerce")
                                reg_bucket = (reg_conf.reindex(X.index).fillna(0.0) >= float(exp_cfg.get("calibration_regime_conf_cut", 0.58))).astype(int)
                                calibrator = fns.fit_probability_calibrator(
                                    probs.to_numpy(dtype=float),
                                    (pd.to_numeric(y_sig, errors="coerce").fillna(0.0) > 0).astype(int).to_numpy(),
                                    regime_bucket=reg_bucket.to_numpy(),
                                    isotonic_min_samples=int(exp_cfg.get("isotonic_min_samples", 1000)),
                                    bucket_min_samples=int(exp_cfg.get("calibration_bucket_min_samples", 300)),
                                )
                                probs_cal_arr = fns.apply_probability_calibrator(
                                    calibrator,
                                    probs.to_numpy(dtype=float),
                                    regime_bucket=reg_bucket.to_numpy(),
                                )
                                probs_cal = pd.Series(np.asarray(probs_cal_arr, dtype=float), index=probs.index)
                                if isinstance(calibrator, dict):
                                    calibrator_type = str(calibrator.get("type", ""))
                            except Exception:
                                probs_cal = probs

                        calib = _compute_brier_ece(
                            probs_cal,
                            y_sig,
                            bins=int(sym_cfg.get("no_trade", {}).get("ece_bins", 10)),
                        )

                        no_trade_cfg = sym_cfg.get("no_trade", {})
                        base_threshold = float(no_trade_cfg.get("probability_threshold", 0.58))
                        threshold_grid = [base_threshold]
                        if bool(no_trade_cfg.get("enabled", True)) and bool(no_trade_cfg.get("calibrate_threshold", True)) and probs is not None:
                            threshold_grid = sorted(
                                {
                                    float(v)
                                    for v in no_trade_cfg.get("threshold_grid", [0.52, 0.55, 0.58, 0.62, 0.66])
                                    if 0.0 <= float(v) <= 1.0
                                }
                            )
                            if not threshold_grid:
                                threshold_grid = [base_threshold]

                        best_run: Optional[Dict[str, Any]] = None
                        no_trade_obj = no_trade_cfg.get("objective", objective_cfg)
                        min_nt_trades = int(no_trade_cfg.get("min_trades", min_trades))
                        max_nt_dd = float(no_trade_cfg.get("max_dd_cap", 0.20))
                        for th in threshold_grid:
                            cfg_nt = dict(no_trade_cfg)
                            cfg_nt["probability_threshold"] = float(th)
                            backtest_signals = _build_signals_for_backtest(sig.loc[X.index], probs_cal, cfg_nt)
                            # Calibration fallback: if calibrated probabilities collapse and filter out
                            # all candidates, retry with raw probabilities to avoid false-empty runs.
                            if backtest_signals.empty and (probs is not None) and (probs_cal is not None):
                                if not probs.equals(probs_cal):
                                    backtest_signals = _build_signals_for_backtest(sig.loc[X.index], probs, cfg_nt)
                            if (
                                (not backtest_signals.empty)
                                and callable(fns.compute_expected_value)
                                and callable(fns.apply_trade_gating)
                            ):
                                prob_s = pd.to_numeric(backtest_signals.get("model_probability", pd.Series(index=backtest_signals.index, data=np.nan)), errors="coerce").fillna(0.0)
                                gw = pd.Series(float(sym_cfg.get("barrier", {}).get("up_atr_mult", 1.0)), index=backtest_signals.index, dtype=float)
                                gl = pd.Series(float(sym_cfg.get("barrier", {}).get("down_atr_mult", 1.0)), index=backtest_signals.index, dtype=float)
                                ec = (
                                    pd.to_numeric(merged.get("expected_cost_proxy_bps", pd.Series(index=backtest_signals.index, data=0.0)), errors="coerce")
                                    .reindex(backtest_signals.index)
                                    .fillna(0.0)
                                    / 10000.0
                                )
                                ev = fns.compute_expected_value(prob_s, gw, gl, ec)
                                gate_df = pd.DataFrame(index=backtest_signals.index)
                                gate_df["p"] = prob_s
                                gate_df["ev"] = ev
                                spread_shock = (
                                    pd.to_numeric(merged.get("spread_proxy_bps", pd.Series(index=backtest_signals.index, data=0.0)), errors="coerce")
                                    .reindex(backtest_signals.index)
                                    .fillna(0.0)
                                )
                                shock_thr = float(spread_shock.quantile(0.8)) if len(spread_shock) else 0.0
                                dyn_thr = float(th) + float(exp_cfg.get("spread_shock_threshold_bump", 0.03)) * (spread_shock > shock_thr).astype(float)
                                gate_df["dynamic_threshold"] = dyn_thr
                                gate = fns.apply_trade_gating(
                                    gate_df,
                                    p_col="p",
                                    ev_col="ev",
                                    min_ev=float(exp_cfg.get("min_ev", 0.0)),
                                    base_p_threshold=float(th),
                                    dynamic_threshold_col="dynamic_threshold",
                                )
                                backtest_signals = backtest_signals[gate.reindex(backtest_signals.index).fillna(0).astype(int) > 0]
                            if backtest_signals.empty:
                                continue
                            bt_raw = fns.backtest_from_signals(df_ltf, backtest_signals, sym_cfg.get("cost", {}))
                            if isinstance(bt_raw, tuple) and len(bt_raw) >= 2:
                                trade_log, equity_curve = bt_raw[0], bt_raw[1]
                            else:
                                trade_log = (
                                    bt_raw.get("trade_log", pd.DataFrame()) if isinstance(bt_raw, dict) else pd.DataFrame()
                                )
                                equity_curve = (
                                    bt_raw.get("equity_curve", pd.Series(dtype=float))
                                    if isinstance(bt_raw, dict)
                                    else pd.Series(dtype=float)
                                )

                            tm = _trade_metrics(
                                _to_dataframe(trade_log),
                                equity_curve,
                                df_ltf.index,
                                settings=exp_cfg,
                                signals=backtest_signals,
                            )
                            score = _score_objective(tm, no_trade_obj)
                            feasible = (int(tm["trades"]) >= min_nt_trades) and (float(tm["max_dd"]) <= max_nt_dd)
                            if best_run is None:
                                best_run = {
                                    "tm": tm,
                                    "score": score,
                                    "threshold": float(th),
                                    "trade_log": trade_log,
                                    "equity_curve": equity_curve,
                                    "signals": backtest_signals,
                                    "feasible": feasible,
                                }
                                continue
                            if feasible and not best_run["feasible"]:
                                best_run = {
                                    "tm": tm,
                                    "score": score,
                                    "threshold": float(th),
                                    "trade_log": trade_log,
                                    "equity_curve": equity_curve,
                                    "signals": backtest_signals,
                                    "feasible": feasible,
                                }
                                continue
                            if feasible == best_run["feasible"] and score > float(best_run["score"]):
                                best_run = {
                                    "tm": tm,
                                    "score": score,
                                    "threshold": float(th),
                                    "trade_log": trade_log,
                                    "equity_curve": equity_curve,
                                    "signals": backtest_signals,
                                    "feasible": feasible,
                                }

                        if best_run is None:
                            continue

                        tm = best_run["tm"]
                        trade_log_df = _to_dataframe(best_run["trade_log"])
                        sess_break = _session_breakdown(trade_log_df, tz_name, sym_cfg.get("sessions", {}))
                        regime_break = _regime_breakdown(trade_log_df, merged["regime"])
                        obj_score = _score_objective(tm, objective_cfg)
                        kill_switch = (
                            (float(tm.get("expectancy_r", 0.0)) <= float(risk_ctrl.get("kill_switch_expectancy_max", 0.0)))
                            and (float(tm.get("sharpe_trade", 0.0)) <= float(risk_ctrl.get("kill_switch_sharpe_max", 0.0)))
                            and (
                                float(tm.get("window_pass_rate", 0.0))
                                <= float(risk_ctrl.get("kill_switch_window_pass_max", 0.35))
                            )
                        )
                        deploy_eligible = (
                            (not kill_switch)
                            and (int(tm["trades"]) >= min_trades)
                            and (float(tm.get("window_pass_rate", 0.0)) >= float(exp_cfg.get("min_window_pass_rate", 0.40)))
                        )
                        hf = _pair_prefs(symbol, sym_cfg).get("hard_filters", {})
                        deploy_eligible = deploy_eligible and (
                            float(tm.get("expectancy_r", 0.0)) >= float(hf.get("expectancy_min", 0.0))
                        )
                        deploy_eligible = deploy_eligible and (
                            float(tm.get("net_expectancy_after_cost", 0.0)) >= float(hf.get("net_expectancy_min", 0.0))
                        )
                        deploy_eligible = deploy_eligible and (
                            float(tm.get("window_pass_rate", 0.0)) >= float(hf.get("window_pass_min", 0.0))
                        )

                        row = {
                            "symbol": symbol,
                            "ltf_timeframe": str(ltf_tf),
                            "strategy": strategy,
                            "style": STRATEGY_STYLE[strategy],
                            "hybrid_router": False,
                            "regime_variant": variant.get("id", "v0"),
                            "regime_strategy_map_json": "{}",
                            "regime_router_perf_json": "{}",
                            "allowed_regimes": ",".join(STRATEGY_ALLOWED_REGIMES[strategy]),
                            "objective_score": round(float(obj_score), 6),
                            "expectancy_r": round(float(tm["expectancy_r"]), 6),
                            "net_expectancy_after_cost": round(float(tm["net_expectancy_after_cost"]), 6),
                            "profit_factor": round(float(tm["profit_factor"]), 6),
                            "win_rate": round(float(tm["win_rate"]), 6),
                            "sharpe_trade": round(float(tm["sharpe_trade"]), 6),
                            "max_dd": round(float(tm["max_dd"]), 6),
                            "rolling_3m_expectancy": round(float(tm["rolling_3m_expectancy"]), 6),
                            "rolling_3m_sharpe": round(float(tm["rolling_3m_sharpe"]), 6),
                            "worst_3m_drawdown": round(float(tm["worst_3m_drawdown"]), 6),
                            "window_pass_rate": round(float(tm["window_pass_rate"]), 6),
                            "trades": int(tm["trades"]),
                            "trades_per_month": round(float(tm["trades_per_month"]), 4),
                            "trades_per_day": round(float(tm["trades_per_day"]), 4),
                            "data_quality_missing_ratio": round(float(dq.get("missing_ratio", 0.0)), 6),
                            "data_quality_max_gap_bars": round(float(dq.get("max_gap_bars", 0.0)), 4),
                            "data_quality_warn": bool(dq.get("warn", 0.0) > 0),
                            "cost_per_trade": round(float(tm["cost_per_trade"]), 6),
                            "financing_per_trade": round(float(tm["financing_per_trade"]), 6),
                            "cost_to_gross_pnl_ratio": round(float(tm["cost_to_gross_pnl_ratio"]), 6),
                            "avg_hold_hours": round(float(tm["avg_hold_hours"]), 4),
                            "overnight_trade_pct": round(float(tm["overnight_trade_pct"]), 6),
                            "signal_flip_rate": round(float(tm["signal_flip_rate"]), 6),
                            "long_expectancy_r": round(float(tm["long_expectancy_r"]), 6),
                            "short_expectancy_r": round(float(tm["short_expectancy_r"]), 6),
                            "long_win_rate": round(float(tm["long_win_rate"]), 6),
                            "short_win_rate": round(float(tm["short_win_rate"]), 6),
                            "brier_score": round(float(calib.get("brier_score", float("nan"))), 6)
                            if pd.notna(calib.get("brier_score", np.nan))
                            else np.nan,
                            "ece": round(float(calib.get("ece", float("nan"))), 6)
                            if pd.notna(calib.get("ece", np.nan))
                            else np.nan,
                            "calibrator_type": calibrator_type,
                            "fold_diagnostics_json": json.dumps(_safe_jsonable(fold_diagnostics), separators=(",", ":")),
                            "feature_transform_stats_json": json.dumps(
                                _safe_jsonable(prep_meta.get("feature_transform_stats", {})), separators=(",", ":")
                            ),
                            "calibrated_no_trade_threshold": float(best_run["threshold"]),
                            "session_breakdown_json": json.dumps(_safe_jsonable(sess_break), separators=(",", ":")),
                            "regime_breakdown_json": json.dumps(_safe_jsonable(regime_break), separators=(",", ":")),
                            "no_trade_threshold": float(sym_cfg.get("no_trade", {}).get("probability_threshold", 0.58)),
                            "kill_switch": bool(kill_switch),
                            "deploy_eligible": bool(deploy_eligible),
                            "deployment_role": "",
                            "model_selected": False,
                            "artifact_model_path": "",
                            "artifact_meta_path": "",
                            "artifact_trade_log_path": "",
                        }
                        rows.append(row)

                        model_candidates.append(
                            {
                                "row": row,
                                "models": wf_models,
                                "wf_metrics": wf_metrics,
                                "feature_cols": list(X.columns),
                                "label_meta": label_meta,
                                "regime_variant": variant,
                                "sym_cfg": sym_cfg,
                                "calibrator": calibrator,
                                "session_filter": _pair_prefs(symbol, sym_cfg).get("preferred_sessions", []),
                                "ltf_timeframe": str(ltf_tf),
                                "trade_log": best_run.get("trade_log"),
                            }
                        )
                        strategy_eval_records.append(
                            {
                                "strategy": strategy,
                                "signals": sig.loc[X.index].copy(),
                                "objective_score": float(obj_score),
                                "regime_breakdown": regime_break,
                                "feature_cols": feature_cols,
                            }
                        )

                    # Build hybrid regime router candidate: choose best strategy per regime.
                    regime_map, regime_router_perf = _select_regime_strategy_map(
                        strategy_eval_records,
                        objective_cfg,
                        min_regime_trades=int(exp_cfg.get("regime_router_min_regime_trades", 20)),
                    )
                    min_mapped = int(exp_cfg.get("regime_router_min_mapped_regimes", 2))
                    if regime_map and (len(regime_map) >= max(1, min_mapped)):
                        strat_to_record = {str(r["strategy"]): r for r in strategy_eval_records}
                        hybrid_parts: List[pd.DataFrame] = []
                        for regime_name, strat_name in regime_map.items():
                            rec = strat_to_record.get(strat_name)
                            if rec is None:
                                continue
                            s = rec["signals"]
                            ridx = merged.index[merged["regime"] == regime_name]
                            idx = s.index.intersection(ridx)
                            if len(idx) == 0:
                                continue
                            part = s.loc[idx].copy()
                            part["router_regime"] = regime_name
                            part["router_strategy"] = strat_name
                            hybrid_parts.append(part)

                        if hybrid_parts:
                            hybrid_sig = pd.concat(hybrid_parts, axis=0).sort_index()
                            hybrid_sig = hybrid_sig[~hybrid_sig.index.duplicated(keep="last")]
                            sig_idx = hybrid_sig.index.intersection(y.index)
                            if len(sig_idx) >= 20:
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

                                sig_extra_cols = [
                                    c
                                    for c in hybrid_sig.columns
                                    if c.startswith("sig_") or c in {"direction", "router_regime", "router_strategy", "sig_strategy"}
                                ]
                                X_raw = merged.loc[sig_idx, feature_cols].copy().join(hybrid_sig.loc[sig_idx, sig_extra_cols], how="left")
                                y_sig = y.loc[sig_idx].copy()
                                for c in sig_extra_cols:
                                    if c.startswith("sig_") or c == "direction":
                                        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)

                                X, y_sig, prep_meta = _prepare_training_matrix(X_raw, y_sig, fns)
                                if len(X) >= 20:
                                    horizon = int(sym_cfg.get("barrier", {}).get("horizon_bars", 8))
                                    label_end_ts = _estimate_label_end_ts(pd.DatetimeIndex(X.index), pd.DatetimeIndex(df_ltf.index), horizon)
                                    splits_cfg_local = dict(sym_cfg.get("splits", {}))
                                    custom_splits = _build_custom_purged_splits(pd.DatetimeIndex(X.index), label_end_ts, splits_cfg_local, fns)
                                    if custom_splits:
                                        splits_cfg_local["custom_splits"] = [(tr.tolist(), te.tolist()) for tr, te in custom_splits]

                                    wf_raw = fns.walkforward_train_eval(
                                        X, y_sig, sym_cfg.get("model", {}), splits_cfg_local
                                    )
                                    wf_metrics, wf_models = _extract_wf_output(wf_raw)
                                    probs = _extract_probability_series(wf_metrics, X.index)
                                    fold_diagnostics: List[Dict[str, Any]] = []
                                    if probs is not None and callable(fns.compute_fold_diagnostics) and custom_splits:
                                        exp_cost = pd.to_numeric(
                                            merged.get("expected_cost_proxy_bps", pd.Series(index=X.index, data=0.0)),
                                            errors="coerce",
                                        )
                                        exp_cost = exp_cost.reindex(X.index).fillna(0.0) / 10000.0
                                        for fid, (_, te) in enumerate(custom_splits, start=1):
                                            tei = np.asarray(te, dtype=int)
                                            if len(tei) == 0:
                                                continue
                                            cinfo = {
                                                "expected_cost": exp_cost.iloc[tei],
                                                "threshold": float(sym_cfg.get("no_trade", {}).get("probability_threshold", 0.58)),
                                            }
                                            try:
                                                fd = fns.compute_fold_diagnostics(y_sig.iloc[tei], probs.iloc[tei], cinfo, fid)
                                                if isinstance(fd, dict):
                                                    fold_diagnostics.append(fd)
                                            except Exception:
                                                pass

                                    calibrator = None
                                    probs_cal = probs
                                    calibrator_type = ""
                                    if probs is not None and callable(fns.fit_probability_calibrator) and callable(fns.apply_probability_calibrator):
                                        try:
                                            reg_conf = pd.to_numeric(
                                                merged.get("regime_confidence", pd.Series(index=X.index, data=np.nan)),
                                                errors="coerce",
                                            )
                                            reg_bucket = (
                                                reg_conf.reindex(X.index).fillna(0.0)
                                                >= float(exp_cfg.get("calibration_regime_conf_cut", 0.58))
                                            ).astype(int)
                                            calibrator = fns.fit_probability_calibrator(
                                                probs.to_numpy(dtype=float),
                                                (pd.to_numeric(y_sig, errors="coerce").fillna(0.0) > 0).astype(int).to_numpy(),
                                                regime_bucket=reg_bucket.to_numpy(),
                                                isotonic_min_samples=int(exp_cfg.get("isotonic_min_samples", 1000)),
                                                bucket_min_samples=int(exp_cfg.get("calibration_bucket_min_samples", 300)),
                                            )
                                            probs_cal_arr = fns.apply_probability_calibrator(
                                                calibrator,
                                                probs.to_numpy(dtype=float),
                                                regime_bucket=reg_bucket.to_numpy(),
                                            )
                                            probs_cal = pd.Series(np.asarray(probs_cal_arr, dtype=float), index=probs.index)
                                            if isinstance(calibrator, dict):
                                                calibrator_type = str(calibrator.get("type", ""))
                                        except Exception:
                                            probs_cal = probs
                                    calib = _compute_brier_ece(
                                        probs_cal,
                                        y_sig,
                                        bins=int(sym_cfg.get("no_trade", {}).get("ece_bins", 10)),
                                    )

                                    no_trade_cfg = sym_cfg.get("no_trade", {})
                                    base_threshold = float(no_trade_cfg.get("probability_threshold", 0.58))
                                    threshold_grid = [base_threshold]
                                    if bool(no_trade_cfg.get("enabled", True)) and bool(no_trade_cfg.get("calibrate_threshold", True)) and probs is not None:
                                        threshold_grid = sorted(
                                            {
                                                float(v)
                                                for v in no_trade_cfg.get("threshold_grid", [0.52, 0.55, 0.58, 0.62, 0.66])
                                                if 0.0 <= float(v) <= 1.0
                                            }
                                        )
                                        if not threshold_grid:
                                            threshold_grid = [base_threshold]

                                    best_run: Optional[Dict[str, Any]] = None
                                    no_trade_obj = no_trade_cfg.get("objective", objective_cfg)
                                    min_nt_trades = int(no_trade_cfg.get("min_trades", min_trades))
                                    max_nt_dd = float(no_trade_cfg.get("max_dd_cap", 0.20))

                                    for th in threshold_grid:
                                        cfg_nt = dict(no_trade_cfg)
                                        cfg_nt["probability_threshold"] = float(th)
                                        backtest_signals = _build_signals_for_backtest(hybrid_sig.loc[X.index], probs_cal, cfg_nt)
                                        # Calibration fallback for hybrid router as well.
                                        if backtest_signals.empty and (probs is not None) and (probs_cal is not None):
                                            if not probs.equals(probs_cal):
                                                backtest_signals = _build_signals_for_backtest(hybrid_sig.loc[X.index], probs, cfg_nt)
                                        if (
                                            (not backtest_signals.empty)
                                            and callable(fns.compute_expected_value)
                                            and callable(fns.apply_trade_gating)
                                        ):
                                            prob_s = pd.to_numeric(
                                                backtest_signals.get("model_probability", pd.Series(index=backtest_signals.index, data=np.nan)),
                                                errors="coerce",
                                            ).fillna(0.0)
                                            gw = pd.Series(float(sym_cfg.get("barrier", {}).get("up_atr_mult", 1.0)), index=backtest_signals.index, dtype=float)
                                            gl = pd.Series(float(sym_cfg.get("barrier", {}).get("down_atr_mult", 1.0)), index=backtest_signals.index, dtype=float)
                                            ec = (
                                                pd.to_numeric(
                                                    merged.get("expected_cost_proxy_bps", pd.Series(index=backtest_signals.index, data=0.0)),
                                                    errors="coerce",
                                                )
                                                .reindex(backtest_signals.index)
                                                .fillna(0.0)
                                                / 10000.0
                                            )
                                            ev = fns.compute_expected_value(prob_s, gw, gl, ec)
                                            gate_df = pd.DataFrame(index=backtest_signals.index)
                                            gate_df["p"] = prob_s
                                            gate_df["ev"] = ev
                                            spread_shock = (
                                                pd.to_numeric(
                                                    merged.get("spread_proxy_bps", pd.Series(index=backtest_signals.index, data=0.0)),
                                                    errors="coerce",
                                                )
                                                .reindex(backtest_signals.index)
                                                .fillna(0.0)
                                            )
                                            shock_thr = float(spread_shock.quantile(0.8)) if len(spread_shock) else 0.0
                                            dyn_thr = float(th) + float(exp_cfg.get("spread_shock_threshold_bump", 0.03)) * (
                                                spread_shock > shock_thr
                                            ).astype(float)
                                            gate_df["dynamic_threshold"] = dyn_thr
                                            gate = fns.apply_trade_gating(
                                                gate_df,
                                                p_col="p",
                                                ev_col="ev",
                                                min_ev=float(exp_cfg.get("min_ev", 0.0)),
                                                base_p_threshold=float(th),
                                                dynamic_threshold_col="dynamic_threshold",
                                            )
                                            backtest_signals = backtest_signals[
                                                gate.reindex(backtest_signals.index).fillna(0).astype(int) > 0
                                            ]
                                        if backtest_signals.empty:
                                            continue
                                        bt_raw = fns.backtest_from_signals(df_ltf, backtest_signals, sym_cfg.get("cost", {}))
                                        if isinstance(bt_raw, tuple) and len(bt_raw) >= 2:
                                            trade_log, equity_curve = bt_raw[0], bt_raw[1]
                                        else:
                                            trade_log = (
                                                bt_raw.get("trade_log", pd.DataFrame()) if isinstance(bt_raw, dict) else pd.DataFrame()
                                            )
                                            equity_curve = (
                                                bt_raw.get("equity_curve", pd.Series(dtype=float))
                                                if isinstance(bt_raw, dict)
                                                else pd.Series(dtype=float)
                                            )
                                        tm = _trade_metrics(
                                            _to_dataframe(trade_log),
                                            equity_curve,
                                            df_ltf.index,
                                            settings=exp_cfg,
                                            signals=backtest_signals,
                                        )
                                        score = _score_objective(tm, no_trade_obj)
                                        feasible = (int(tm["trades"]) >= min_nt_trades) and (float(tm["max_dd"]) <= max_nt_dd)
                                        if best_run is None:
                                            best_run = {
                                                "tm": tm,
                                                "score": score,
                                                "threshold": float(th),
                                                "trade_log": trade_log,
                                                "equity_curve": equity_curve,
                                                "signals": backtest_signals,
                                                "feasible": feasible,
                                            }
                                            continue
                                        if feasible and not best_run["feasible"]:
                                            best_run = {
                                                "tm": tm,
                                                "score": score,
                                                "threshold": float(th),
                                                "trade_log": trade_log,
                                                "equity_curve": equity_curve,
                                                "signals": backtest_signals,
                                                "feasible": feasible,
                                            }
                                            continue
                                        if feasible == best_run["feasible"] and score > float(best_run["score"]):
                                            best_run = {
                                                "tm": tm,
                                                "score": score,
                                                "threshold": float(th),
                                                "trade_log": trade_log,
                                                "equity_curve": equity_curve,
                                                "signals": backtest_signals,
                                                "feasible": feasible,
                                            }

                                    if best_run is not None:
                                        tm = best_run["tm"]
                                        trade_log_df = _to_dataframe(best_run["trade_log"])
                                        sess_break = _session_breakdown(trade_log_df, tz_name, sym_cfg.get("sessions", {}))
                                        regime_break = _regime_breakdown(trade_log_df, merged["regime"])
                                        obj_score = _score_objective(tm, objective_cfg)
                                        kill_switch = (
                                            (float(tm.get("expectancy_r", 0.0)) <= float(risk_ctrl.get("kill_switch_expectancy_max", 0.0)))
                                            and (float(tm.get("sharpe_trade", 0.0)) <= float(risk_ctrl.get("kill_switch_sharpe_max", 0.0)))
                                            and (
                                                float(tm.get("window_pass_rate", 0.0))
                                                <= float(risk_ctrl.get("kill_switch_window_pass_max", 0.35))
                                            )
                                        )
                                        deploy_eligible = (
                                            (not kill_switch)
                                            and (int(tm["trades"]) >= min_trades)
                                            and (
                                                float(tm.get("window_pass_rate", 0.0))
                                                >= float(exp_cfg.get("min_window_pass_rate", 0.40))
                                            )
                                        )
                                        hf = _pair_prefs(symbol, sym_cfg).get("hard_filters", {})
                                        deploy_eligible = deploy_eligible and (
                                            float(tm.get("expectancy_r", 0.0)) >= float(hf.get("expectancy_min", 0.0))
                                        )
                                        deploy_eligible = deploy_eligible and (
                                            float(tm.get("net_expectancy_after_cost", 0.0))
                                            >= float(hf.get("net_expectancy_min", 0.0))
                                        )
                                        deploy_eligible = deploy_eligible and (
                                            float(tm.get("window_pass_rate", 0.0)) >= float(hf.get("window_pass_min", 0.0))
                                        )

                                        row = {
                                            "symbol": symbol,
                                            "ltf_timeframe": str(ltf_tf),
                                            "strategy": "hybrid_regime_router",
                                            "style": "hybrid",
                                            "hybrid_router": True,
                                            "regime_variant": variant.get("id", "v0"),
                                            "regime_strategy_map_json": json.dumps(_safe_jsonable(regime_map), separators=(",", ":")),
                                            "regime_router_perf_json": json.dumps(_safe_jsonable(regime_router_perf), separators=(",", ":")),
                                            "allowed_regimes": ",".join(REGIME_NAMES),
                                            "objective_score": round(float(obj_score), 6),
                                            "expectancy_r": round(float(tm["expectancy_r"]), 6),
                                            "net_expectancy_after_cost": round(float(tm["net_expectancy_after_cost"]), 6),
                                            "profit_factor": round(float(tm["profit_factor"]), 6),
                                            "win_rate": round(float(tm["win_rate"]), 6),
                                            "sharpe_trade": round(float(tm["sharpe_trade"]), 6),
                                            "max_dd": round(float(tm["max_dd"]), 6),
                                            "rolling_3m_expectancy": round(float(tm["rolling_3m_expectancy"]), 6),
                                            "rolling_3m_sharpe": round(float(tm["rolling_3m_sharpe"]), 6),
                                            "worst_3m_drawdown": round(float(tm["worst_3m_drawdown"]), 6),
                                            "window_pass_rate": round(float(tm["window_pass_rate"]), 6),
                                            "trades": int(tm["trades"]),
                                            "trades_per_month": round(float(tm["trades_per_month"]), 4),
                                            "trades_per_day": round(float(tm["trades_per_day"]), 4),
                                            "data_quality_missing_ratio": round(float(dq.get("missing_ratio", 0.0)), 6),
                                            "data_quality_max_gap_bars": round(float(dq.get("max_gap_bars", 0.0)), 4),
                                            "data_quality_warn": bool(dq.get("warn", 0.0) > 0),
                                            "cost_per_trade": round(float(tm["cost_per_trade"]), 6),
                                            "financing_per_trade": round(float(tm["financing_per_trade"]), 6),
                                            "cost_to_gross_pnl_ratio": round(float(tm["cost_to_gross_pnl_ratio"]), 6),
                                            "avg_hold_hours": round(float(tm["avg_hold_hours"]), 4),
                                            "overnight_trade_pct": round(float(tm["overnight_trade_pct"]), 6),
                                            "signal_flip_rate": round(float(tm["signal_flip_rate"]), 6),
                                            "long_expectancy_r": round(float(tm["long_expectancy_r"]), 6),
                                            "short_expectancy_r": round(float(tm["short_expectancy_r"]), 6),
                                            "long_win_rate": round(float(tm["long_win_rate"]), 6),
                                            "short_win_rate": round(float(tm["short_win_rate"]), 6),
                                            "brier_score": round(float(calib.get("brier_score", float("nan"))), 6)
                                            if pd.notna(calib.get("brier_score", np.nan))
                                            else np.nan,
                                            "ece": round(float(calib.get("ece", float("nan"))), 6)
                                            if pd.notna(calib.get("ece", np.nan))
                                            else np.nan,
                                            "calibrator_type": calibrator_type,
                                            "fold_diagnostics_json": json.dumps(_safe_jsonable(fold_diagnostics), separators=(",", ":")),
                                            "feature_transform_stats_json": json.dumps(
                                                _safe_jsonable(prep_meta.get("feature_transform_stats", {})), separators=(",", ":")
                                            ),
                                            "calibrated_no_trade_threshold": float(best_run["threshold"]),
                                            "session_breakdown_json": json.dumps(_safe_jsonable(sess_break), separators=(",", ":")),
                                            "regime_breakdown_json": json.dumps(_safe_jsonable(regime_break), separators=(",", ":")),
                                            "no_trade_threshold": float(
                                                sym_cfg.get("no_trade", {}).get("probability_threshold", 0.58)
                                            ),
                                            "kill_switch": bool(kill_switch),
                                            "deploy_eligible": bool(deploy_eligible),
                                            "deployment_role": "",
                                            "model_selected": False,
                                            "artifact_model_path": "",
                                            "artifact_meta_path": "",
                                            "artifact_trade_log_path": "",
                                        }
                                        rows.append(row)
                                        model_candidates.append(
                                            {
                                                "row": row,
                                                "models": wf_models,
                                                "wf_metrics": wf_metrics,
                                                "feature_cols": list(X.columns),
                                                "label_meta": label_meta,
                                                "regime_variant": variant,
                                                "sym_cfg": sym_cfg,
                                                "session_filter": _pair_prefs(symbol, sym_cfg).get("preferred_sessions", []),
                                                "ltf_timeframe": str(ltf_tf),
                                                "regime_strategy_map": regime_map,
                                                "calibrator": calibrator,
                                                "trade_log": best_run.get("trade_log"),
                                            }
                                        )

    if not rows:
        pd.DataFrame(columns=_ranking_headers()).to_csv(ranking_csv, index=False)
        return {"ranking_csv": str(ranking_csv), "rows": 0, "selected": 0}

    rank_df = pd.DataFrame(rows)
    rank_df = rank_df.sort_values(
        by=["symbol", "objective_score", "expectancy_r", "profit_factor", "win_rate", "trades"],
        ascending=[True, False, False, False, False, False],
    )

    selected = []
    for symbol, g in rank_df.groupby("symbol", sort=False):
        g_ok = g[
            (g["trades"] >= min_trades)
            & (g["deploy_eligible"] == True)
            & (g["kill_switch"] == False)
            & (g["window_pass_rate"] >= float(exp_cfg.get("min_window_pass_rate", 0.40)))
        ]
        if g_ok.empty:
            continue
        chosen: List[int] = []
        if hybrid_min_per_symbol > 0:
            g_h = g_ok[g_ok["hybrid_router"] == True].sort_values(
                by=["objective_score", "expectancy_r", "profit_factor", "win_rate", "trades"],
                ascending=[False, False, False, False, False],
            )
            chosen.extend(g_h.head(hybrid_min_per_symbol).index.tolist())

        remain = max(0, top_k - len(chosen))
        if remain > 0:
            g_rest = g_ok.drop(index=chosen, errors="ignore")
            chosen.extend(g_rest.head(remain).index.tolist())
        selected.extend(chosen)

    selected_set = set(selected)
    deploy_cfg = cfg.get("deployment_policy", {})
    champion_n = max(
        0,
        int(deploy_cfg.get("champion_per_symbol", deploy_cfg.get("champion_count", 1))),
    )
    challenger_n = max(
        0,
        int(deploy_cfg.get("challenger_per_symbol", deploy_cfg.get("challenger_count", 1))),
    )
    for symbol, g in rank_df.groupby("symbol", sort=False):
        g_sel = g[g.index.isin(selected_set)].sort_values(
            by=["objective_score", "expectancy_r", "profit_factor", "win_rate", "trades"],
            ascending=[False, False, False, False, False],
        )
        if g_sel.empty:
            continue
        champ_idx = g_sel.head(champion_n).index.tolist()
        rank_df.loc[champ_idx, "deployment_role"] = "champion"
        if challenger_n > 0:
            rest = g_sel.drop(index=champ_idx, errors="ignore")
            chall_idx = rest.head(challenger_n).index.tolist()
            rank_df.loc[chall_idx, "deployment_role"] = "challenger"

    model_dir = Path(out_cfg.get("model_dir", "models/research/regime_strategy"))
    save_trade_logs = bool(out_cfg.get("save_selected_trade_logs", True))

    for cand in model_candidates:
        r = cand["row"]
        key_idx = rank_df.index[
            (rank_df["symbol"] == r["symbol"])
            & (rank_df["ltf_timeframe"] == r["ltf_timeframe"])
            & (rank_df["strategy"] == r["strategy"])
            & (rank_df["regime_variant"] == r["regime_variant"])
            & (rank_df["objective_score"] == r["objective_score"])
            & (rank_df["trades"] == r["trades"])
        ]
        if len(key_idx) == 0:
            continue
        idx = int(key_idx[0])
        if idx not in selected_set:
            continue

        base_name = (
            f"{r['symbol']}_{r['strategy']}_{r['regime_variant']}_"
            f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        meta = {
            "symbol": r["symbol"],
            "ltf_timeframe": r["ltf_timeframe"],
            "strategy": r["strategy"],
            "style": r["style"],
            "regime_rules": cand["regime_variant"],
            "session_filter": cand["session_filter"],
            "feature_list": cand["feature_cols"],
            "barrier_params": cand.get("sym_cfg", {}).get("barrier", cfg.get("barrier", {})),
            "model_cfg": cand.get("sym_cfg", {}).get("model", cfg.get("model", {})),
            "splits_cfg": cand.get("sym_cfg", {}).get("splits", cfg.get("splits", {})),
            "cost_cfg": cand.get("sym_cfg", {}).get("cost", cfg.get("cost", {})),
            "no_trade": cand.get("sym_cfg", {}).get("no_trade", cfg.get("no_trade", {})),
            "entry_refinement": cfg.get("entry_refinement", {}),
            "risk_controls": cfg.get("risk_controls", {}),
            "deployment_role": str(rank_df.loc[idx, "deployment_role"] or ""),
            "ranking_metrics": {k: _safe_jsonable(v) for k, v in r.items()},
            "walkforward_metrics": _safe_jsonable(cand["wf_metrics"]),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        artifacts = _save_model_artifact(cand["models"], meta, model_dir / r["symbol"], base_name)
        trade_log_path = ""
        if save_trade_logs:
            trade_log_path = _save_trade_log_artifact(cand.get("trade_log"), model_dir / r["symbol"], base_name)
            if trade_log_path:
                meta["trade_log_path"] = trade_log_path
                try:
                    with Path(artifacts.get("meta_path", "")).open("w", encoding="utf-8") as f:
                        json.dump(_safe_jsonable(meta), f, indent=2)
                except Exception:
                    pass
        rank_df.loc[idx, "model_selected"] = True
        rank_df.loc[idx, "artifact_model_path"] = artifacts.get("model_path", "")
        rank_df.loc[idx, "artifact_meta_path"] = artifacts.get("meta_path", "")
        rank_df.loc[idx, "artifact_trade_log_path"] = trade_log_path

    rank_df.to_csv(ranking_csv, index=False)

    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ranking_csv": str(ranking_csv),
        "rows": int(len(rank_df)),
        "selected_models": int((rank_df["model_selected"] == True).sum()),
        "symbols": sorted(rank_df["symbol"].unique().tolist()),
        "entry_timeframes": sorted(rank_df["ltf_timeframe"].astype(str).unique().tolist()),
        "kill_switched_rows": int((rank_df["kill_switch"] == True).sum()),
        "deploy_eligible_rows": int((rank_df["deploy_eligible"] == True).sum()),
        "champion_rows": int((rank_df["deployment_role"] == "champion").sum()),
        "challenger_rows": int((rank_df["deployment_role"] == "challenger").sum()),
        "top_k": int(top_k),
        "min_trades": int(min_trades),
    }
    manifest_path = Path(out_cfg.get("manifest_json", "data/research/regime_strategy_manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "ranking_csv": str(ranking_csv),
        "manifest_json": str(manifest_path),
        "rows": int(len(rank_df)),
        "selected": int((rank_df["model_selected"] == True).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime->strategy routing research (D1/H1 regimes, M15/M5 entries).")
    p.add_argument("--pipeline-module", default="", help="Python module containing pipeline functions.")
    p.add_argument("--config", default="", help="Path to JSON/YAML config override.")
    p.add_argument("--template-only", action="store_true", help="Write ranking CSV headers only and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config or None)
    if args.template_only:
        res = run_research(cfg, None, template_only=True)
    else:
        if not args.pipeline_module:
            raise SystemExit("--pipeline-module is required unless --template-only is used.")
        fns = _load_pipeline(args.pipeline_module)
        res = run_research(cfg, fns, template_only=False)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
