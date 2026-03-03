from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


SESSION_NAMES = [
    "asia",
    "pre_london",
    "london_open",
    "london_continuation",
    "ny_open",
    "ny_overlap_postdata",
]


def _require_columns(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {ctx}: {miss}")


def _safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-9) -> pd.Series:
    return num / (den.abs() + eps)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = pd.to_numeric(df["close"], errors="coerce").shift(1)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _context_mode(config: Dict[str, object] | None) -> str:
    mode = str((config or {}).get("temporal_context_mode", "hybrid")).strip().lower()
    if mode not in {"engineered", "sequence", "hybrid"}:
        raise ValueError("temporal_context_mode must be one of: engineered, sequence, hybrid.")
    return mode


def _to_int_list(v: object, default: List[int]) -> List[int]:
    if v is None:
        return list(default)
    out: List[int] = []
    for x in list(v):  # type: ignore[arg-type]
        n = int(x)
        if n > 0:
            out.append(n)
    if not out:
        return list(default)
    return sorted(set(out))


def _short_horizon_context(df: pd.DataFrame, atr14: pd.Series, config: Dict[str, object] | None = None) -> pd.DataFrame:
    cfg = config or {}
    lookbacks = _to_int_list(cfg.get("short_context_lookbacks"), [8, 16, 32, 64])
    lag_bars = _to_int_list(cfg.get("short_context_lag_bars"), [1, 2, 3, 4, 8])

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    eps = 1e-9
    ret1 = c.pct_change()
    bar_range = (h - l).abs()
    body = c - o
    ret_atr = _safe_div(c.diff(), atr14)
    body_atr = _safe_div(body, atr14)
    range_atr = _safe_div(bar_range, atr14)

    out = pd.DataFrame(index=df.index)
    for lg in lag_bars:
        out[f"sf_ctx_ret_atr_lag_{lg}"] = ret_atr.shift(lg)
        out[f"sf_ctx_body_atr_lag_{lg}"] = body_atr.shift(lg)
        out[f"sf_ctx_range_atr_lag_{lg}"] = range_atr.shift(lg)
        out[f"sf_ctx_ret1_lag_{lg}"] = ret1.shift(lg)

    sign1 = np.sign(c.diff())
    for n in lookbacks:
        n = int(n)
        net = c - c.shift(n)
        gross = c.diff().abs().rolling(n, min_periods=n).sum()
        out[f"sf_ctx_net_move_atr_{n}"] = _safe_div(net, atr14)
        out[f"sf_ctx_eff_ratio_{n}"] = (net.abs() / (gross + eps))
        out[f"sf_ctx_momo_persist_{n}"] = sign1.rolling(n, min_periods=n).mean().abs()
        flip = sign1.ne(sign1.shift(1)).astype(float)
        out[f"sf_ctx_reversal_flip_rate_{n}"] = flip.rolling(n, min_periods=n).mean()

        prev_high = h.shift(1).rolling(n, min_periods=n).max()
        prev_low = l.shift(1).rolling(n, min_periods=n).min()
        took_high = (h > prev_high).astype(float)
        took_low = (l < prev_low).astype(float)
        reject_high = took_high * (c < prev_high).astype(float)
        reject_low = took_low * (c > prev_low).astype(float)
        out[f"sf_ctx_break_accept_up_count_{n}"] = (c > prev_high).astype(float).rolling(n, min_periods=n).sum()
        out[f"sf_ctx_break_accept_dn_count_{n}"] = (c < prev_low).astype(float).rolling(n, min_periods=n).sum()
        out[f"sf_ctx_sweep_reject_high_count_{n}"] = reject_high.rolling(n, min_periods=n).sum()
        out[f"sf_ctx_sweep_reject_low_count_{n}"] = reject_low.rolling(n, min_periods=n).sum()

        tr = _true_range(df)
        atr_short = tr.rolling(max(4, n // 2), min_periods=max(4, n // 2)).mean()
        atr_long = tr.rolling(n, min_periods=n).mean()
        out[f"sf_ctx_comp_exp_ratio_{n}"] = _safe_div(atr_short, atr_long)
    return out


def _slow_regime_context(df: pd.DataFrame, atr14: pd.Series, config: Dict[str, object] | None = None) -> pd.DataFrame:
    cfg = config or {}
    slow_days = _to_int_list(cfg.get("slow_regime_days"), [10, 20, 40])  # ~2w,4w,8w
    bars_per_day = int(cfg.get("bars_per_day", 96))
    c = pd.to_numeric(df["close"], errors="coerce")
    spread = pd.to_numeric(df.get("sf_spread_proxy", pd.Series(index=df.index, data=np.nan)), errors="coerce")
    ret1 = c.pct_change()
    tr = _true_range(df)

    out = pd.DataFrame(index=df.index)
    for d in slow_days:
        n = int(d) * int(bars_per_day)
        n = max(n, 96)
        atr_med = atr14.rolling(n, min_periods=max(96, n // 4)).median()
        atr_q75 = atr14.rolling(n, min_periods=max(96, n // 4)).quantile(0.75)
        atr_q25 = atr14.rolling(n, min_periods=max(96, n // 4)).quantile(0.25)
        out[f"sf_regime_vol_med_ratio_{d}d"] = _safe_div(atr14, atr_med)
        out[f"sf_regime_vol_iqr_norm_{d}d"] = _safe_div((atr14 - atr_med), (atr_q75 - atr_q25))

        range_s = (tr.rolling(n, min_periods=max(96, n // 4)).mean())
        out[f"sf_regime_spread_range_ratio_{d}d"] = _safe_div(spread, range_s)
        eff = (c.diff(max(8, n // 8)).abs()) / (c.diff().abs().rolling(max(8, n // 8), min_periods=max(8, n // 8)).sum() + 1e-9)
        out[f"sf_regime_trend_eff_mean_{d}d"] = eff.rolling(max(16, n // 8), min_periods=max(16, n // 16)).mean()
        out[f"sf_regime_ret_vol_{d}d"] = ret1.rolling(n, min_periods=max(96, n // 4)).std(ddof=0)

        event = pd.to_numeric(df.get("event_window_flag", pd.Series(index=df.index, data=0.0)), errors="coerce").fillna(0.0)
        out[f"sf_regime_event_intensity_{d}d"] = event.rolling(n, min_periods=max(96, n // 4)).mean()
    return out


def _bounded_sequence_context(df: pd.DataFrame, atr14: pd.Series, config: Dict[str, object] | None = None) -> pd.DataFrame:
    cfg = config or {}
    seq_len = int(cfg.get("sequence_lookback_bars", 128))
    seq_stride = int(cfg.get("sequence_stride_bars", 8))
    seq_len = max(8, seq_len)
    seq_stride = max(1, seq_stride)
    seq_lags = list(range(1, seq_len + 1, seq_stride))

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    bar_range = (h - l).abs()
    body = c - o
    ret_atr = _safe_div(c.diff(), atr14)
    body_atr = _safe_div(body, atr14)
    range_atr = _safe_div(bar_range, atr14)

    out = pd.DataFrame(index=df.index)
    for lag in seq_lags:
        out[f"sf_seq_ret_atr_l{lag:03d}"] = ret_atr.shift(lag)
        out[f"sf_seq_body_atr_l{lag:03d}"] = body_atr.shift(lag)
        out[f"sf_seq_range_atr_l{lag:03d}"] = range_atr.shift(lag)
    return out


def _session_structural_context(df: pd.DataFrame, config: Dict[str, object] | None = None) -> pd.DataFrame:
    cfg = config or {}
    bars_1d = int(cfg.get("bars_per_day", 96))
    lookbacks = _to_int_list(cfg.get("session_context_lookbacks_bars"), [bars_1d, 3 * bars_1d, 5 * bars_1d])

    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    prior_high = pd.to_numeric(df["prior_session_high"], errors="coerce")
    prior_low = pd.to_numeric(df["prior_session_low"], errors="coerce")
    sess_open = pd.to_numeric(df["current_session_open"], errors="coerce")
    sess_range = pd.to_numeric(df["current_session_developing_range"], errors="coerce")

    broke_high = (h > prior_high).astype(float)
    broke_low = (l < prior_low).astype(float)
    fail_high = broke_high * (c <= prior_high).astype(float)
    fail_low = broke_low * (c >= prior_low).astype(float)
    cont_high = broke_high * (c > prior_high).astype(float)
    cont_low = broke_low * (c < prior_low).astype(float)
    disp_from_open = c - sess_open

    out = pd.DataFrame(index=df.index)
    for n in lookbacks:
        n = int(max(8, n))
        out[f"ss_ctx_repeat_fail_high_{n}"] = fail_high.rolling(n, min_periods=max(8, n // 4)).sum()
        out[f"ss_ctx_repeat_fail_low_{n}"] = fail_low.rolling(n, min_periods=max(8, n // 4)).sum()
        out[f"ss_ctx_repeat_cont_high_{n}"] = cont_high.rolling(n, min_periods=max(8, n // 4)).sum()
        out[f"ss_ctx_repeat_cont_low_{n}"] = cont_low.rolling(n, min_periods=max(8, n // 4)).sum()
        out[f"ss_ctx_session_open_disp_mean_{n}"] = disp_from_open.abs().rolling(n, min_periods=max(8, n // 4)).mean()
        out[f"ss_ctx_dev_range_mean_{n}"] = sess_range.rolling(n, min_periods=max(8, n // 4)).mean()

    prev_session = df["session_bucket"].astype(str).shift(1)
    cur_session = df["session_bucket"].astype(str)
    asia_to_london = (
        (prev_session.isin(["asia", "pre_london"]))
        & (cur_session.isin(["london_open", "london_continuation"]))
    ).astype(float)
    london_to_ny = (
        (prev_session.isin(["london_open", "london_continuation"]))
        & (cur_session.isin(["ny_open", "ny_overlap_postdata"]))
    ).astype(float)
    out["ss_ctx_asia_to_london_state"] = asia_to_london * np.sign(disp_from_open).fillna(0.0)
    out["ss_ctx_london_to_ny_state"] = london_to_ny * np.sign(disp_from_open).fillna(0.0)
    return out


def build_shared_features(df: pd.DataFrame, config: Dict[str, object] | None = None) -> pd.DataFrame:
    """Build shared, leakage-safe features using backward-only calculations."""

    _require_columns(
        df,
        [
            "open",
            "high",
            "low",
            "close",
            "prior_day_high",
            "prior_day_low",
            "prior_session_high",
            "prior_session_low",
        ],
        "build_shared_features",
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input index must be DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("Input index must be timezone-aware.")

    x = df.copy()
    o = pd.to_numeric(x["open"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")

    bar_range = (h - l).abs()
    body = (c - o).abs()
    upper_wick = (h - np.maximum(o, c)).clip(lower=0.0)
    lower_wick = (np.minimum(o, c) - l).clip(lower=0.0)

    tr = _true_range(x)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr48 = tr.rolling(48, min_periods=48).mean()
    atr96 = tr.rolling(96, min_periods=96).mean()

    ret1 = c.pct_change()
    gross_move_8 = c.diff().abs().rolling(8, min_periods=8).sum()
    net_move_8 = c.diff(8).abs()
    gross_move_16 = c.diff().abs().rolling(16, min_periods=16).sum()
    net_move_16 = c.diff(16).abs()

    roll_anchor_12 = c.shift(1).rolling(12, min_periods=12).mean()
    above_anchor = (c > roll_anchor_12).astype(float)

    prior_day_high = pd.to_numeric(x["prior_day_high"], errors="coerce")
    prior_day_low = pd.to_numeric(x["prior_day_low"], errors="coerce")
    prior_sess_high = pd.to_numeric(x["prior_session_high"], errors="coerce")
    prior_sess_low = pd.to_numeric(x["prior_session_low"], errors="coerce")

    spread_proxy = (
        pd.to_numeric(x.get("spread_proxy_bps", pd.Series(index=x.index, dtype=float)), errors="coerce")
        if "spread_proxy_bps" in x.columns
        else pd.to_numeric(x.get("spread", pd.Series(index=x.index, dtype=float)), errors="coerce")
    )

    out = pd.DataFrame(index=x.index)
    out["sf_realized_range"] = bar_range
    out["sf_vol_scale_atr14"] = atr14
    out["sf_candle_body"] = body
    out["sf_upper_wick"] = upper_wick
    out["sf_lower_wick"] = lower_wick
    out["sf_body_to_range_ratio"] = _safe_div(body, bar_range)
    out["sf_wick_asymmetry"] = _safe_div((upper_wick - lower_wick), bar_range)
    out["sf_compression_ratio_14_48"] = _safe_div(atr14, atr48)
    out["sf_expansion_ratio_14_96"] = _safe_div(atr14, atr96)
    out["sf_directional_efficiency_8"] = _safe_div(net_move_8, gross_move_8)
    out["sf_directional_efficiency_16"] = _safe_div(net_move_16, gross_move_16)
    out["sf_net_move_8"] = net_move_8
    out["sf_gross_move_8"] = gross_move_8
    out["sf_short_persistence_above_anchor12"] = above_anchor.shift(1).rolling(12, min_periods=12).mean()
    out["sf_dist_prior_day_high_atr"] = _safe_div(c - prior_day_high, atr14)
    out["sf_dist_prior_day_low_atr"] = _safe_div(c - prior_day_low, atr14)
    out["sf_dist_prior_session_high_atr"] = _safe_div(c - prior_sess_high, atr14)
    out["sf_dist_prior_session_low_atr"] = _safe_div(c - prior_sess_low, atr14)
    out["sf_spread_proxy"] = spread_proxy
    out["sf_spread_range_ratio"] = _safe_div(spread_proxy, bar_range)
    out["sf_ret1"] = ret1

    mode = _context_mode(config)
    if mode in {"engineered", "hybrid"}:
        out = out.join(_short_horizon_context(x, atr14, config))
        out = out.join(_slow_regime_context(out.join(x), atr14, config))
    if mode in {"sequence", "hybrid"}:
        out = out.join(_bounded_sequence_context(x, atr14, config))
    return out.replace([np.inf, -np.inf], np.nan)


def _build_base_session_features(df: pd.DataFrame, config: Dict[str, object] | None = None) -> pd.DataFrame:
    _require_columns(
        df,
        [
            "session_id",
            "session_bucket",
            "close",
            "high",
            "low",
            "current_session_open",
            "current_session_developing_range",
            "current_close_position_in_session_range",
            "prior_session_high",
            "prior_session_low",
        ],
        "build_session_features",
    )

    x = df.copy()
    c = pd.to_numeric(x["close"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    sess_open = pd.to_numeric(x["current_session_open"], errors="coerce")
    sess_rng = pd.to_numeric(x["current_session_developing_range"], errors="coerce")
    prior_high = pd.to_numeric(x["prior_session_high"], errors="coerce")
    prior_low = pd.to_numeric(x["prior_session_low"], errors="coerce")

    out = pd.DataFrame(index=x.index)
    out["ss_dist_from_session_open"] = (c - sess_open).abs()
    out["ss_signed_displacement_from_session_open"] = c - sess_open
    out["ss_current_dev_session_range_width"] = sess_rng
    out["ss_position_in_dev_session_range"] = pd.to_numeric(
        x["current_close_position_in_session_range"], errors="coerce"
    )
    out["ss_breakout_depth_above_prior_session_high"] = (h - prior_high).clip(lower=0.0)
    out["ss_breakout_depth_below_prior_session_low"] = (prior_low - l).clip(lower=0.0)

    breakout_accept_up = (c > prior_high).astype(int)
    breakout_accept_dn = (c < prior_low).astype(int)
    sid = pd.to_numeric(x["session_id"], errors="coerce").astype("Int64")
    out["ss_breakout_accept_count_up"] = breakout_accept_up.groupby(sid).cumsum()
    out["ss_breakout_accept_count_dn"] = breakout_accept_dn.groupby(sid).cumsum()

    prev_high_8 = h.shift(1).rolling(8, min_periods=8).max()
    prev_low_8 = l.shift(1).rolling(8, min_periods=8).min()
    took_prev_high = (h > prev_high_8).astype(int)
    took_prev_low = (l < prev_low_8).astype(int)
    close_back_inside_high = (c < prev_high_8).astype(int)
    close_back_inside_low = (c > prev_low_8).astype(int)
    out["ss_sweep_reject_high_8"] = took_prev_high * close_back_inside_high
    out["ss_sweep_reject_low_8"] = took_prev_low * close_back_inside_low

    prev_session = x["session_bucket"].astype(str).shift(1)
    cur_session = x["session_bucket"].astype(str)
    out["ss_transition_asia_to_london"] = (
        (prev_session.isin(["asia", "pre_london"]))
        & (cur_session.isin(["london_open", "london_continuation"]))
    ).astype(int)
    out["ss_transition_london_to_ny"] = (
        (prev_session.isin(["london_open", "london_continuation"]))
        & (cur_session.isin(["ny_open", "ny_overlap_postdata"]))
    ).astype(int)
    mode = _context_mode(config)
    if mode in {"engineered", "hybrid"}:
        out = out.join(_session_structural_context(x, config))
    return out


def build_session_features(df: pd.DataFrame, config: Dict[str, object] | None = None) -> Dict[str, pd.DataFrame]:
    """Build session-routed feature blocks; each sleeve only populated on its rows."""

    base = _build_base_session_features(df, config=config)
    out: Dict[str, pd.DataFrame] = {}
    s = df["session_bucket"].astype(str)
    for name in SESSION_NAMES:
        mask = s.eq(name)
        block = base.where(mask, np.nan)
        out[name] = block
    return out


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic interaction features for later routing/gating."""

    _require_columns(
        df,
        [
            "session_bucket",
            "sf_dist_prior_day_high_atr",
            "sf_dist_prior_day_low_atr",
            "sf_compression_ratio_14_48",
            "sf_spread_proxy",
            "sf_vol_scale_atr14",
            "sf_directional_efficiency_8",
            "ss_breakout_accept_count_up",
            "ss_breakout_accept_count_dn",
            "ss_sweep_reject_high_8",
            "ss_sweep_reject_low_8",
            "ss_signed_displacement_from_session_open",
        ],
        "build_interaction_features",
    )

    x = df.copy()
    s = x["session_bucket"].astype(str)
    event_state = pd.to_numeric(
        x.get("event_window_flag", pd.Series(index=x.index, data=0.0)),
        errors="coerce",
    ).fillna(0.0)

    out = pd.DataFrame(index=x.index)
    out["if_session_london_x_struct_dist"] = (
        s.isin(["london_open", "london_continuation"]).astype(float)
        * (pd.to_numeric(x["sf_dist_prior_day_high_atr"], errors="coerce").abs())
    )
    out["if_session_ny_x_compression"] = s.isin(["ny_open", "ny_overlap_postdata"]).astype(float) * pd.to_numeric(
        x["sf_compression_ratio_14_48"], errors="coerce"
    )
    out["if_event_x_breakout_accept"] = event_state * (
        pd.to_numeric(x["ss_breakout_accept_count_up"], errors="coerce")
        + pd.to_numeric(x["ss_breakout_accept_count_dn"], errors="coerce")
    )
    out["if_spread_x_barrier_scale_proxy"] = pd.to_numeric(x["sf_spread_proxy"], errors="coerce") * pd.to_numeric(
        x["sf_vol_scale_atr14"], errors="coerce"
    )
    out["if_vol_x_sweep_proxy"] = pd.to_numeric(x["sf_vol_scale_atr14"], errors="coerce") * (
        pd.to_numeric(x["ss_sweep_reject_high_8"], errors="coerce")
        + pd.to_numeric(x["ss_sweep_reject_low_8"], errors="coerce")
    )
    out["if_trend_eff_x_session_open_dist"] = pd.to_numeric(
        x["sf_directional_efficiency_8"], errors="coerce"
    ) * pd.to_numeric(x["ss_signed_displacement_from_session_open"], errors="coerce").abs()
    out["if_session_asia_x_struct_dist_low"] = s.eq("asia").astype(float) * pd.to_numeric(
        x["sf_dist_prior_day_low_atr"], errors="coerce"
    ).abs()
    return out.replace([np.inf, -np.inf], np.nan)


def build_feature_registry(
    shared_df: pd.DataFrame,
    session_blocks: Dict[str, pd.DataFrame],
    interaction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build optional registry describing feature routing scope."""

    rows: List[Dict[str, str]] = []
    for c in shared_df.columns:
        rows.append(
            {
                "feature_name": c,
                "feature_block": "shared",
                "scope": "all_heads",
                "allowed_sessions": "all",
            }
        )
    for sess, block in session_blocks.items():
        for c in block.columns:
            rows.append(
                {
                    "feature_name": c,
                    "feature_block": "session_specific",
                    "scope": "specialist",
                    "allowed_sessions": sess,
                }
            )
    for c in interaction_df.columns:
        rows.append(
            {
                "feature_name": c,
                "feature_block": "interaction",
                "scope": "conditional",
                "allowed_sessions": "all",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(ignore_index=True)
