"""Compatibility adapter exposing the expected research pipeline function names.

Functions exported:
- load_ohlcv
- make_features
- make_labels
- walkforward_train_eval
- backtest_from_signals
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtesting.core.timeframe import Timeframe
from backtesting.data.manager import DataManager
from backtesting.data.warehouse import DataWarehouse


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out.sort_index()


def _tf_from_str(timeframe: str) -> Timeframe:
    return Timeframe.from_oanda_granularity(str(timeframe).upper())


def _load_direct_tf_file(symbol: str, timeframe: str) -> pd.DataFrame | None:
    base = Path("data/backtesting") / symbol
    pq = base / f"{timeframe.upper()}.parquet"
    csv = base / f"{timeframe.upper()}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, index_col=0, parse_dates=True)
    return None


def _save_direct_tf_file(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    base = Path("data/backtesting") / symbol
    base.mkdir(parents=True, exist_ok=True)
    pq = base / f"{timeframe.upper()}.parquet"
    try:
        df.to_parquet(pq, engine="pyarrow", compression="snappy", index=True)
    except Exception:
        df.to_csv(base / f"{timeframe.upper()}.csv", index=True)


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


def _parse_event_timestamps(events: Any) -> pd.DatetimeIndex:
    vals: List[pd.Timestamp] = []
    if events is None:
        return pd.DatetimeIndex([], tz="UTC")
    if isinstance(events, (str, pd.Timestamp)):
        events = [events]
    for e in events:
        ts = None
        if isinstance(e, dict):
            ts = e.get("ts") or e.get("time") or e.get("timestamp")
        else:
            ts = e
        if ts is None:
            continue
        try:
            t = pd.Timestamp(ts)
            if t.tz is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            vals.append(t)
        except Exception:
            continue
    if not vals:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(sorted(set(vals)), tz="UTC")


def _event_window_features(index: pd.DatetimeIndex, macro_cfg: Dict[str, Any]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    ev = _parse_event_timestamps(macro_cfg.get("events_utc", []))
    pre_min = int(macro_cfg.get("event_pre_minutes", 60))
    post_min = int(macro_cfg.get("event_post_minutes", 90))
    out = pd.DataFrame(index=idx)
    if len(ev) == 0:
        out["minutes_to_next_event"] = 1e6
        out["minutes_since_prev_event"] = 1e6
        out["event_window_flag"] = 0.0
        out["event_pre_window_flag"] = 0.0
        out["event_post_window_flag"] = 0.0
        return out
    next_pos = ev.searchsorted(idx, side="left")
    prev_pos = next_pos - 1

    next_ts = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
    prev_ts = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    m_next = next_pos < len(ev)
    m_prev = prev_pos >= 0
    if np.any(m_next):
        next_ts.iloc[np.where(m_next)[0]] = ev[next_pos[m_next]]
    if np.any(m_prev):
        prev_ts.iloc[np.where(m_prev)[0]] = ev[prev_pos[m_prev]]

    mins_to = ((next_ts - pd.Series(idx, index=idx)).dt.total_seconds() / 60.0).fillna(1e6)
    mins_since = ((pd.Series(idx, index=idx) - prev_ts).dt.total_seconds() / 60.0).fillna(1e6)
    pre_flag = ((mins_to >= 0.0) & (mins_to <= float(pre_min))).astype(float)
    post_flag = ((mins_since >= 0.0) & (mins_since <= float(post_min))).astype(float)

    out["minutes_to_next_event"] = mins_to
    out["minutes_since_prev_event"] = mins_since
    out["event_window_flag"] = ((pre_flag > 0) | (post_flag > 0)).astype(float)
    out["event_pre_window_flag"] = pre_flag
    out["event_post_window_flag"] = post_flag
    return out


def _rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    w = max(10, int(window))
    return s.rolling(w, min_periods=max(10, w // 2)).apply(
        lambda x: float((x <= x[-1]).mean()),
        raw=True,
    )


def _resample_ohlcv(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    out = x.resample(tf.to_pandas_freq()).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out


def _resample_ohlcv_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    out = x.resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out


def _asof_series_to_index(index_utc: pd.DatetimeIndex, s: pd.Series, allow_exact: bool = True) -> pd.Series:
    left = pd.DataFrame({"ts": pd.DatetimeIndex(index_utc).sort_values()})
    right = pd.DataFrame({"ts": pd.DatetimeIndex(s.index), "v": pd.to_numeric(s, errors="coerce")}).sort_values("ts")
    merged = pd.merge_asof(left, right, on="ts", direction="backward", allow_exact_matches=allow_exact)
    out = pd.Series(merged["v"].to_numpy(), index=merged["ts"])
    return out.reindex(index_utc)


def compute_asia_range_features(df_m15: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """Session-anchored Asia features using completed 00:00-07:45 London bars only."""
    x = _ensure_utc_index(df_m15)
    idx_local = pd.DatetimeIndex(x.index).tz_convert(tz)
    trade_date = idx_local.normalize()
    minute_of_day = (idx_local.hour * 60) + idx_local.minute

    atr14 = pd.to_numeric(x.get("atr14", _atr(x, 14)), errors="coerce")
    asia_mask = (minute_of_day >= 0) & (minute_of_day <= ((7 * 60) + 45))
    after_asia = minute_of_day >= (8 * 60)

    asia_high_by_day = x.loc[asia_mask, "high"].groupby(trade_date[asia_mask]).max()
    asia_low_by_day = x.loc[asia_mask, "low"].groupby(trade_date[asia_mask]).min()

    day_index = pd.Index(trade_date)
    asia_high = pd.Series(day_index.map(asia_high_by_day.to_dict()), index=x.index, dtype=float)
    asia_low = pd.Series(day_index.map(asia_low_by_day.to_dict()), index=x.index, dtype=float)
    asia_high = asia_high.where(after_asia, np.nan)
    asia_low = asia_low.where(after_asia, np.nan)

    asia_range = asia_high - asia_low
    asia_mid = 0.5 * (asia_high + asia_low)
    close = pd.to_numeric(x["close"], errors="coerce")

    out = pd.DataFrame(index=x.index)
    out["asia_high"] = asia_high
    out["asia_low"] = asia_low
    out["asia_range"] = asia_range
    out["asia_mid"] = asia_mid
    out["asia_range_atr"] = asia_range / (atr14 + 1e-9)
    out["dist_to_asia_high_atr"] = (close - asia_high) / (atr14 + 1e-9)
    out["dist_to_asia_low_atr"] = (close - asia_low) / (atr14 + 1e-9)
    out["dist_to_asia_mid_atr"] = (close - asia_mid) / (atr14 + 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_london_open_features(df_m15: pd.DataFrame, tz: str = "Europe/London") -> pd.DataFrame:
    """London-open anchored features (08:00 M15 bar), no lookahead, same-date forward fill only."""
    x = _ensure_utc_index(df_m15)
    idx_local = pd.DatetimeIndex(x.index).tz_convert(tz)
    trade_date = idx_local.normalize()
    minute_of_day = (idx_local.hour * 60) + idx_local.minute
    after_lo = minute_of_day >= (8 * 60)

    atr14 = pd.to_numeric(x.get("atr14", _atr(x, 14)), errors="coerce")
    asia = compute_asia_range_features(x, tz=tz)
    asia_high = pd.to_numeric(asia["asia_high"], errors="coerce")
    asia_low = pd.to_numeric(asia["asia_low"], errors="coerce")

    lo_mask = minute_of_day == (8 * 60)
    lo_open_by_day = x.loc[lo_mask, "open"].groupby(trade_date[lo_mask]).first()
    day_index = pd.Index(trade_date)
    lo_open_price = pd.Series(day_index.map(lo_open_by_day.to_dict()), index=x.index, dtype=float).where(after_lo, np.nan)

    high = pd.to_numeric(x["high"], errors="coerce")
    low = pd.to_numeric(x["low"], errors="coerce")
    close = pd.to_numeric(x["close"], errors="coerce")

    break_up = high > asia_high
    break_dn = low < asia_low
    lo_break = pd.Series(np.where(break_up, 1, np.where(break_dn, -1, 0)), index=x.index, dtype=float).where(after_lo, np.nan)

    out = pd.DataFrame(index=x.index)
    out["lo_open_price"] = lo_open_price
    out["dist_from_lo_open_atr"] = (close - lo_open_price) / (atr14 + 1e-9)
    out["lo_break_of_asia"] = lo_break
    out["lo_break_depth_up_atr"] = np.maximum(high - asia_high, 0.0) / (atr14 + 1e-9)
    out["lo_break_depth_dn_atr"] = np.maximum(asia_low - low, 0.0) / (atr14 + 1e-9)
    out.loc[~after_lo, ["lo_break_depth_up_atr", "lo_break_depth_dn_atr"]] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def compute_htf_distance_features(df_m15: pd.DataFrame, df_h1: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
    """HTF structural distance features using completed bars only."""
    x = _ensure_utc_index(df_m15)
    _ = _ensure_utc_index(df_h1)  # kept for signature compatibility and future extensions
    d1 = _ensure_utc_index(df_d1)

    atr14 = pd.to_numeric(x.get("atr14", _atr(x, 14)), errors="coerce")
    close = pd.to_numeric(x["close"], errors="coerce")

    prev_day_high_s = pd.to_numeric(d1["high"], errors="coerce").shift(1)
    prev_day_low_s = pd.to_numeric(d1["low"], errors="coerce").shift(1)
    daily_range = (pd.to_numeric(d1["high"], errors="coerce") - pd.to_numeric(d1["low"], errors="coerce")).shift(1)
    adr20_s = daily_range.rolling(20, min_periods=20).mean()

    prev_day_high = _asof_series_to_index(x.index, prev_day_high_s, allow_exact=True)
    prev_day_low = _asof_series_to_index(x.index, prev_day_low_s, allow_exact=True)
    adr20 = _asof_series_to_index(x.index, adr20_s, allow_exact=True)

    d1_week = pd.DataFrame(index=d1.index)
    d1_week["high"] = pd.to_numeric(d1["high"], errors="coerce")
    d1_week["low"] = pd.to_numeric(d1["low"], errors="coerce")
    week_key = pd.DatetimeIndex(d1_week.index).tz_convert("UTC").tz_localize(None).to_period("W-SUN")
    w_high = d1_week.groupby(week_key)["high"].max()
    w_low = d1_week.groupby(week_key)["low"].min()
    week_end = pd.DatetimeIndex(w_high.index.to_timestamp(how="end")).tz_localize("UTC")
    prev_week_high_s = pd.Series(w_high.shift(1).to_numpy(), index=week_end, dtype=float).dropna()
    prev_week_low_s = pd.Series(w_low.shift(1).to_numpy(), index=week_end, dtype=float).dropna()
    prev_week_high = _asof_series_to_index(x.index, prev_week_high_s, allow_exact=True)
    prev_week_low = _asof_series_to_index(x.index, prev_week_low_s, allow_exact=True)

    asia = compute_asia_range_features(x)
    asia_range = pd.to_numeric(asia["asia_range"], errors="coerce")

    out = pd.DataFrame(index=x.index)
    out["dist_prev_day_high_atr"] = (close - prev_day_high) / (atr14 + 1e-9)
    out["dist_prev_day_low_atr"] = (close - prev_day_low) / (atr14 + 1e-9)
    out["dist_prev_week_high_atr"] = (close - prev_week_high) / (atr14 + 1e-9)
    out["dist_prev_week_low_atr"] = (close - prev_week_low) / (atr14 + 1e-9)
    out["adr20"] = adr20
    out["asia_range_pct_of_adr"] = asia_range / (adr20 + 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def _winsorize_expanding_past(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_periods: int = 128,
    global_clip: Tuple[float, float] = (-50.0, 50.0),
) -> pd.DataFrame:
    """Past-only winsorization: thresholds computed from expanding history and shifted by 1 bar."""
    out = df.copy()
    lo_g, hi_g = float(global_clip[0]), float(global_clip[1])
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        lo = s.expanding(min_periods=min_periods).quantile(lower_q).shift(1)
        hi = s.expanding(min_periods=min_periods).quantile(upper_q).shift(1)
        mask = lo.notna() & hi.notna() & s.notna()
        if mask.any():
            s2 = s.copy()
            s2.loc[mask] = s.loc[mask].clip(lower=lo.loc[mask], upper=hi.loc[mask])
            s = s2
        out[col] = s.clip(lower=lo_g, upper=hi_g)
    return out


def compute_candle_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    o = pd.to_numeric(x["open"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")
    eps = 1e-9

    bar_range = (h - l).clip(lower=0.0)
    upper_wick_pct = (h - np.maximum(o, c)) / np.maximum(bar_range, eps)
    lower_wick_pct = (np.minimum(o, c) - l) / np.maximum(bar_range, eps)
    close_location = (c - l) / np.maximum(bar_range, eps)
    wick_imbalance = upper_wick_pct - lower_wick_pct
    body_sign = np.sign(c - o)
    body_direction_persist_3 = body_sign.shift(1).rolling(3, min_periods=3).mean()

    out = pd.DataFrame(index=x.index)
    out["bar_range"] = bar_range
    out["upper_wick_pct"] = upper_wick_pct.clip(0.0, 1.0)
    out["lower_wick_pct"] = lower_wick_pct.clip(0.0, 1.0)
    out["close_location"] = close_location.clip(0.0, 1.0)
    out["wick_imbalance"] = wick_imbalance.clip(-1.0, 1.0)
    out["body_direction_persist_3"] = body_direction_persist_3.clip(-1.0, 1.0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return _winsorize_expanding_past(out, lower_q=0.01, upper_q=0.99, min_periods=128, global_clip=(-20.0, 20.0))


def compute_trend_quality_features(df: pd.DataFrame, lookbacks: tuple[int, ...] = (8, 16, 32)) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")
    eps = 1e-9

    diff = c.diff()
    abs_diff = diff.abs()
    sign_diff = np.sign(diff)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out = pd.DataFrame(index=x.index)
    for n0 in lookbacks:
        n = int(n0)
        if n <= 1:
            continue
        eff_ratio = (c - c.shift(n)).abs() / (abs_diff.rolling(n, min_periods=n).sum() + eps)
        directional_consistency = sign_diff.rolling(n, min_periods=n).sum().abs() / float(n)
        rolling_high_n = h.shift(1).rolling(n, min_periods=n).max()
        rolling_low_n = l.shift(1).rolling(n, min_periods=n).min()
        choppiness = 100.0 * np.log10((tr.rolling(n, min_periods=n).sum() + eps) / np.maximum((rolling_high_n - rolling_low_n), eps)) / np.log10(float(n))

        out[f"eff_ratio_{n}"] = eff_ratio
        out[f"directional_consistency_{n}"] = directional_consistency
        out[f"choppiness_{n}"] = choppiness

    out = out.replace([np.inf, -np.inf], np.nan)
    for col in [c for c in out.columns if c.startswith("eff_ratio_") or c.startswith("directional_consistency_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(0.0, 1.0)
    for col in [c for c in out.columns if c.startswith("choppiness_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(0.0, 100.0)
    return _winsorize_expanding_past(out, lower_q=0.01, upper_q=0.99, min_periods=128, global_clip=(-20.0, 120.0))


def compute_distribution_shape_features(df: pd.DataFrame, lookbacks: tuple[int, ...] = (16, 32)) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    c = pd.to_numeric(x["close"], errors="coerce")
    ret1 = c.pct_change()

    out = pd.DataFrame(index=x.index)
    for n0 in lookbacks:
        n = int(n0)
        if n <= 2:
            continue
        out[f"ret_skew_{n}"] = ret1.rolling(n, min_periods=n).skew()
        out[f"ret_kurt_{n}"] = ret1.rolling(n, min_periods=n).kurt()
        out[f"downside_semivar_{n}"] = np.square(np.minimum(ret1, 0.0)).rolling(n, min_periods=n).mean()

    out = out.replace([np.inf, -np.inf], np.nan)
    for col in [c for c in out.columns if c.startswith("downside_semivar_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0)
    return _winsorize_expanding_past(out, lower_q=0.01, upper_q=0.99, min_periods=128, global_clip=(-10.0, 10.0))


def compute_sweep_features(df: pd.DataFrame, lookbacks: tuple[int, ...] = (4, 8, 16)) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    c = pd.to_numeric(x["close"], errors="coerce")
    out = pd.DataFrame(index=x.index)
    for n0 in lookbacks:
        n = int(n0)
        if n <= 1:
            continue
        prev_high_n = h.shift(1).rolling(n, min_periods=n).max()
        prev_low_n = l.shift(1).rolling(n, min_periods=n).min()
        took_prev_high_n = (h > prev_high_n).astype(float)
        took_prev_low_n = (l < prev_low_n).astype(float)
        close_back_inside_high_n = (c < prev_high_n).astype(float)
        close_back_inside_low_n = (c > prev_low_n).astype(float)
        out[f"took_prev_high_{n}"] = took_prev_high_n
        out[f"took_prev_low_{n}"] = took_prev_low_n
        out[f"close_back_inside_high_{n}"] = close_back_inside_high_n
        out[f"close_back_inside_low_{n}"] = close_back_inside_low_n
        out[f"sweep_reject_high_{n}"] = took_prev_high_n * close_back_inside_high_n
        out[f"sweep_reject_low_{n}"] = took_prev_low_n * close_back_inside_low_n
    return out.replace([np.inf, -np.inf], np.nan)


def drop_redundant_features(df: pd.DataFrame, corr_threshold: float = 0.92) -> list[str]:
    x = df.copy()
    cols = list(x.columns)
    if not cols:
        return []
    corr = x.corr(numeric_only=True).abs()
    keep: List[str] = []
    for c in cols:
        if c not in corr.columns:
            keep.append(c)
            continue
        drop = False
        for k in keep:
            if (k in corr.index) and (c in corr.columns):
                v = corr.loc[k, c]
                if pd.notna(v) and float(v) >= float(corr_threshold):
                    drop = True
                    break
        if not drop:
            keep.append(c)
    return keep


def select_and_transform_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    x = df.reindex(columns=list(feature_cols)).copy()
    cols = list(x.columns)

    # 1) Remove duplicate session encodings (prefer sess_* over is_*).
    has_is = any(c.startswith("is_") for c in cols)
    has_sess = any(c.startswith("sess_") for c in cols)
    if has_is and has_sess:
        cols = [c for c in cols if not c.startswith("is_")]

    # 2) Remove raw HTF levels; keep normalized distances/scores only.
    raw_htf_tokens = (
        "_ema20",
        "_ema50",
        "_ema200",
        "_bb_up",
        "_bb_dn",
        "_rolling_high",
        "_rolling_low",
        "_atr14",
    )
    cols = [c for c in cols if not (c.startswith("h1_") or c.startswith("d1_")) or not any(tok in c for tok in raw_htf_tokens)]

    # 3) Drop carry proxy by default; optional override via config column flag.
    cols = [c for c in cols if c != "carry_proxy"]

    # 4) Drop integer-coded regime IDs unless explicitly one-hot encoded.
    invalid_regime_cols = []
    for c in cols:
        if c in {"regime_variant_id", "h1_regime", "d1_regime", "regime"}:
            invalid_regime_cols.append(c)
            continue
        s = x[c]
        if pd.api.types.is_integer_dtype(s) and ("regime" in c.lower()) and ("_onehot_" not in c.lower()):
            invalid_regime_cols.append(c)
    cols = [c for c in cols if c not in set(invalid_regime_cols)]

    # 5) Deterministic order preserved, then remove high-corr redundancy.
    x = x.reindex(columns=cols)
    keep = drop_redundant_features(x, corr_threshold=0.92)
    x = x.reindex(columns=keep)

    stats: Dict[str, Any] = {
        "selected_columns": list(x.columns),
        "scalers": {},
        "dropped_invalid_regime_cols": invalid_regime_cols,
    }

    # 6) Fit scaling on provided frame only (caller should pass train fold).
    out = pd.DataFrame(index=x.index)
    for c in x.columns:
        s = pd.to_numeric(x[c], errors="coerce")
        if s.notna().sum() == 0:
            out[c] = s
            stats["scalers"][c] = {"method": "none"}
            continue
        q01 = float(s.quantile(0.01))
        q99 = float(s.quantile(0.99))
        s_clip = s.clip(lower=q01, upper=q99)

        # Robust scaling for fat tails.
        skew = float(s_clip.skew()) if s_clip.notna().sum() > 3 else 0.0
        kurt = float(s_clip.kurt()) if s_clip.notna().sum() > 3 else 0.0
        fat_tail = (abs(skew) > 1.0) or (kurt > 3.0)
        if fat_tail:
            med = float(s_clip.median())
            q25 = float(s_clip.quantile(0.25))
            q75 = float(s_clip.quantile(0.75))
            iqr = max(q75 - q25, 1e-9)
            z = (s_clip - med) / iqr
            stats["scalers"][c] = {"method": "robust", "median": med, "iqr": iqr, "clip_q01": q01, "clip_q99": q99}
        else:
            mu = float(s_clip.mean())
            sd = float(s_clip.std(ddof=0))
            sd = sd if sd > 1e-9 else 1.0
            z = (s_clip - mu) / sd
            stats["scalers"][c] = {"method": "zscore", "mean": mu, "std": sd, "clip_q01": q01, "clip_q99": q99}
        out[c] = z

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, stats


def compute_vol_scaled_triple_barrier_labels(
    df: pd.DataFrame,
    sigma_col: str,
    up_mult: float,
    dn_mult: float,
    min_horizon_bars: int,
    max_horizon_bars: int,
) -> pd.DataFrame:
    x = _ensure_utc_index(df)
    close = pd.to_numeric(x["close"], errors="coerce")
    sigma = pd.to_numeric(x[sigma_col], errors="coerce").ffill().bfill().fillna(0.0)
    n = int(len(x))
    min_h = max(1, int(min_horizon_bars))
    max_h = max(min_h, int(max_horizon_bars))

    # Deterministic horizon from sigma_t only: higher sigma => shorter horizon.
    sig_norm = (sigma / (1.0 + sigma)).clip(0.0, 1.0)
    h_float = (max_h - (max_h - min_h) * sig_norm).fillna(float(max_h))
    horizons = np.rint(h_float.to_numpy(dtype=float)).astype(int)
    horizons = np.clip(horizons, min_h, max_h)

    label_side = np.zeros(n, dtype=int)
    label_end = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    label_hbars = horizons.astype(int)

    idx = pd.DatetimeIndex(x.index)
    for i in range(n):
        if pd.isna(close.iloc[i]) or pd.isna(sigma.iloc[i]):
            continue
        h_i = int(label_hbars[i])
        end_i = min(n - 1, i + h_i)
        up_barrier = float(close.iloc[i] * (1.0 + float(up_mult) * float(sigma.iloc[i])))
        dn_barrier = float(close.iloc[i] * (1.0 - float(dn_mult) * float(sigma.iloc[i])))
        side = 0
        end_ts = idx[end_i]
        for j in range(i + 1, end_i + 1):
            px = float(close.iloc[j])
            if px >= up_barrier:
                side = 1
                end_ts = idx[j]
                break
            if px <= dn_barrier:
                side = -1
                end_ts = idx[j]
                break
        label_side[i] = int(side)
        label_end[i] = end_ts.tz_convert("UTC").tz_localize(None) if isinstance(end_ts, pd.Timestamp) else pd.NaT

    out = pd.DataFrame(index=x.index)
    out["label_side"] = label_side
    out["label_end_ts"] = pd.to_datetime(label_end, utc=True, errors="coerce")
    out["label_horizon_bars"] = label_hbars
    return out


def compute_meta_label(
    df: pd.DataFrame,
    base_prob_col: str,
    realized_net_edge_col: str,
    base_threshold: float,
) -> pd.Series:
    p = pd.to_numeric(df[base_prob_col], errors="coerce").fillna(0.0)
    edge = pd.to_numeric(df[realized_net_edge_col], errors="coerce").fillna(0.0)
    return ((p >= float(base_threshold)) & (edge > 0.0)).astype(int)


def generate_purged_walkforward_splits(
    index: pd.Index,
    label_end_ts: pd.Series,
    n_splits: int,
    test_size: int,
    embargo_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    tsz = max(1, int(test_size))
    ns = max(1, int(n_splits))
    emb = max(0, int(embargo_bars))
    if n < tsz + 2:
        return []

    total_test = ns * tsz
    start0 = max(0, n - total_test)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    end_series = pd.to_datetime(label_end_ts, utc=True, errors="coerce").reindex(idx)
    start_series = pd.Series(idx, index=idx)

    for k in range(ns):
        t0 = start0 + (k * tsz)
        t1 = min(n, t0 + tsz)
        if t1 <= t0:
            continue
        test_idx = np.arange(t0, t1, dtype=int)
        test_start = idx[t0]
        test_end = idx[t1 - 1]

        overlap = (start_series <= test_end) & (end_series >= test_start)
        overlap = overlap.fillna(False)

        embargo_mask = pd.Series(False, index=idx)
        emb_end_pos = min(n - 1, (t1 - 1) + emb)
        if emb_end_pos > (t1 - 1):
            embargo_mask.iloc[t1 : emb_end_pos + 1] = True

        test_mask = pd.Series(False, index=idx)
        test_mask.iloc[t0:t1] = True
        train_mask = ~(test_mask | overlap | embargo_mask)
        train_idx = np.where(train_mask.to_numpy())[0].astype(int)
        splits.append((train_idx, test_idx))
    return splits


def compute_fold_diagnostics(y_true, y_prob, costs, fold_id) -> dict:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(y_prob), errors="coerce").clip(1e-6, 1 - 1e-6)
    m = y.notna() & p.notna()
    y = y[m].astype(float)
    p = p[m].astype(float)
    if len(y) == 0:
        return {"fold_id": fold_id, "n": 0}

    brier = float(((p - y) ** 2).mean())
    logloss = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())

    # Calibration fit y ~= a + b*logit(p)
    lp = np.log(p / (1.0 - p))
    slope = np.nan
    intercept = np.nan
    if len(y) >= 20 and np.isfinite(lp).all():
        X = np.column_stack([np.ones(len(lp)), lp.to_numpy()])
        try:
            beta = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
            intercept = float(beta[0])
            slope = float(beta[1])
        except Exception:
            pass

    if isinstance(costs, dict):
        gross_win = pd.to_numeric(pd.Series(costs.get("gross_win", 1.0)), errors="coerce").reindex(y.index, fill_value=1.0)
        gross_loss = pd.to_numeric(pd.Series(costs.get("gross_loss", 1.0)), errors="coerce").reindex(y.index, fill_value=1.0)
        expected_cost = pd.to_numeric(pd.Series(costs.get("expected_cost", 0.0)), errors="coerce").reindex(y.index, fill_value=0.0)
        realized_r = pd.to_numeric(pd.Series(costs.get("realized_r", 0.0)), errors="coerce").reindex(y.index, fill_value=0.0)
        thr = float(costs.get("threshold", 0.5))
    else:
        gross_win = pd.Series(1.0, index=y.index)
        gross_loss = pd.Series(1.0, index=y.index)
        expected_cost = pd.Series(0.0, index=y.index)
        realized_r = pd.Series(0.0, index=y.index)
        thr = 0.5

    ev = (p * gross_win) - ((1.0 - p) * gross_loss) - expected_cost
    trade_mask = p >= thr
    trade_count = int(trade_mask.sum())
    hit_rate = float(y[trade_mask].mean()) if trade_count > 0 else 0.0
    ev_at_threshold = float(ev[trade_mask].mean()) if trade_count > 0 else 0.0

    eq = (1.0 + realized_r.where(trade_mask, 0.0).fillna(0.0)).cumprod()
    peak = eq.cummax().replace(0, np.nan)
    dd = ((peak - eq) / peak).fillna(0.0)
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    return {
        "fold_id": fold_id,
        "n": int(len(y)),
        "brier_score": brier,
        "log_loss": logloss,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_value_at_threshold": ev_at_threshold,
        "trade_count": trade_count,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
    }


def _fit_logistic_gd(X: np.ndarray, y: np.ndarray, max_iter: int = 300, lr: float = 0.05, l2: float = 1e-6, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 1e-3, size=X.shape[1])
    yv = y.astype(float)
    for _ in range(max_iter):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        g = (X.T @ (p - yv)) / max(1, len(yv))
        g += l2 * w
        w = w - (lr * g)
    return w


def fit_beta_calibrator(p_raw: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=float)
    if len(p) < 30:
        return fit_platt_calibrator(p, yy)
    X = np.column_stack([np.ones(len(p)), np.log(p), np.log(1.0 - p)])
    try:
        w = _fit_logistic_gd(X, yy, max_iter=400, lr=0.05, l2=1e-6, seed=42)
        return {"type": "beta", "w": w.tolist()}
    except Exception:
        return fit_platt_calibrator(p, yy)


def apply_beta_calibrator(calibrator, p_raw: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)
    if not isinstance(calibrator, dict):
        return p
    ctype = str(calibrator.get("type", ""))
    if ctype == "beta":
        w = np.asarray(calibrator.get("w", [0.0, 1.0, -1.0]), dtype=float)
        X = np.column_stack([np.ones(len(p)), np.log(p), np.log(1.0 - p)])
        z = X @ w
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if ctype == "platt":
        a = float(calibrator.get("a", 1.0))
        b = float(calibrator.get("b", 0.0))
        z = a * np.log(p / (1.0 - p)) + b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if ctype == "isotonic":
        xp = np.asarray(calibrator.get("x", []), dtype=float)
        fp = np.asarray(calibrator.get("y", []), dtype=float)
        if len(xp) >= 2 and len(fp) == len(xp):
            return np.interp(p, xp, fp, left=fp[0], right=fp[-1])
    return p


def fit_platt_calibrator(p_raw: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)
    yy = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(p)), np.log(p / (1.0 - p))])
    w = _fit_logistic_gd(X, yy, max_iter=300, lr=0.05, l2=1e-6, seed=42)
    return {"type": "platt", "a": float(w[1]), "b": float(w[0])}


def fit_isotonic_calibrator(p_raw: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(p_raw, dtype=float), 0.0, 1.0)
    yy = np.asarray(y, dtype=float)
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        yhat = ir.fit_transform(p, yy)
        xp = np.asarray(p, dtype=float)
        ord_idx = np.argsort(xp)
        xp = xp[ord_idx]
        fp = np.asarray(yhat, dtype=float)[ord_idx]
        # collapse duplicate xp deterministically
        uxp, idx_first = np.unique(xp, return_index=True)
        ufp = fp[idx_first]
        return {"type": "isotonic", "x": uxp.tolist(), "y": ufp.tolist()}
    except Exception:
        return fit_platt_calibrator(p, yy)


def fit_probability_calibrator(
    p_raw: np.ndarray,
    y: np.ndarray,
    regime_bucket: np.ndarray | None = None,
    isotonic_min_samples: int = 1000,
    bucket_min_samples: int = 300,
):
    p = np.asarray(p_raw, dtype=float)
    yy = np.asarray(y, dtype=float)
    n = len(p)
    if regime_bucket is not None:
        b = np.asarray(regime_bucket)
        out = {"type": "bucketed", "models": {}}
        for key in [0, 1]:
            m = b == key
            if int(m.sum()) < int(bucket_min_samples):
                continue
            if int(m.sum()) >= int(isotonic_min_samples):
                out["models"][str(key)] = fit_isotonic_calibrator(p[m], yy[m])
            else:
                out["models"][str(key)] = fit_beta_calibrator(p[m], yy[m])
        if out["models"]:
            return out
    if n >= int(isotonic_min_samples):
        return fit_isotonic_calibrator(p, yy)
    return fit_beta_calibrator(p, yy)


def apply_probability_calibrator(calibrator, p_raw: np.ndarray, regime_bucket: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(p_raw, dtype=float)
    if not isinstance(calibrator, dict):
        return np.clip(p, 0.0, 1.0)
    if calibrator.get("type") == "bucketed" and regime_bucket is not None:
        b = np.asarray(regime_bucket)
        out = np.clip(p.copy(), 0.0, 1.0)
        for key, model in calibrator.get("models", {}).items():
            m = b == int(key)
            if m.any():
                out[m] = apply_beta_calibrator(model, out[m])
        return np.clip(out, 0.0, 1.0)
    return np.clip(apply_beta_calibrator(calibrator, p), 0.0, 1.0)


def compute_expected_value(
    p_up: pd.Series,
    gross_win: pd.Series,
    gross_loss: pd.Series,
    expected_cost: pd.Series,
) -> pd.Series:
    p = pd.to_numeric(pd.Series(p_up), errors="coerce").fillna(0.0)
    idx = p.index
    gw = pd.Series(pd.to_numeric(pd.Series(gross_win), errors="coerce").fillna(0.0).to_numpy(), index=idx)
    gl = pd.Series(pd.to_numeric(pd.Series(gross_loss), errors="coerce").fillna(0.0).to_numpy(), index=idx)
    ec = pd.Series(pd.to_numeric(pd.Series(expected_cost), errors="coerce").fillna(0.0).to_numpy(), index=idx)
    n = min(len(p), len(gw), len(gl), len(ec))
    p = p.iloc[:n]
    gw = gw.iloc[:n]
    gl = gl.iloc[:n]
    ec = ec.iloc[:n]
    return (p * gw) - ((1.0 - p) * gl) - ec


def apply_trade_gating(
    df: pd.DataFrame,
    p_col: str,
    ev_col: str,
    min_ev: float,
    base_p_threshold: float,
    dynamic_threshold_col: str | None = None,
) -> pd.Series:
    p = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0)
    ev = pd.to_numeric(df[ev_col], errors="coerce").fillna(0.0)
    thr = pd.Series(float(base_p_threshold), index=df.index, dtype=float)
    if dynamic_threshold_col and dynamic_threshold_col in df.columns:
        dyn = pd.to_numeric(df[dynamic_threshold_col], errors="coerce")
        thr = dyn.where(dyn.notna(), thr)
    return ((ev > float(min_ev)) & (p > thr)).astype(int)


def set_deterministic_seed(seed: int = 42) -> None:
    np.random.seed(int(seed))


def _backfill_in_small_chunks(
    dm: DataManager,
    symbol: str,
    tf: Timeframe,
    start_naive: dt.datetime,
    end_naive: dt.datetime,
    chunk_days: int,
) -> None:
    cur = start_naive
    step = dt.timedelta(days=max(1, int(chunk_days)))
    while cur <= end_naive:
        nxt = min(end_naive, cur + step)
        try:
            dm.ensure_data(
                instrument=symbol,
                base_timeframe=tf,
                start_date=cur,
                end_date=nxt,
                timeframes=[tf],
                force_download=False,
            )
        except Exception as exc:
            print(f"Warning: chunk download failed for {symbol} {tf.name} {cur}..{nxt} ({exc})")
            # In this environment repeated retries are expensive; fail fast and
            # continue with whatever local data we have.
            break
        cur = nxt + dt.timedelta(seconds=1)


def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV from local warehouse; fallback to ensure_data if missing."""
    tf_str = str(timeframe).upper()
    if tf_str == "M5":
        # M5 is not represented in the Timeframe enum in this codebase.
        df_m5 = _load_direct_tf_file(symbol, "M5")
        if df_m5 is not None and not df_m5.empty:
            return _ensure_utc_index(df_m5[["open", "high", "low", "close", "volume"]].astype(float, errors="ignore"))
        # Derive M5 from local M1 when available.
        df_m1 = load_ohlcv(symbol, "M1")
        df_m5 = _resample_ohlcv_freq(df_m1, "5T")
        _save_direct_tf_file(df_m5, symbol, "M5")
        return _ensure_utc_index(df_m5[["open", "high", "low", "close", "volume"]].astype(float, errors="ignore"))

    tf = _tf_from_str(tf_str)
    wh = DataWarehouse(Path("data/backtesting"))
    df = wh.load(symbol, tf)
    # Prefer deriving HTF from local M15 when available to avoid unnecessary network calls.
    m15 = wh.load(symbol, Timeframe.M15)
    if tf != Timeframe.M15 and m15 is not None and not m15.empty:
        df_from_m15 = _resample_ohlcv(m15, tf)
        if df is None or df.empty:
            df = df_from_m15
        else:
            dfx = _ensure_utc_index(df)
            if pd.DatetimeIndex(df_from_m15.index).min() < pd.DatetimeIndex(dfx.index).min():
                df = df_from_m15
    dm = DataManager({"data_dir": "data/backtesting"})

    end = dt.datetime.now(dt.timezone.utc)
    years = int(os.getenv("REGIME_TRAIN_YEARS", "10") or "10")
    chunk_days = int(os.getenv("REGIME_DOWNLOAD_CHUNK_DAYS", "120") or "120")
    disable_download = str(os.getenv("REGIME_DISABLE_DOWNLOAD", "0")).lower() in {"1", "true", "yes", "on"}
    start = end - dt.timedelta(days=365 * max(1, years))
    # DataManager compares against cached index which may be tz-naive.
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)

    needs_download = df is None or df.empty
    if not needs_download:
        dfx = _ensure_utc_index(df)
        cached_min = pd.DatetimeIndex(dfx.index).min()
        needs_download = cached_min > start
    if disable_download:
        needs_download = False

    if needs_download:
        try:
            _backfill_in_small_chunks(dm, symbol, tf, start_naive, end_naive, chunk_days=chunk_days)
            df = wh.load(symbol, tf)
            # If direct HTF still missing after backfill, derive from local M15.
            if (df is None or df.empty) and tf != Timeframe.M15 and m15 is not None and not m15.empty:
                df = _resample_ohlcv(m15, tf)
        except Exception as exc:
            if df is not None and not df.empty:
                print(f"Warning: download failed for {symbol} {timeframe}; using local data only ({exc})")
            else:
                raise
    if df is None or df.empty:
        raise RuntimeError(f"No data found for {symbol} {timeframe}")
    need_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing OHLCV columns for {symbol} {timeframe}: {missing}")
    return _ensure_utc_index(df[need_cols].astype(float, errors="ignore"))


def make_features(df: pd.DataFrame, timeframe: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Create non-leaky rolling/lagged feature set."""
    x = _ensure_utc_index(df)
    c = x["close"].astype(float)
    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    v = x["volume"].astype(float)

    atr14 = _atr(x, 14)
    ema20 = _ema(c, 20)
    ema50 = _ema(c, 50)
    ema200 = _ema(c, 200)

    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std(ddof=0)
    bbw = (4.0 * bb_std) / (bb_mid.abs() + 1e-9)

    vol_ma = v.rolling(20, min_periods=20).mean()
    vol_sd = v.rolling(20, min_periods=20).std(ddof=0)
    ret1 = c.pct_change(1)
    vol10 = ret1.rolling(10, min_periods=10).std(ddof=0)
    vol30 = ret1.rolling(30, min_periods=30).std(ddof=0)

    slope20 = (ema20 - ema20.shift(5)) / 5.0
    slope50 = (ema50 - ema50.shift(10)) / 10.0
    trend_dir = np.sign(ema20 - ema50)
    trend_switch = (trend_dir != trend_dir.shift(1)).astype(float).fillna(0.0)
    trend_age = trend_switch.groupby((trend_switch > 0).cumsum()).cumcount().astype(float)

    impulse = ((c - o).abs() / (atr14 + 1e-9)).clip(0.0, 10.0)
    impulse_ema = impulse.ewm(span=8, adjust=False, min_periods=8).mean()
    vol_persist = (vol10 / (vol30 + 1e-9)).clip(0.0, 10.0)

    out = pd.DataFrame(index=x.index)
    out["ret1"] = ret1
    out["ret4"] = c.pct_change(4)
    out["ret12"] = c.pct_change(12)
    out["range_pct"] = (h - l) / (c.abs() + 1e-9)
    out["body_pct"] = (c - o) / (o.abs() + 1e-9)
    out["atr14"] = atr14
    out["atr_pct"] = atr14 / (c.abs() + 1e-9)
    out["ema20_50_spread"] = (ema20 - ema50) / (c.abs() + 1e-9)
    out["ema50_200_spread"] = (ema50 - ema200) / (c.abs() + 1e-9)
    out["dist_ema20_atr"] = (c - ema20) / (atr14 + 1e-9)
    out["dist_ema50_atr"] = (c - ema50) / (atr14 + 1e-9)
    out["ema20_slope_atr"] = slope20 / (atr14 + 1e-9)
    out["ema50_slope_atr"] = slope50 / (atr14 + 1e-9)
    out["trend_age_bars"] = trend_age
    out["trend_flip_rate_30"] = trend_switch.rolling(30, min_periods=30).mean()
    out["bbw"] = bbw
    out["vol_z"] = (v - vol_ma) / (vol_sd + 1e-9)
    out["realized_vol_10"] = vol10
    out["realized_vol_30"] = vol30
    out["vol_persistence"] = vol_persist
    out["impulse_ema"] = impulse_ema

    idx_utc = pd.DatetimeIndex(x.index).tz_convert("UTC")
    hour = idx_utc.hour.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * (hour / 24.0))
    out["hour_cos"] = np.cos(2.0 * np.pi * (hour / 24.0))
    out["is_tokyo"] = ((hour >= 0) & (hour < 9)).astype(float)
    out["is_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    out["is_ny"] = ((hour >= 12) & (hour < 21)).astype(float)
    out["is_overlap_ln_ny"] = ((hour >= 12) & (hour < 16)).astype(float)
    out["session_transition"] = (
        out["is_tokyo"].diff().abs().fillna(0.0)
        + out["is_london"].diff().abs().fillna(0.0)
        + out["is_ny"].diff().abs().fillna(0.0)
    ).clip(0.0, 1.0)

    macro_cfg = (config or {}).get("macro", {})
    carry_proxy = float(macro_cfg.get("carry_proxy", 0.0))
    out["carry_proxy"] = carry_proxy
    evf = _event_window_features(x.index, macro_cfg)
    out = pd.concat([out, evf], axis=1)

    # Session/slippage proxy: static session spread + volatility/event stress.
    spread_map = macro_cfg.get(
        "session_spread_bps",
        {"tokyo": 2.2, "london": 1.4, "newyork": 1.6, "overlap": 1.2, "offhours": 2.6},
    )
    base_spread = (
        out["is_overlap_ln_ny"] * float(spread_map.get("overlap", 1.2))
        + ((out["is_overlap_ln_ny"] <= 0) & (out["is_london"] > 0)).astype(float) * float(spread_map.get("london", 1.4))
        + ((out["is_overlap_ln_ny"] <= 0) & (out["is_ny"] > 0)).astype(float) * float(spread_map.get("newyork", 1.6))
        + ((out["is_tokyo"] > 0)).astype(float) * float(spread_map.get("tokyo", 2.2))
    )
    base_spread = base_spread.where(base_spread > 0, float(spread_map.get("offhours", 2.6)))
    atr_pct = out["atr_pct"].clip(lower=0.0)
    atr_rank = _rolling_percentile_rank(atr_pct, int(macro_cfg.get("slippage_vol_window", 240))).fillna(0.5)
    stress = (1.0 + 0.8 * atr_rank + 0.2 * out["vol_z"].clip(lower=0.0).fillna(0.0)).clip(0.5, 3.5)
    event_stress = 1.0 + (float(macro_cfg.get("event_slippage_mult", 0.6)) * out["event_window_flag"].fillna(0.0))
    out["spread_proxy_bps"] = base_spread
    out["slippage_proxy_bps"] = (base_spread * stress * event_stress).clip(0.2, 25.0)
    out["expected_cost_proxy_bps"] = out["spread_proxy_bps"] + out["slippage_proxy_bps"]

    # Module 1/2/3 feature blocks (all backward-computable).
    x_ext = x.copy()
    x_ext["atr14"] = atr14
    tz_name = str((config or {}).get("timezone", "Europe/London"))
    asia = compute_asia_range_features(x_ext, tz=tz_name)
    lo = compute_london_open_features(x_ext, tz=tz_name)

    # Prefer externally provided HTF frames; fallback to deterministic resample from LTF.
    h1_df = (config or {}).get("_df_h1")
    d1_df = (config or {}).get("_df_d1")
    if isinstance(h1_df, pd.DataFrame) and isinstance(d1_df, pd.DataFrame):
        htf_dist = compute_htf_distance_features(x_ext, h1_df, d1_df)
    else:
        h1_local = _resample_ohlcv_freq(x[["open", "high", "low", "close", "volume"]], "1H")
        d1_local = _resample_ohlcv_freq(x[["open", "high", "low", "close", "volume"]], "1D")
        htf_dist = compute_htf_distance_features(x_ext, h1_local, d1_local)

    micro = compute_candle_microstructure_features(x_ext)
    trend_q = compute_trend_quality_features(x_ext, lookbacks=tuple((config or {}).get("trend_quality_lookbacks", (8, 16, 32))))
    dist_shape = compute_distribution_shape_features(
        x_ext, lookbacks=tuple((config or {}).get("distribution_lookbacks", (16, 32)))
    )
    sweep = compute_sweep_features(x_ext, lookbacks=tuple((config or {}).get("sweep_lookbacks", (4, 8, 16))))
    out = pd.concat([out, asia, lo, htf_dist, micro, trend_q, dist_shape, sweep], axis=1)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_labels(df_ltf: pd.DataFrame, atr_series: pd.Series, barrier_cfg: Dict[str, Any]) -> Tuple[pd.Series, Dict[str, Any]]:
    """Horizon/barrier labels with optional neutral/no-trade filtering."""
    x = _ensure_utc_index(df_ltf)
    c = x["close"].astype(float)
    atr = pd.to_numeric(atr_series, errors="coerce").reindex(x.index)

    horizon = int(barrier_cfg.get("horizon_bars", 8))
    up_mult = float(barrier_cfg.get("up_atr_mult", 0.8))
    dn_mult = float(barrier_cfg.get("down_atr_mult", 0.8))

    fut = c.shift(-horizon)
    move = fut - c
    up_thr = up_mult * atr
    dn_thr = dn_mult * atr
    neutral_mult = float(barrier_cfg.get("neutral_atr_mult", 0.25))
    three_class = bool(barrier_cfg.get("three_class_labels", False))
    dyn_enabled = bool(barrier_cfg.get("dynamic_neutral_enabled", True))

    neutral_mult_series = pd.Series(neutral_mult, index=x.index, dtype=float)
    if dyn_enabled:
        vol_window = int(barrier_cfg.get("neutral_vol_window", 240))
        vol_rank = _rolling_percentile_rank((atr / (c.abs() + 1e-9)).clip(lower=0.0), vol_window).fillna(0.5)
        vol_lo = float(barrier_cfg.get("neutral_vol_rank_low", 0.30))
        vol_hi = float(barrier_cfg.get("neutral_vol_rank_high", 0.70))
        mult_low = float(barrier_cfg.get("neutral_mult_low_vol", max(0.05, neutral_mult * 0.8)))
        mult_high = float(barrier_cfg.get("neutral_mult_high_vol", neutral_mult * 1.6))
        neutral_mult_series = pd.Series(neutral_mult, index=x.index, dtype=float)
        neutral_mult_series = neutral_mult_series.where(vol_rank > vol_hi, mult_high)
        neutral_mult_series = neutral_mult_series.where(vol_rank >= vol_lo, mult_low)

        event_mult = float(barrier_cfg.get("neutral_mult_event", neutral_mult * 2.0))
        ev = _parse_event_timestamps(barrier_cfg.get("event_schedule_utc", []))
        if len(ev) > 0:
            idx = pd.DatetimeIndex(x.index).tz_convert("UTC")
            pre_m = int(barrier_cfg.get("event_pre_minutes", 60))
            post_m = int(barrier_cfg.get("event_post_minutes", 120))
            next_pos = ev.searchsorted(idx, side="left")
            prev_pos = next_pos - 1
            next_ts = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
            prev_ts = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")
            m_next = next_pos < len(ev)
            m_prev = prev_pos >= 0
            if np.any(m_next):
                next_ts.iloc[np.where(m_next)[0]] = ev[next_pos[m_next]]
            if np.any(m_prev):
                prev_ts.iloc[np.where(m_prev)[0]] = ev[prev_pos[m_prev]]
            mins_to = ((next_ts - pd.Series(idx, index=idx)).dt.total_seconds() / 60.0).fillna(1e6)
            mins_since = ((pd.Series(idx, index=idx) - prev_ts).dt.total_seconds() / 60.0).fillna(1e6)
            event_flag = ((mins_to >= 0.0) & (mins_to <= pre_m)) | ((mins_since >= 0.0) & (mins_since <= post_m))
            neutral_mult_series = neutral_mult_series.where(~event_flag, event_mult)
    neutral_mult_series = neutral_mult_series.clip(lower=0.05)

    y = pd.Series(np.nan, index=x.index, dtype=float)
    y[move > up_thr] = 1.0
    y[move < -dn_thr] = 0.0
    if three_class:
        neutral = move.abs() <= (neutral_mult_series * atr)
        y[neutral] = np.nan
    else:
        y = y.fillna(0.0)

    y = y.iloc[:-horizon] if horizon > 0 else y
    dropped_neutral = int(y.isna().sum())
    if three_class:
        y = y.dropna()
    y = y.astype(int)
    meta = {
        "horizon_bars": horizon,
        "up_atr_mult": up_mult,
        "down_atr_mult": dn_mult,
        "three_class_labels": three_class,
        "neutral_atr_mult": neutral_mult,
        "dynamic_neutral_enabled": dyn_enabled,
        "neutral_mult_dynamic_mean": float(neutral_mult_series.mean()),
        "neutral_mult_dynamic_p90": float(neutral_mult_series.quantile(0.9)),
        "neutral_dropped_rows": dropped_neutral,
    }
    return y, meta


@dataclass
class _LinearProbModel:
    feature_names: List[str]
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.feature_names].to_numpy(dtype=float)
        z = (arr - self.mu) / self.sd
        xb = np.hstack([np.ones((len(z), 1)), z]) @ self.w
        p = 1.0 / (1.0 + np.exp(-np.clip(xb, -40, 40)))
        return np.column_stack([1.0 - p, p])


def _fit_linear_prob(X: pd.DataFrame, y: pd.Series, ridge: float = 1e-3) -> _LinearProbModel:
    fn = list(X.columns)
    arr = X.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd[sd == 0] = 1.0
    z = (arr - mu) / sd

    yv = pd.to_numeric(y, errors="coerce").fillna(0).to_numpy(dtype=float)
    yb = (yv > 0).astype(float)

    Xd = np.hstack([np.ones((len(z), 1)), z])
    eye = np.eye(Xd.shape[1])
    eye[0, 0] = 0.0
    a = (Xd.T @ Xd) + (ridge * eye)
    b = Xd.T @ yb
    w = np.linalg.solve(a, b)
    return _LinearProbModel(feature_names=fn, mu=mu, sd=sd, w=w)


def walkforward_train_eval(X: pd.DataFrame, y: pd.Series, model_cfg: Dict[str, Any], splits_cfg: Dict[str, Any]):
    """Walk-forward train/eval returning metrics + trained models."""
    X1 = X.copy()
    y1 = y.reindex(X1.index)
    m = pd.concat([X1, y1.rename("_y")], axis=1).dropna(how="any")
    if m.empty:
        return {"oof_proba": pd.Series(dtype=float), "folds": 0}, []

    yv = m.pop("_y")
    Xv = m
    n = len(Xv)

    train_ratio = float(splits_cfg.get("train_ratio", 0.7))
    folds = int(splits_cfg.get("folds", 5))
    min_train = int(splits_cfg.get("min_train", 200))
    ridge = float(model_cfg.get("ridge", 1e-3))
    custom_splits = splits_cfg.get("custom_splits")

    if isinstance(custom_splits, list) and len(custom_splits) > 0:
        models: List[_LinearProbModel] = []
        oof = pd.Series(index=Xv.index, dtype=float)
        for split in custom_splits:
            if not isinstance(split, (list, tuple)) or len(split) != 2:
                continue
            tr_idx = np.asarray(split[0], dtype=int)
            te_idx = np.asarray(split[1], dtype=int)
            if len(tr_idx) < min_train or len(te_idx) == 0:
                continue
            tr_idx = tr_idx[(tr_idx >= 0) & (tr_idx < len(Xv))]
            te_idx = te_idx[(te_idx >= 0) & (te_idx < len(Xv))]
            if len(tr_idx) < min_train or len(te_idx) == 0:
                continue
            Xtr, ytr = Xv.iloc[tr_idx], yv.iloc[tr_idx]
            Xte = Xv.iloc[te_idx]
            model = _fit_linear_prob(Xtr, ytr, ridge=ridge)
            p = model.predict_proba(Xte)[:, 1]
            oof.iloc[te_idx] = p
            models.append(model)
        if models:
            return {"oof_proba": oof.ffill().fillna(0.5), "folds": len(models), "split_mode": "purged_custom"}, models

    base_train = max(min_train, int(n * train_ratio))
    rem = n - base_train
    if rem <= 0:
        model = _fit_linear_prob(Xv, yv, ridge=ridge)
        p = pd.Series(model.predict_proba(Xv)[:, 1], index=Xv.index)
        return {"oof_proba": p, "folds": 1}, [model]

    step = max(20, rem // max(1, folds))
    models: List[_LinearProbModel] = []
    oof = pd.Series(index=Xv.index, dtype=float)

    tr_end = base_train
    while tr_end < n:
        te_end = min(n, tr_end + step)
        Xtr, ytr = Xv.iloc[:tr_end], yv.iloc[:tr_end]
        Xte = Xv.iloc[tr_end:te_end]

        if len(Xtr) < min_train or Xte.empty:
            break

        model = _fit_linear_prob(Xtr, ytr, ridge=ridge)
        p = model.predict_proba(Xte)[:, 1]
        oof.iloc[tr_end:te_end] = p
        models.append(model)
        tr_end = te_end

    if not models:
        model = _fit_linear_prob(Xv, yv, ridge=ridge)
        p = pd.Series(model.predict_proba(Xv)[:, 1], index=Xv.index)
        return {"oof_proba": p, "folds": 1}, [model]

    return {"oof_proba": oof.ffill().fillna(0.5), "folds": len(models)}, models


def backtest_from_signals(df: pd.DataFrame, signals: pd.DataFrame, cost_cfg: Dict[str, Any]):
    """Lightweight signal backtest producing trade_log + equity_curve."""
    px = _ensure_utc_index(df)["close"].astype(float)
    s = signals.copy()
    if s.empty:
        eq = pd.Series(index=px.index, data=1.0, dtype=float)
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "pnl", "r_multiple"]), eq

    dir_col = "signal" if "signal" in s.columns else "direction"
    if dir_col not in s.columns:
        raise ValueError("signals must include 'signal' or 'direction' column")

    horizon = int(cost_cfg.get("horizon_bars", 8))
    spread_bps = float(cost_cfg.get("spread_bps", 1.0))
    fee_bps = float(cost_cfg.get("fee_bps", 0.5))
    total_cost = (spread_bps + fee_bps) / 10000.0

    entries = pd.DatetimeIndex(s.index).intersection(px.index)
    trade_rows = []

    for ts in entries:
        i = px.index.get_loc(ts)
        if isinstance(i, slice):
            continue
        j = i + horizon
        if j >= len(px):
            continue
        entry = float(px.iloc[i])
        exit_ = float(px.iloc[j])
        d = float(s.at[ts, dir_col])
        if d == 0:
            continue
        ret = (exit_ / max(entry, 1e-9)) - 1.0
        pnl = (d * ret) - total_cost
        trade_rows.append(
            {
                "entry_time": px.index[i],
                "exit_time": px.index[j],
                "direction": int(np.sign(d)),
                "pnl": float(pnl),
                "r_multiple": float(pnl),
            }
        )

    trades = pd.DataFrame(trade_rows)
    eq = pd.Series(index=px.index, data=1.0, dtype=float)
    if not trades.empty:
        for _, row in trades.iterrows():
            t = pd.Timestamp(row["exit_time"])
            if t in eq.index:
                eq.loc[t:] = eq.loc[t:] * (1.0 + float(row["pnl"]))
    return trades, eq
