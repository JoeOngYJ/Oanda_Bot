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
