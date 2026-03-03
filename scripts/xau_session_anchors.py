from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.xau_session_ingestion import SessionConfig, assign_session_bucket


def _validate_input_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame index must be a DatetimeIndex.")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("Input DataFrame index must be timezone-aware.")
    if not bool(idx.is_monotonic_increasing):
        raise ValueError("Input DataFrame index must be monotonic increasing.")
    if bool(idx.duplicated().any()):
        raise ValueError("Input DataFrame index contains duplicate timestamps.")


def _validate_price_columns(df: pd.DataFrame) -> None:
    req = ["open", "high", "low", "close"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for session anchors: {miss}")


def _build_session_id(session_bucket: pd.Series) -> pd.Series:
    return session_bucket.astype(str).ne(session_bucket.astype(str).shift(1)).cumsum().astype(int)


def compute_developing_session_state(df: pd.DataFrame, session_id: pd.Series) -> pd.DataFrame:
    """Compute current session developing state using only bars observed up to t."""

    out = pd.DataFrame(index=df.index)
    grp = session_id
    out["current_session_open"] = pd.to_numeric(df["open"], errors="coerce").groupby(grp).transform("first")
    out["current_session_elapsed_bars"] = (pd.Series(np.arange(len(df)), index=df.index).groupby(grp).cumcount() + 1).astype(int)
    out["current_session_developing_high"] = pd.to_numeric(df["high"], errors="coerce").groupby(grp).cummax()
    out["current_session_developing_low"] = pd.to_numeric(df["low"], errors="coerce").groupby(grp).cummin()
    out["current_session_developing_range"] = out["current_session_developing_high"] - out["current_session_developing_low"]
    rng = out["current_session_developing_range"]
    out["current_close_position_in_session_range"] = np.where(
        rng > 0,
        (pd.to_numeric(df["close"], errors="coerce") - out["current_session_developing_low"]) / rng,
        np.nan,
    )
    return out


def compute_prior_completed_session_levels(df: pd.DataFrame, session_id: pd.Series) -> pd.DataFrame:
    """Map prior fully completed session OHLC extremes to each bar in current session."""

    agg = (
        df.assign(_session_id=session_id.values)
        .groupby("_session_id", sort=True)
        .agg(
            session_open=("open", "first"),
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
        )
    )
    prior = agg.shift(1)
    out = pd.DataFrame(index=df.index)
    sid = session_id.astype(int)
    out["prior_session_high"] = sid.map(prior["session_high"])
    out["prior_session_low"] = sid.map(prior["session_low"])
    out["prior_session_open"] = sid.map(prior["session_open"])
    out["prior_session_close"] = sid.map(prior["session_close"])
    return out


def compute_prior_day_levels(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Map prior fully completed trading day OHLC levels to each bar."""

    idx_local_day = pd.DatetimeIndex(df.index).tz_convert(tz).normalize()
    day_agg = (
        df.assign(_trade_day=idx_local_day)
        .groupby("_trade_day", sort=True)
        .agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
    )
    prior_day = day_agg.shift(1)
    out = pd.DataFrame(index=df.index)
    out["prior_day_high"] = idx_local_day.map(prior_day["day_high"].to_dict())
    out["prior_day_low"] = idx_local_day.map(prior_day["day_low"].to_dict())
    out["prior_day_open"] = idx_local_day.map(prior_day["day_open"].to_dict())
    out["prior_day_close"] = idx_local_day.map(prior_day["day_close"].to_dict())
    return out


def build_session_anchors(df: pd.DataFrame, session_config: SessionConfig) -> pd.DataFrame:
    """Build leakage-safe session anchors and structural state features."""

    _validate_input_index(df)
    _validate_price_columns(df)

    base = assign_session_bucket(df, session_config, add_helper_columns=False).copy()
    session_id = _build_session_id(base["session_bucket"])

    current = compute_developing_session_state(base, session_id)
    prior_sess = compute_prior_completed_session_levels(base, session_id)
    prior_day = compute_prior_day_levels(base, session_config.tz)

    out = pd.concat([base[["session_bucket"]], current, prior_sess, prior_day], axis=1)
    out["session_id"] = session_id.astype(int)
    return out


def detect_current_session_future_extreme_leakage(
    df: pd.DataFrame,
    session_id_col: str = "session_id",
    high_col: str = "high",
    low_col: str = "low",
    dev_high_col: str = "current_session_developing_high",
    dev_low_col: str = "current_session_developing_low",
) -> pd.DataFrame:
    """Detect impossible early equality with future full-session extremes."""

    req = [session_id_col, high_col, low_col, dev_high_col, dev_low_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for leakage check: {miss}")

    rows: List[Dict[str, object]] = []
    x = df.copy()
    for sid, g in x.groupby(session_id_col, sort=True):
        full_high = float(pd.to_numeric(g[high_col], errors="coerce").max())
        full_low = float(pd.to_numeric(g[low_col], errors="coerce").min())
        first_high_ts = g.index[pd.to_numeric(g[high_col], errors="coerce").eq(full_high)][0]
        first_low_ts = g.index[pd.to_numeric(g[low_col], errors="coerce").eq(full_low)][0]

        pre_high = g.loc[g.index < first_high_ts]
        pre_low = g.loc[g.index < first_low_ts]

        high_leak = pre_high[pd.to_numeric(pre_high[dev_high_col], errors="coerce").eq(full_high)]
        low_leak = pre_low[pd.to_numeric(pre_low[dev_low_col], errors="coerce").eq(full_low)]

        for ts in high_leak.index:
            rows.append({"timestamp": ts, "session_id": sid, "leak_type": "high", "full_extreme": full_high})
        for ts in low_leak.index:
            rows.append({"timestamp": ts, "session_id": sid, "leak_type": "low", "full_extreme": full_low})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "session_id", "leak_type", "full_extreme"])
    return pd.DataFrame(rows).sort_values(["timestamp", "session_id"]).reset_index(drop=True)


def verify_prior_session_levels_constant(
    df: pd.DataFrame,
    session_id_col: str = "session_id",
) -> pd.DataFrame:
    """Verify prior session levels are constant within each next session."""

    req = [session_id_col, "prior_session_high", "prior_session_low", "prior_session_open", "prior_session_close"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for prior session constant check: {miss}")

    rows: List[Dict[str, object]] = []
    cols = ["prior_session_high", "prior_session_low", "prior_session_open", "prior_session_close"]
    for sid, g in df.groupby(session_id_col, sort=True):
        for c in cols:
            nuniq = int(g[c].nunique(dropna=False))
            if nuniq > 1:
                rows.append({"session_id": sid, "column": c, "nunique": nuniq})
    return pd.DataFrame(rows, columns=["session_id", "column", "nunique"])


def verify_prior_day_levels_constant(
    df: pd.DataFrame,
    tz: str,
) -> pd.DataFrame:
    """Verify prior day levels are constant during each current trading day."""

    req = ["prior_day_high", "prior_day_low", "prior_day_open", "prior_day_close"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for prior day constant check: {miss}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex for day-level validation.")

    local_day = pd.DatetimeIndex(df.index).tz_convert(tz).normalize()
    cols = ["prior_day_high", "prior_day_low", "prior_day_open", "prior_day_close"]
    rows: List[Dict[str, object]] = []
    for d in pd.Index(local_day).unique():
        g = df.loc[local_day == d, cols]
        for c in cols:
            nuniq = int(g[c].nunique(dropna=False))
            if nuniq > 1:
                rows.append({"trade_day": str(pd.Timestamp(d).date()), "column": c, "nunique": nuniq})
    return pd.DataFrame(rows, columns=["trade_day", "column", "nunique"])
