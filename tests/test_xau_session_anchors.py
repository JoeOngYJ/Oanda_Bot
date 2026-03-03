from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_session_anchors as xsa
from scripts.xau_session_ingestion import default_session_config


def _synthetic_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=96, freq="15min", tz="UTC")
    # Build deterministic bars with obvious intra-session evolving highs/lows.
    base = 2000.0
    close = base + np.sin(np.linspace(0.0, 8.0, len(idx))) * 2.0
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15
    # Force a late-session new high and low so developing extremes differ early.
    high[20] += 4.0
    low[22] -= 4.0
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_no_future_leakage_with_developing_extremes():
    df = _synthetic_df()
    cfg = default_session_config("UTC")
    anchors = xsa.build_session_anchors(df, cfg)
    chk = xsa.detect_current_session_future_extreme_leakage(df.join(anchors))
    assert chk.empty


def test_truncation_invariance_no_lookahead():
    df = _synthetic_df()
    cfg = default_session_config("UTC")
    t = df.index[40]
    full = xsa.build_session_anchors(df, cfg)
    trunc = xsa.build_session_anchors(df.loc[:t], cfg)
    cols = [
        "current_session_open",
        "current_session_elapsed_bars",
        "current_session_developing_high",
        "current_session_developing_low",
        "current_session_developing_range",
        "current_close_position_in_session_range",
        "prior_session_high",
        "prior_session_low",
        "prior_session_open",
        "prior_session_close",
        "prior_day_high",
        "prior_day_low",
        "prior_day_open",
        "prior_day_close",
    ]
    for c in cols:
        a = full.loc[t, c]
        b = trunc.loc[t, c]
        if pd.isna(a) and pd.isna(b):
            continue
        assert np.isclose(float(a), float(b), equal_nan=True), c


def test_anchor_resets_at_new_session():
    df = _synthetic_df()
    cfg = default_session_config("UTC")
    anchors = xsa.build_session_anchors(df, cfg)
    starts = anchors["session_id"].ne(anchors["session_id"].shift(1))
    # elapsed bars restart from 1 at each session start
    assert (anchors.loc[starts, "current_session_elapsed_bars"] == 1).all()
    # current session open equals current row open at session start
    merged = df.join(anchors)
    assert np.allclose(
        merged.loc[starts, "current_session_open"].to_numpy(dtype=float),
        merged.loc[starts, "open"].to_numpy(dtype=float),
    )


def test_prior_completed_session_levels_roll_only_on_session_completion():
    df = _synthetic_df()
    cfg = default_session_config("UTC")
    anchors = xsa.build_session_anchors(df, cfg)
    viol = xsa.verify_prior_session_levels_constant(anchors)
    assert viol.empty
    # Ensure prior session level changes only at a new session boundary.
    col = "prior_session_high"
    prev = anchors[col].shift(1)
    changed = ~(anchors[col].eq(prev) | (anchors[col].isna() & prev.isna()))
    non_boundary_change = changed & anchors["session_id"].eq(anchors["session_id"].shift(1))
    assert not bool(non_boundary_change.fillna(False).any())


def test_leakage_detector_catches_intentional_future_extreme_use():
    df = _synthetic_df()
    cfg = default_session_config("UTC")
    anchors = xsa.build_session_anchors(df, cfg)
    # Intentionally leak: overwrite developing highs with full session high.
    z = df.join(anchors).copy()
    full_high = z.groupby("session_id")["high"].transform("max")
    z["current_session_developing_high"] = full_high
    leak = xsa.detect_current_session_future_extreme_leakage(z)
    assert not leak.empty
