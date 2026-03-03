from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_labeling as xl


def _label_cfg(event_mode: str = "include") -> xl.LabelConfig:
    tp = {
        "asia": 5.0,
        "pre_london": 1.0,
        "london_open": 0.5,
        "london_continuation": 0.5,
        "ny_open": 0.2,
        "ny_overlap_postdata": 0.5,
    }
    sl = {
        "asia": 5.0,
        "pre_london": 1.0,
        "london_open": 0.5,
        "london_continuation": 0.5,
        "ny_open": 0.2,
        "ny_overlap_postdata": 0.5,
    }
    hz = {
        "asia": 3,
        "pre_london": 3,
        "london_open": 3,
        "london_continuation": 3,
        "ny_open": 3,
        "ny_overlap_postdata": 3,
    }
    nb = {
        "asia": 0.01,
        "pre_london": 0.00001,
        "london_open": 0.00001,
        "london_continuation": 0.00001,
        "ny_open": 0.00001,
        "ny_overlap_postdata": 0.00001,
    }
    return xl.LabelConfig(
        tp_mult_by_session=tp,
        sl_mult_by_session=sl,
        horizon_by_session=hz,
        neutral_band_by_session=nb,
        vol_col="sigma",
        event_col="event_window_flag",
        event_mode=event_mode,
        spread_col="spread_proxy_bps",
        min_net_edge=0.0,
        max_mae_mult=3.0,
        max_spread_for_exec=5.0,
    )


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-02-01 00:00:00+00:00", periods=10, freq="15min", tz="UTC")
    close = np.array([100.00, 100.10, 100.18, 100.21, 100.30, 100.00, 99.80, 99.70, 99.95, 100.05])
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    # Make a distinct up-move later for ny_open row.
    high[6] = max(high[6], 100.40)
    # Avoid immediate SL touches in NY bars so tighter TP can resolve directionally.
    ny_ix = np.arange(5, len(idx))
    low[ny_ix] = np.minimum(open_[ny_ix], close[ny_ix]) - 0.005
    high[ny_ix] = np.maximum(high[ny_ix], np.maximum(open_[ny_ix], close[ny_ix]) + 0.08)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "sigma": 0.1,
            "spread_proxy_bps": 0.01,
            "event_window_flag": 0.0,
            "session_bucket": [
                "asia",
                "asia",
                "asia",
                "pre_london",
                "pre_london",
                "ny_open",
                "ny_open",
                "ny_open",
                "ny_open",
                "ny_open",
            ],
        },
        index=idx,
    )
    return df


def test_sessions_use_different_barrier_params():
    df = _make_df()
    cfg = _label_cfg("include")
    out = xl.build_session_conditioned_labels(df, cfg)
    # Asia row should be neutral under larger tp/sl and neutral band.
    assert int(out.iloc[0]["y_dir"]) == 0
    # NY-open rows should include directional outcomes under tighter barriers.
    assert (out.loc[df["session_bucket"].eq("ny_open"), "y_dir"].isin([-1, 1])).any()


def test_neutral_labels_produced():
    df = _make_df()
    cfg = _label_cfg("include")
    out = xl.build_session_conditioned_labels(df, cfg)
    assert (out["y_dir"] == 0).any()


def test_event_window_exclude_and_suspend():
    df = _make_df()
    df.loc[df.index[3:5], "event_window_flag"] = 1.0

    out_ex = xl.build_session_conditioned_labels(df, _label_cfg("exclude"))
    assert out_ex.loc[df.index[3:5], "y_dir"].isna().all()
    assert out_ex.loc[df.index[3:5], "label_state"].eq("excluded_event").all()

    out_su = xl.build_session_conditioned_labels(df, _label_cfg("suspend"))
    assert (out_su.loc[df.index[3:5], "y_dir"] == 0).all()
    assert out_su.loc[df.index[3:5], "label_state"].eq("suspended_event").all()


def test_class_weights_differ_by_session_when_balance_differs():
    labels = pd.DataFrame(
        {
            "session_bucket": ["asia"] * 6 + ["ny_open"] * 6,
            "y_dir": [1, 1, 1, 0, 0, -1, 1, 0, -1, -1, -1, -1],
        }
    )
    w = xl.compute_session_class_weights(labels)
    assert "asia" in w and "ny_open" in w
    assert w["asia"][1] != w["ny_open"][1]


def test_no_feature_mutation_or_label_leakback():
    df = _make_df()
    cols_before = list(df.columns)
    df_before = df.copy(deep=True)
    out = xl.build_session_conditioned_labels(df, _label_cfg("include"))
    assert list(df.columns) == cols_before
    pd.testing.assert_frame_equal(df, df_before)
    assert all(c.startswith("y_") or c in {"time_to_resolution_bucket", "label_state", "label_horizon_bars"} for c in out.columns)


def test_meta_exec_label_uses_cost_components():
    df = _make_df()
    cfg_lo = _label_cfg("include")
    cfg_lo = xl.LabelConfig(
        **{
            **cfg_lo.__dict__,
            "slippage_bps": 0.0,
            "commission_bps_per_side": 0.0,
            "min_net_edge": 0.0,
        }
    )
    cfg_hi = xl.LabelConfig(
        **{
            **cfg_lo.__dict__,
            "slippage_bps": 20.0,
            "commission_bps_per_side": 20.0,
        }
    )
    lo = xl.build_session_conditioned_labels(df, cfg_lo)
    hi = xl.build_session_conditioned_labels(df, cfg_hi)
    assert int(hi["y_meta_exec"].sum()) <= int(lo["y_meta_exec"].sum())
