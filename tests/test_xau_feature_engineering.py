from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_feature_engineering as xfe
from scripts.xau_session_anchors import build_session_anchors
from scripts.xau_session_ingestion import default_session_config


def _make_df(n: int = 480) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(123)
    ret = rng.normal(0.0, 0.0007, size=n)
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0010, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    spread_proxy_bps = 2.0 + np.abs(rng.normal(0.0, 0.5, size=n))
    event_flag = np.zeros(n)
    event_flag[100:110] = 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "spread_proxy_bps": spread_proxy_bps,
            "event_window_flag": event_flag,
        },
        index=idx,
    )


def _build_full_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    anchors = build_session_anchors(df, default_session_config("UTC"))
    base = df.join(anchors)
    cfg = {
        "temporal_context_mode": "hybrid",
        "short_context_lookbacks": [8, 16, 32],
        "slow_regime_days": [2, 4],
        "sequence_lookback_bars": 64,
        "sequence_stride_bars": 8,
    }
    shared = xfe.build_shared_features(base, config=cfg)
    sess = xfe.build_session_features(base, config=cfg)
    # use active-row collapsed session block for interactions
    sess_union = pd.DataFrame(index=base.index)
    for block in sess.values():
        sess_union = sess_union.combine_first(block)
    merged = base.join(shared).join(sess_union)
    return merged


def test_expected_columns_exist():
    df = _make_df()
    anchors = build_session_anchors(df, default_session_config("UTC"))
    base = df.join(anchors)
    shared = xfe.build_shared_features(base, config={"temporal_context_mode": "hybrid"})
    session_blocks = xfe.build_session_features(base, config={"temporal_context_mode": "hybrid"})
    sess_union = pd.DataFrame(index=base.index)
    for b in session_blocks.values():
        sess_union = sess_union.combine_first(b)
    inter = xfe.build_interaction_features(base.join(shared).join(sess_union))

    expected_shared = {
        "sf_realized_range",
        "sf_vol_scale_atr14",
        "sf_body_to_range_ratio",
        "sf_wick_asymmetry",
        "sf_directional_efficiency_8",
        "sf_dist_prior_day_high_atr",
        "sf_dist_prior_session_low_atr",
        "sf_spread_range_ratio",
        "sf_ctx_eff_ratio_16",
        "sf_regime_vol_med_ratio_20d",
        "sf_seq_ret_atr_l001",
    }
    expected_session = {
        "ss_dist_from_session_open",
        "ss_current_dev_session_range_width",
        "ss_breakout_accept_count_up",
        "ss_sweep_reject_high_8",
        "ss_transition_london_to_ny",
        "ss_ctx_repeat_fail_high_96",
    }
    expected_inter = {
        "if_session_london_x_struct_dist",
        "if_event_x_breakout_accept",
        "if_spread_x_barrier_scale_proxy",
    }
    assert expected_shared.issubset(set(shared.columns))
    for _, block in session_blocks.items():
        assert expected_session.issubset(set(block.columns))
    assert expected_inter.issubset(set(inter.columns))


def test_no_nan_explosion_beyond_warmup_windows():
    df = _make_df()
    feat = _build_full_feature_df(df)
    warm = feat.iloc[120:]
    feature_cols = [c for c in feat.columns if c.startswith(("sf_", "ss_", "if_"))]
    nan_ratio = warm[feature_cols].isna().mean().mean()
    assert nan_ratio < 0.20


def test_breakout_acceptance_no_future_close_usage():
    df = _make_df()
    cfg = default_session_config("UTC")
    anchors_full = build_session_anchors(df, cfg)
    base_full = df.join(anchors_full)
    sess_full = xfe.build_session_features(base_full, config={"temporal_context_mode": "hybrid"})
    union_full = pd.DataFrame(index=base_full.index)
    for b in sess_full.values():
        union_full = union_full.combine_first(b)

    t = df.index[250]
    anchors_trunc = build_session_anchors(df.loc[:t], cfg)
    base_trunc = df.loc[:t].join(anchors_trunc)
    sess_trunc = xfe.build_session_features(base_trunc, config={"temporal_context_mode": "hybrid"})
    union_trunc = pd.DataFrame(index=base_trunc.index)
    for b in sess_trunc.values():
        union_trunc = union_trunc.combine_first(b)

    for c in ["ss_breakout_accept_count_up", "ss_breakout_accept_count_dn"]:
        a = union_full.loc[t, c]
        b = union_trunc.loc[t, c]
        assert np.isclose(float(a), float(b), equal_nan=True)


def test_session_specific_blocks_populate_only_on_matching_rows():
    df = _make_df()
    anchors = build_session_anchors(df, default_session_config("UTC"))
    base = df.join(anchors)
    blocks = xfe.build_session_features(base, config={"temporal_context_mode": "hybrid"})
    s = base["session_bucket"].astype(str)

    for sess, block in blocks.items():
        active = s.eq(sess)
        assert block.loc[active, "ss_dist_from_session_open"].notna().all()
        assert block.loc[~active, "ss_dist_from_session_open"].isna().all()


def test_interaction_features_deterministic():
    df = _make_df()
    feat1 = _build_full_feature_df(df)
    feat2 = _build_full_feature_df(df)
    inter1 = xfe.build_interaction_features(feat1)
    inter2 = xfe.build_interaction_features(feat2)
    pd.testing.assert_frame_equal(inter1, inter2)


def test_short_context_causal_at_timestamp():
    df = _make_df()
    cfg = {"temporal_context_mode": "engineered", "short_context_lookbacks": [8, 16]}
    t = df.index[260]
    anchors_full = build_session_anchors(df, default_session_config("UTC"))
    anchors_trunc = build_session_anchors(df.loc[:t], default_session_config("UTC"))
    base_full = df.join(anchors_full)
    base_trunc = df.loc[:t].join(anchors_trunc)

    full = xfe.build_shared_features(base_full, config=cfg)
    trunc = xfe.build_shared_features(base_trunc, config=cfg)
    cols = ["sf_ctx_eff_ratio_8", "sf_ctx_break_accept_up_count_16", "sf_ctx_comp_exp_ratio_16"]
    for c in cols:
        assert np.isclose(float(full.loc[t, c]), float(trunc.loc[t, c]), equal_nan=True)


def test_sequence_feature_shape_deterministic():
    df = _make_df()
    anchors = build_session_anchors(df, default_session_config("UTC"))
    base = df.join(anchors)
    cfg = {"temporal_context_mode": "sequence", "sequence_lookback_bars": 64, "sequence_stride_bars": 8}
    a = xfe.build_shared_features(base, config=cfg)
    b = xfe.build_shared_features(base, config=cfg)
    seq_cols = [c for c in a.columns if c.startswith("sf_seq_")]
    assert len(seq_cols) == 24  # 3 signals x 8 lag taps
    assert seq_cols == [c for c in b.columns if c.startswith("sf_seq_")]
    pd.testing.assert_frame_equal(a[seq_cols], b[seq_cols])
