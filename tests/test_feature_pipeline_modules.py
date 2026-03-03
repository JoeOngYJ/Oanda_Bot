from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pipeline_compat_adapter import (
    apply_beta_calibrator,
    apply_trade_gating,
    compute_asia_range_features,
    compute_candle_microstructure_features,
    compute_distribution_shape_features,
    compute_expected_value,
    compute_htf_distance_features,
    compute_london_open_features,
    compute_meta_label,
    compute_sweep_features,
    compute_trend_quality_features,
    compute_vol_scaled_triple_barrier_labels,
    fit_beta_calibrator,
    generate_purged_walkforward_splits,
    select_and_transform_features,
    set_deterministic_seed,
)


def _make_m15_df(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    ret = rng.normal(0.0, 0.0008, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0012, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    vol = rng.integers(100, 1000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_expected_columns_created():
    df = _make_m15_df()
    df["atr14"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    h1 = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    d1 = df.resample("1d").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    asia = compute_asia_range_features(df)
    lo = compute_london_open_features(df)
    micro = compute_candle_microstructure_features(df)
    sweep = compute_sweep_features(df)
    htf = compute_htf_distance_features(df, h1, d1)
    labels = compute_vol_scaled_triple_barrier_labels(df, sigma_col="atr14", up_mult=2.0, dn_mult=2.0, min_horizon_bars=4, max_horizon_bars=24)

    for c in ["asia_high", "asia_low", "asia_range_atr", "dist_to_asia_mid_atr"]:
        assert c in asia.columns
    for c in ["lo_open_price", "dist_from_lo_open_atr", "lo_break_of_asia"]:
        assert c in lo.columns
    for c in ["upper_wick_pct", "wick_imbalance", "body_direction_persist_3"]:
        assert c in micro.columns
    for c in ["sweep_reject_high_4", "sweep_reject_low_16"]:
        assert c in sweep.columns
    for c in ["dist_prev_day_high_atr", "dist_prev_week_low_atr", "adr20", "asia_range_pct_of_adr"]:
        assert c in htf.columns
    for c in ["label_side", "label_end_ts", "label_horizon_bars"]:
        assert c in labels.columns

    p = np.linspace(0.01, 0.99, len(df))
    y = (df["close"].pct_change().fillna(0.0) > 0).astype(int).to_numpy()
    cal = fit_beta_calibrator(p, y)
    p_cal = apply_beta_calibrator(cal, p)
    assert len(p_cal) == len(p)

    ev = compute_expected_value(pd.Series(p_cal, index=df.index), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index), pd.Series(0.01, index=df.index))
    gate = apply_trade_gating(pd.DataFrame({"p": p_cal, "ev": ev}, index=df.index), p_col="p", ev_col="ev", min_ev=0.0, base_p_threshold=0.5)
    assert len(gate) == len(df)
    assert set(np.unique(gate.dropna().astype(int))) <= {0, 1}


def test_no_nans_after_warmup_and_feature_selection():
    df = _make_m15_df()
    df["atr14"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    feat = pd.concat(
        [
            compute_asia_range_features(df),
            compute_london_open_features(df),
            compute_candle_microstructure_features(df),
            compute_trend_quality_features(df),
            compute_distribution_shape_features(df),
            compute_sweep_features(df),
        ],
        axis=1,
    )
    keep_rows = feat.index[300:]
    local = pd.DatetimeIndex(keep_rows).tz_convert("Europe/London")
    after_0800 = ((local.hour * 60 + local.minute) >= 8 * 60)
    sub = feat.loc[keep_rows[after_0800]]
    transformed, stats = select_and_transform_features(sub, list(sub.columns))
    assert len(stats["selected_columns"]) == transformed.shape[1]
    assert transformed.notna().all().all()


def test_no_future_leakage_recompute_matches_truncated():
    df = _make_m15_df()
    df["atr14"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    t = df.index[700]

    full = compute_asia_range_features(df)
    trunc = compute_asia_range_features(df.loc[:t])
    assert np.isclose(full.loc[t, "asia_range_atr"], trunc.loc[t, "asia_range_atr"], equal_nan=True)

    full_micro = compute_candle_microstructure_features(df)
    trunc_micro = compute_candle_microstructure_features(df.loc[:t])
    assert np.isclose(full_micro.loc[t, "wick_imbalance"], trunc_micro.loc[t, "wick_imbalance"], equal_nan=True)

    full_sweep = compute_sweep_features(df)
    trunc_sweep = compute_sweep_features(df.loc[:t])
    assert np.isclose(full_sweep.loc[t, "sweep_reject_high_8"], trunc_sweep.loc[t, "sweep_reject_high_8"], equal_nan=True)


def test_htf_merge_non_leaky_and_completed_only():
    df = _make_m15_df(2000)
    df["atr14"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    h1 = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    d1 = df.resample("1d").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    htf = compute_htf_distance_features(df, h1, d1)

    # Reconstruct mapped prev_day_high from distance definition and verify it equals shifted daily high mapping.
    recon_prev_day_high = df["close"] - (htf["dist_prev_day_high_atr"] * df["atr14"])
    expected_prev_day_high = d1["high"].shift(1).reindex(df.index, method="ffill")
    m = recon_prev_day_high.notna() & expected_prev_day_high.notna()
    assert np.allclose(recon_prev_day_high[m].to_numpy(), expected_prev_day_high[m].to_numpy(), atol=1e-6, rtol=1e-6)


def test_purged_walkforward_splits_and_embargo():
    idx = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    label_end = pd.Series(idx + pd.Timedelta(minutes=15 * 5), index=idx)
    splits = generate_purged_walkforward_splits(idx, label_end, n_splits=3, test_size=40, embargo_bars=8)
    assert len(splits) == 3
    for tr, te in splits:
        assert np.all(np.diff(te) == 1)
        ts = idx[te[0]]
        te_end = idx[te[-1]]
        emb_end = te[-1] + 8
        tr_set = set(tr.tolist())
        for i in tr:
            s = idx[i]
            e = label_end.iloc[i]
            overlap = (s <= te_end) and (e >= ts)
            assert not overlap
            if te[-1] < i <= min(len(idx) - 1, emb_end):
                assert i not in tr_set


def test_reproducibility_same_seed_same_outputs():
    df = _make_m15_df()
    y = (df["close"].pct_change().fillna(0.0) > 0).astype(int).to_numpy()
    p = np.linspace(0.01, 0.99, len(df))

    set_deterministic_seed(42)
    c1 = fit_beta_calibrator(p, y)
    p1 = apply_beta_calibrator(c1, p)
    ev1 = compute_expected_value(pd.Series(p1), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index), pd.Series(0.01, index=df.index))
    g1 = apply_trade_gating(pd.DataFrame({"p": p1, "ev": ev1}), p_col="p", ev_col="ev", min_ev=0.0, base_p_threshold=0.55)

    set_deterministic_seed(42)
    c2 = fit_beta_calibrator(p, y)
    p2 = apply_beta_calibrator(c2, p)
    ev2 = compute_expected_value(pd.Series(p2), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index), pd.Series(0.01, index=df.index))
    g2 = apply_trade_gating(pd.DataFrame({"p": p2, "ev": ev2}), p_col="p", ev_col="ev", min_ev=0.0, base_p_threshold=0.55)

    assert np.allclose(p1, p2)
    assert np.allclose(ev1.to_numpy(), ev2.to_numpy())
    assert np.array_equal(g1.to_numpy(), g2.to_numpy())


def test_meta_label_shape():
    df = pd.DataFrame(
        {
            "p": [0.2, 0.8, 0.9, 0.55],
            "edge": [-0.1, 0.2, -0.05, 0.01],
        }
    )
    m = compute_meta_label(df, "p", "edge", 0.6)
    assert m.tolist() == [0, 1, 0, 0]
