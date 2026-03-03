from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.xau_wfo_models as xwm


def _make_df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    close = 2000.0 + np.linspace(0.0, 10.0, n)
    df = pd.DataFrame({"close": close}, index=idx)
    # label windows end 8 bars after current.
    df["label_end_ts"] = df.index.to_series().shift(-8)
    return df


def test_split_boundaries_and_no_leak():
    df = _make_df(320)
    cfg = xwm.SplitConfig(n_splits=3, val_size=40, max_label_horizon_bars=8, embargo_bars=4, min_train_size=100)
    splits = xwm.make_purged_walk_forward_splits(df, cfg)
    assert len(splits) == 3

    for s in splits:
        tr = np.array(s["train_indices"], dtype=int)
        va = np.array(s["validation_indices"], dtype=int)
        assert len(np.intersect1d(tr, va)) == 0
        assert tr.max() < va.min()
        # purge bars are excluded from train
        assert tr.max() <= va.min() - cfg.max_label_horizon_bars - 1
        assert s["purge_window"]["bars"] == cfg.max_label_horizon_bars
        assert 0 <= s["embargo_window"]["bars"] <= cfg.embargo_bars


def test_repeated_training_same_seed_identical_outputs():
    n = 180
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    Xs = pd.DataFrame({"a": np.linspace(0, 1, n), "b": np.linspace(1, 2, n)}, index=idx)
    ys = pd.Series((Xs["a"] > 0.5).astype(int), index=idx)
    Zx = pd.DataFrame({"sx": np.sin(np.linspace(0, 2, n))}, index=idx)

    trunk1 = xwm.fit_shared_trunk(Xs, ys, seed=7)
    trunk2 = xwm.fit_shared_trunk(Xs, ys, seed=7)
    assert xwm.to_jsonable(trunk1) == xwm.to_jsonable(trunk2)

    Z1 = xwm.transform_shared_trunk(trunk1, Xs)
    Z2 = xwm.transform_shared_trunk(trunk2, Xs)
    pd.testing.assert_frame_equal(Z1, Z2)

    head1 = xwm.fit_session_head("asia", Z1, Zx, ys, seed=11)
    head2 = xwm.fit_session_head("asia", Z1, Zx, ys, seed=11)
    assert xwm.to_jsonable(head1) == xwm.to_jsonable(head2)

    p1 = xwm.predict_session_head(head1, Z1, Zx)
    p2 = xwm.predict_session_head(head2, Z1, Zx)
    pd.testing.assert_series_equal(p1, p2)


def test_feature_schema_mismatch_raises():
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    X = pd.DataFrame({"a": np.linspace(0, 1, n), "b": np.linspace(1, 2, n)}, index=idx)
    y = pd.Series((X["a"] > 0.5).astype(int), index=idx)
    trunk = xwm.fit_shared_trunk(X, y, seed=3)

    bad = X[["b", "a"]].copy()
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        xwm.transform_shared_trunk(trunk, bad)


def test_anchored_yearly_splits_are_leakage_safe():
    idx = pd.date_range("2022-01-01", "2025-12-31 23:45:00", freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(1900.0, 2200.0, len(idx))}, index=idx)
    df["label_end_ts"] = df.index.to_series().shift(-8)
    cfg = xwm.SplitConfig(
        n_splits=3,
        val_size=100,
        max_label_horizon_bars=8,
        embargo_bars=16,
        min_train_size=1000,
        split_mode="anchored_yearly",
        anchor_train_start="2022-01-01T00:00:00Z",
        validation_years=[2023, 2024, 2025],
    )
    splits = xwm.make_purged_walk_forward_splits(df, cfg)
    assert len(splits) == 3
    for s in splits:
        tr = np.array(s["train_indices"], dtype=int)
        va = np.array(s["validation_indices"], dtype=int)
        assert len(tr) > 0 and len(va) > 0
        assert tr.max() < va.min()
        # train label windows must end before validation start.
        le = pd.to_datetime(df.iloc[tr]["label_end_ts"], utc=True, errors="coerce")
        assert bool(((le.isna()) | (le < df.index[va.min()])).all())
