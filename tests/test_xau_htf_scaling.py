from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.xau_htf_scaling as xhs


def test_align_htf_excludes_incomplete_future_bar():
    m15_idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=10, freq="15min", tz="UTC")  # until 02:15
    m15 = pd.DataFrame({"close": np.arange(len(m15_idx), dtype=float)}, index=m15_idx)

    htf_idx = pd.date_range("2025-01-01 00:00:00+00:00", periods=3, freq="1h", tz="UTC")
    htf = pd.DataFrame({"regime_score": [10.0, 20.0, 30.0]}, index=htf_idx)

    out = xhs.align_htf_features(m15, htf, ["regime_score"])
    assert len(out) == len(m15)
    # first completed H1 bar appears at 01:00 -> align first value from 01:00 onward
    assert pd.isna(out.loc[pd.Timestamp("2025-01-01 00:45:00+00:00"), "regime_score"])
    assert out.loc[pd.Timestamp("2025-01-01 01:00:00+00:00"), "regime_score"] == 10.0
    # bar at htf=02:00 completes at 03:00 and must not be used
    assert out.loc[pd.Timestamp("2025-01-01 02:15:00+00:00"), "regime_score"] == 20.0
    assert 30.0 not in out["regime_score"].dropna().tolist()


def test_scaler_fit_train_only_and_frozen_transform():
    n = 12
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "session_bucket": ["asia"] * 6 + ["ny_open"] * 6,
            "f_global": np.array([1, 2, 3, 4, 5, 6, 100, 110, 120, 130, 140, 150], dtype=float),
            "f_session": np.array([10, 11, 12, 13, 14, 15, 50, 51, 52, 53, 54, 55], dtype=float),
        },
        index=idx,
    )
    train = df.iloc[:8].copy()
    valid = df.iloc[8:].copy()
    bundle = xhs.fit_feature_scalers(
        train_df=train,
        feature_groups={"global": ["f_global"], "session_sensitive": ["f_session"]},
        by_session=True,
    )
    valid_t1 = xhs.transform_feature_scalers(valid.copy(), bundle)
    valid_t2 = xhs.transform_feature_scalers(valid.copy(), bundle)
    pd.testing.assert_frame_equal(valid_t1, valid_t2)
    # Ensure transform is non-identity and deterministic with frozen train stats.
    assert not np.allclose(valid_t1["f_global"].to_numpy(), valid["f_global"].to_numpy())


def test_schema_persistence_and_mismatch_error(tmp_path: Path):
    cols = ["session_bucket", "a", "b"]
    schema_path = tmp_path / "schema.json"
    scaler_path = tmp_path / "scalers.json"

    xhs.save_feature_schema(str(schema_path), cols)
    assert xhs.load_feature_schema(str(schema_path)) == cols

    bundle = {
        "column_order": cols,
        "by_session": False,
        "feature_groups": {"global": ["a"], "session_sensitive": []},
        "global_scalers": {"a": {"method": "robust", "clip_q01": 0.0, "clip_q99": 1.0, "median": 0.5, "iqr": 0.5}},
        "session_scalers": {},
    }
    xhs.save_scaler_metadata(str(scaler_path), bundle)
    loaded = xhs.load_scaler_metadata(str(scaler_path))
    assert loaded["column_order"] == cols

    with pytest.raises(ValueError, match="Inference schema mismatch"):
        xhs.assert_inference_schema(cols, ["session_bucket", "b", "a"])
