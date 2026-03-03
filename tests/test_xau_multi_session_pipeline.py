from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.xau_multi_session_pipeline as msp


def _make_df(n: int = 1200, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    ret = rng.normal(0.0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0012, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    vol = rng.integers(100, 1000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_session_segmentation_deterministic():
    df = _make_df(200)
    s1 = msp.segment_sessions(df.index, msp.SessionConfig())
    s2 = msp.segment_sessions(df.index, msp.SessionConfig())
    assert s1.equals(s2)
    assert set(pd.unique(s1)) <= set(msp.SESSIONS)


def test_developing_session_range_no_hindsight():
    df = _make_df(300)
    sess = msp.segment_sessions(df.index, msp.SessionConfig())
    out = msp.compute_developing_session_range_features(df, sess)
    assert "sess_dev_range" in out.columns
    # truncation invariance at a timestamp (no future leakage)
    t = df.index[180]
    out_full = out.loc[t, "sess_dev_range"]
    out_trunc = msp.compute_developing_session_range_features(df.loc[:t], sess.loc[:t]).loc[t, "sess_dev_range"]
    assert np.isclose(out_full, out_trunc, equal_nan=True)


def test_train_walkforward_artifacts(tmp_path: Path, monkeypatch):
    df = _make_df(2000)
    h1 = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    d1 = df.resample("1d").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

    monkeypatch.setattr(msp, "load_ohlcv_bundle", lambda cfg: (df, h1, d1))
    cfg = msp.PipelineConfig(
        n_splits=2,
        test_size=300,
        embargo_bars=24,
        output_dir=str(tmp_path / "out"),
        min_session_train_rows=50,
    )
    res = msp.train_walkforward(cfg)
    manifest = Path(res["summary_csv"]).with_name("run_manifest.json")
    assert manifest.exists()
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    assert obj["n_folds_saved"] >= 1
    for d in obj["fold_dirs"]:
        p = Path(d)
        assert (p / "scaler_state.pkl").exists()
        assert (p / "shared_trunk.pkl").exists()
        assert (p / "session_heads.pkl").exists()
        assert (p / "calibrators.json").exists()
        assert (p / "thresholds.json").exists()
        assert (p / "dependence_matrix.csv").exists()
        assert (p / "portfolio_controller_state.json").exists()
        assert (p / "portfolio_decisions.csv").exists()
