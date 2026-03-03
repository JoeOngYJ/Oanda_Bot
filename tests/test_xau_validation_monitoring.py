from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.xau_validation_monitoring as xvm


def _make_df(n: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(9)
    latent = rng.normal(0.0, 1.0, size=n)
    y_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.uniform(size=n) < y_prob).astype(int)
    out = pd.DataFrame(index=idx)
    out["y_true"] = y
    out["score_asia"] = latent + rng.normal(0.0, 0.2, size=n)
    out["score_london"] = 0.8 * latent + rng.normal(0.0, 0.25, size=n)
    out["pnl_asia"] = (2 * y - 1) * (0.02 + 0.01 * rng.normal(size=n))
    out["pnl_london"] = (2 * y - 1) * (0.015 + 0.01 * rng.normal(size=n))
    out["label_end_ts"] = pd.Series(idx, index=idx).shift(-8)
    out["regime_bucket"] = np.where(latent > 0.5, "trend", np.where(latent < -0.5, "range", "transition"))
    return out


def test_walk_forward_reproducible(tmp_path: Path):
    df = _make_df()
    cfg = {
        "output_dir": str(tmp_path / "wfo"),
        "sleeves": ["asia", "london"],
        "n_splits": 3,
        "val_size": 120,
        "max_label_horizon_bars": 8,
        "min_train_size": 200,
    }
    r1 = xvm.run_walk_forward_pipeline(df, cfg)
    r2 = xvm.run_walk_forward_pipeline(df, cfg)
    assert r1["folds"] == r2["folds"]
    assert json.dumps(r1["portfolio_summary"], sort_keys=True, default=str) == json.dumps(r2["portfolio_summary"], sort_keys=True, default=str)


def test_sleeve_and_portfolio_metrics_separated(tmp_path: Path):
    df = _make_df()
    cfg = {"output_dir": str(tmp_path / "wfo2"), "sleeves": ["asia", "london"], "n_splits": 2, "val_size": 100, "max_label_horizon_bars": 8, "min_train_size": 180}
    r = xvm.run_walk_forward_pipeline(df, cfg)
    assert "sleeve_summary" in r and "portfolio_summary" in r
    assert isinstance(r["sleeve_summary"], list)
    assert isinstance(r["portfolio_summary"], dict)


def test_drift_metrics_update_on_distribution_shift():
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    ref = pd.DataFrame({"session_bucket": ["asia"] * n, "f1": np.linspace(0, 1, n), "prob": np.linspace(0.2, 0.8, n), "y_true": (np.linspace(0, 1, n) > 0.5).astype(int)}, index=idx)
    cur = ref.copy()
    cur["f1"] = cur["f1"] + 2.0
    cur["prob"] = np.clip(cur["prob"] + 0.1, 0, 1)
    cd = xvm.compute_calibration_drift(ref, cur)
    fd = xvm.compute_feature_drift(ref, cur, ["f1"])
    assert "asia" in cd
    assert fd["global"]["f1"] > 0.1


def test_leave_one_out_attribution():
    idx = pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC")
    a = pd.Series(0.02, index=idx)
    b = pd.Series(-0.01, index=idx)
    port = a + b
    m = xvm.evaluate_portfolio_metrics(port, {"a": a, "b": b})
    assert m["leave_one_out_contribution"]["a"] > 0
    assert m["leave_one_out_contribution"]["b"] < 0
