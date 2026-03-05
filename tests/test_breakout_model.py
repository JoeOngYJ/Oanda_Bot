from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import BreakoutModel, make_breakout_labels, train_breakout_model


def test_make_breakout_labels_rule_and_tail_nan():
    idx = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "bb_width_pct": [0.10, 0.15, 0.25, 0.10, 0.05, 0.30, 0.10, 0.10, 0.05, 0.12, 0.10, 0.09],
            "net_ret": [0.003, 0.0005, 0.004, -0.0035, 0.001, 0.005, -0.003, 0.0001, 0.006, -0.006, 0.002, 0.004],
            "y_opportunity": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    out = make_breakout_labels(df, horizon_bars=2, compression_pct_threshold=0.2, net_ret_threshold=0.002)
    assert "y_breakout" in out.columns
    # row0: compressed + |ret|>thr + opp=1 -> breakout
    assert float(out.iloc[0]["y_breakout"]) == 1.0
    # row1: compressed but |ret|<=thr -> no breakout
    assert float(out.iloc[1]["y_breakout"]) == 0.0
    # last horizon rows are NaN
    assert out.iloc[-2:]["y_breakout"].isna().all()


def test_breakout_model_train_bce_runs():
    rng = np.random.default_rng(99)
    b, t, f = 64, 128, 10
    seq = rng.normal(size=(b, t, f)).astype(np.float32)
    y = rng.integers(0, 2, size=b).astype(np.float32)

    model = BreakoutModel(seq_features=f)
    stats = train_breakout_model(model, seq, y, epochs=3, lr=1e-3)
    assert "final_loss" in stats
    assert np.isfinite(float(stats["final_loss"]))

    p = model(torch.as_tensor(seq[:8], dtype=torch.float32))
    assert p.shape == (8,)
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)
