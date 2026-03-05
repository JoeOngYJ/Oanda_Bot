from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import MeanReversionModel, make_mean_reversion_labels, train_mean_reversion_model


def test_make_mean_reversion_labels_rule_with_range_gate():
    idx = pd.date_range("2024-01-01 12:00:00", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "price_zscore": [2.0, -2.2, 0.5, 1.8, -1.7, 2.1, -2.0, 0.4, 1.6, -1.6],
            "net_ret": [-0.004, 0.005, 0.001, -0.003, 0.0025, -0.0005, 0.004, 0.0, -0.002, 0.003],
            "y_opportunity": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "regime": [1, 1, 1, 0, 1, 1, 2, 1, 1, 1],  # row3 not range, row6 highvol
        },
        index=idx,
    )
    out = make_mean_reversion_labels(df, horizon_bars=2, z_k=1.5, net_ret_threshold=0.002)
    assert "y_mean_reversion" in out.columns

    # row0: extreme positive z + negative net_ret + range -> 1
    assert float(out.iloc[0]["y_mean_reversion"]) == 1.0
    # row3: meets sign condition but non-range regime -> 0
    assert float(out.iloc[3]["y_mean_reversion"]) == 0.0
    # last horizon rows are NaN
    assert out.iloc[-2:]["y_mean_reversion"].isna().all()


def test_mean_reversion_model_train_bce_runs():
    rng = np.random.default_rng(202)
    b, t, f = 64, 128, 10
    seq = rng.normal(size=(b, t, f)).astype(np.float32)
    y = rng.integers(0, 2, size=b).astype(np.float32)

    model = MeanReversionModel(seq_features=f)
    stats = train_mean_reversion_model(model, seq, y, epochs=3, lr=1e-3)
    assert "final_loss" in stats
    assert np.isfinite(float(stats["final_loss"]))

    p = model(torch.as_tensor(seq[:8], dtype=torch.float32))
    assert p.shape == (8,)
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)
