from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import (
    RegimeMLP,
    make_regime_targets,
    train_regime_mlp,
    predict_regime_proba,
)


def test_make_regime_targets_rules_and_classes():
    idx = pd.date_range("2024-01-01 12:00:00", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "h1_slope": [0.0012, 0.00005, 0.0003, 0.0010, 0.0001, 0.0003],
            "h1_adx": [0.35, 0.10, 0.20, 0.30, 0.12, 0.25],
            "h1_vol_pct": [0.40, 0.30, 0.95, 0.20, 0.25, 0.50],
            "spread_est_pct": [0.20, 0.20, 0.30, 0.20, 0.90, 0.20],
        },
        index=idx,
    )
    y = make_regime_targets(df)
    # trend / range / highvol / lowliq classes should all appear in this setup.
    assert set(y.unique().tolist()) <= {0, 1, 2, 3}
    assert 0 in set(y.values.tolist())
    assert 1 in set(y.values.tolist())
    assert 2 in set(y.values.tolist())
    assert 3 in set(y.values.tolist())


def test_regime_mlp_train_and_predict_softmax_shape():
    rng = np.random.default_rng(7)
    n, c = 120, 16
    x = rng.normal(0.0, 1.0, size=(n, c)).astype(np.float32)
    y = rng.integers(0, 4, size=n, dtype=np.int64)

    model = RegimeMLP(ctx_dim=16, hidden=32, out=4, dropout=0.1)
    stats = train_regime_mlp(model, x, y, epochs=5, lr=1e-3)
    assert "final_loss" in stats

    probs = predict_regime_proba(model, x[:10])
    assert probs.shape == (10, 4)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
