from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import OpportunityModel, train_opportunity_model


def test_opportunity_model_forward_shape_and_range():
    b, t, f, c = 8, 128, 12, 16
    seq = torch.randn(b, t, f)
    ctx = torch.randn(b, c)
    regime = torch.softmax(torch.randn(b, 4), dim=1)

    m = OpportunityModel(seq_features=f, ctx_dim=c, regime_dim=4)
    p = m(seq, ctx, regime)

    assert p.shape == (b,)
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)


def test_train_opportunity_model_runs_bce():
    rng = np.random.default_rng(42)
    b, t, f, c = 64, 128, 10, 16
    seq = rng.normal(size=(b, t, f)).astype(np.float32)
    ctx = rng.normal(size=(b, c)).astype(np.float32)
    regime = rng.uniform(size=(b, 4)).astype(np.float32)
    regime = regime / (regime.sum(axis=1, keepdims=True) + 1e-9)
    y = rng.integers(0, 2, size=b).astype(np.float32)

    m = OpportunityModel(seq_features=f, ctx_dim=c, regime_dim=4)
    stats = train_opportunity_model(m, seq, ctx, regime, y, epochs=3, lr=1e-3)

    assert "final_loss" in stats
    assert np.isfinite(float(stats["final_loss"]))
