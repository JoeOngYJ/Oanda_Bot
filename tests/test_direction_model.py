from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import DirectionModel, train_direction_model


def test_direction_model_outputs_long_short_probabilities():
    b, t, f, c = 7, 128, 9, 16
    seq = torch.randn(b, t, f)
    ctx = torch.randn(b, c)
    regime = torch.softmax(torch.randn(b, 4), dim=1)

    m = DirectionModel(seq_features=f, ctx_dim=c, regime_dim=4)
    p = m(seq, ctx, regime)

    assert p.shape == (b, 2)
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)
    assert torch.allclose(p.sum(dim=1), torch.ones(b), atol=1e-6)


def test_train_direction_model_ignores_minus_one_labels():
    rng = np.random.default_rng(123)
    b, t, f, c = 48, 128, 8, 16
    seq = rng.normal(size=(b, t, f)).astype(np.float32)
    ctx = rng.normal(size=(b, c)).astype(np.float32)
    regime = rng.uniform(size=(b, 4)).astype(np.float32)
    regime = regime / (regime.sum(axis=1, keepdims=True) + 1e-9)
    y = rng.integers(0, 2, size=b).astype(np.int64)
    y[:10] = -1  # ignored labels

    m = DirectionModel(seq_features=f, ctx_dim=c, regime_dim=4)
    stats = train_direction_model(m, seq, ctx, regime, y, epochs=3, lr=1e-3)
    assert "final_loss" in stats
    assert stats["valid_count"] == b - 10
