from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import RiskModel, make_risk_labels, train_risk_model


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 1.10 + np.cumsum(np.linspace(-0.0005, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.001
    low = np.minimum(open_, close) - 0.001
    volume = np.linspace(100.0, 1000.0, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_make_risk_labels_realized_vol_and_tail_nan():
    df = _make_ohlcv()
    out = make_risk_labels(df, horizon_bars=8, method="realized_vol")
    assert "y_risk" in out.columns
    assert out["y_risk"].iloc[:-8].notna().any()
    assert out["y_risk"].iloc[-8:].isna().all()


def test_make_risk_labels_future_atr_mode():
    df = _make_ohlcv()
    out = make_risk_labels(df, horizon_bars=6, method="future_atr")
    assert "y_risk" in out.columns
    assert "atr" in out.columns
    assert out["y_risk"].iloc[:-6].notna().any()
    assert out["y_risk"].iloc[-6:].isna().all()


def test_train_risk_model_huber_with_nan_mask():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(80, 32)).astype(np.float32)
    y = rng.normal(loc=0.01, scale=0.005, size=80).astype(np.float32)
    y[:12] = np.nan

    model = RiskModel(in_dim=32, dropout=0.1)
    stats = train_risk_model(model, x, y, epochs=3, lr=1e-3)
    assert "final_loss" in stats
    assert np.isfinite(float(stats["final_loss"]))
    assert stats["valid_count"] == 68
