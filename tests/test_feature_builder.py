from __future__ import annotations

import numpy as np
import pandas as pd

from oanda_bot.features import FeatureBuilder


def _make_ohlcv(n: int, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    ret = rng.normal(0.0, 0.001, size=n)
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0015, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(100, 2000, size=n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_feature_builder_returns_expected_shapes_and_meta():
    m15 = _make_ohlcv(1400, "15min", seed=11)
    m15["spread_c"] = np.linspace(0.01, 0.08, len(m15))
    h1 = _make_ohlcv(600, "1h", seed=22)
    h4 = _make_ohlcv(300, "4h", seed=33)

    fb = FeatureBuilder(seq_len=128)
    seq, ctx, meta = fb.build(m15, h1, h4, instrument="XAU_USD")

    assert seq.shape[0] == 128
    assert seq.dtype == np.float32
    assert seq.shape[1] >= 10

    assert ctx.ndim == 1
    assert ctx.dtype == np.float32
    assert len(ctx) == 11

    assert meta["instrument"] == "XAU_USD"
    assert isinstance(meta["datetime_index"], pd.DatetimeIndex)
    assert len(meta["datetime_index"]) == 128
    assert isinstance(meta["close"], float)
    assert isinstance(meta["atr"], float)


def test_m15_feature_frame_is_truncation_invariant_at_timestamp():
    m15 = _make_ohlcv(1500, "15min", seed=44)
    m15["spread_c"] = np.linspace(0.02, 0.05, len(m15))

    fb = FeatureBuilder(seq_len=128)
    t_aware = m15.index[1200]
    t = t_aware.tz_convert(None)
    full = fb._build_m15_feature_frame(fb._normalize_ohlcv_index(m15))
    trunc = fb._build_m15_feature_frame(fb._normalize_ohlcv_index(m15.loc[:t_aware]))

    cols = ["ret_1", "atr_14", "rsi_14", "adx_14", "bb_width_pct", "atr_pct", "spread_feat"]
    for c in cols:
        assert np.isclose(full.loc[t, c], trunc.loc[t, c], equal_nan=True)
