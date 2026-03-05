from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oanda_bot.features import FeatureLabelDataset, feature_label_collate_fn, CostModel


def _make_ohlcv(n: int, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    ret = rng.normal(0.0, 0.001, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0012, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = rng.integers(100, 5000, size=n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_dataset_on_the_fly_returns_expected_sample_and_collate():
    m15 = _make_ohlcv(1500, "15min", 10)
    h1 = _make_ohlcv(800, "1h", 11)
    h4 = _make_ohlcv(400, "4h", 12)
    m15 = CostModel(min_slip=0.00001, alpha=0.01, commission=0.00001).add_cost_columns(m15, "XAU_USD")

    ds = FeatureLabelDataset(
        instrument="XAU_USD",
        m15_df=m15,
        h1_df=h1,
        h4_df=h4,
        label_kwargs={"horizon_bars": 8, "no_trade_band": 0.0001, "use_costs": True},
    )
    assert len(ds) > 0

    sample = ds[0]
    assert sample["seq"].shape[0] == 128
    assert sample["seq"].ndim == 2
    assert sample["ctx"].ndim == 1
    assert sample["y_direction"].item() in {-1, 0, 1}
    assert isinstance(sample["meta"], dict)

    batch = feature_label_collate_fn([ds[0], ds[1], ds[2]])
    assert batch["seq"].shape[0] == 3
    assert batch["ctx"].shape[0] == 3
    assert len(batch["meta"]) == 3


def test_dataset_precomputed_dataframe_mode():
    pre = pd.DataFrame(
        {
            "seq": [np.ones((128, 4), dtype=np.float32), np.zeros((128, 4), dtype=np.float32)],
            "ctx": [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32)],
            "y_opportunity": [1, 0],
            "y_direction": [1, np.nan],
            "meta": [{"instrument": "EUR_USD"}, {"instrument": "EUR_USD"}],
        }
    )
    ds = FeatureLabelDataset(precomputed_df=pre)
    assert len(ds) == 2

    a = ds[0]
    b = ds[1]
    assert a["seq"].shape == (128, 4)
    assert b["y_direction"].item() == -1
