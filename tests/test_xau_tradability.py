from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_tradability as xt


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    close = pd.Series([2000.0, 2000.5, 2000.6, 2000.6, 2000.7, 2000.9, 2001.0, 2001.0], index=idx)
    high = close + pd.Series([0.2, 0.25, 0.0, 0.0, 0.2, 0.3, 0.3, 0.0], index=idx)
    low = close - pd.Series([0.2, 0.2, 0.0, 0.0, 0.18, 0.25, 0.25, 0.0], index=idx)
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "sf_realized_range": (high - low).abs(),
            "sf_spread_proxy": [2.0, 2.2, 2.4, 2.4, 2.1, 2.2, 2.0, 2.0],
            "sf_vol_scale_atr14": [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0],
            "session_bucket": ["asia", "asia", "asia", "pre_london", "london_open", "london_open", "ny_open", "ny_open"],
        },
        index=idx,
    )


def test_tradable_mask_is_entry_time_deterministic():
    df = _make_df()
    m1 = xt.build_tradable_mask(df)
    trunc = df.iloc[:-1].copy()
    m2 = xt.build_tradable_mask(trunc)
    pd.testing.assert_series_equal(m1["tradable_mask"].iloc[:-1], m2["tradable_mask"], check_names=False)
    pd.testing.assert_series_equal(m1["tradability_score"].iloc[:-1], m2["tradability_score"], check_names=False)


def test_tradability_summary_counts_consistent():
    df = _make_df()
    m = xt.build_tradable_mask(df)
    s = xt.summarize_tradability(m, session=df["session_bucket"])
    assert s["total_bars"] == len(df)
    assert s["retained_bars"] + s["excluded_bars"] == len(df)
    by = s["by_session"]
    assert int(sum(v["bars"] for v in by.values())) == len(df)
    assert np.isclose(sum(v["retained_bars"] for v in by.values()), s["retained_bars"])
