from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oanda_bot.features import make_labels


def test_make_labels_computes_gross_and_net_returns_and_labels():
    idx = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    close = np.linspace(100.0, 112.0, len(idx))
    df = pd.DataFrame({"close": close, "cost_est": 0.002}, index=idx)

    out = make_labels(df, horizon_bars=2, no_trade_band=0.005, use_costs=True)
    for c in ["gross_ret", "net_ret", "y_opportunity", "y_direction"]:
        assert c in out.columns

    i = out.index[0]
    expected_gross = (close[2] / close[0]) - 1.0
    assert np.isclose(float(out.loc[i, "gross_ret"]), expected_gross)
    assert np.isclose(float(out.loc[i, "net_ret"]), expected_gross - 0.002)
    assert float(out.loc[i, "y_opportunity"]) == 1.0
    assert float(out.loc[i, "y_direction"]) == 1.0


def test_make_labels_marks_tail_rows_nan():
    idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx)), "cost_est": 0.0}, index=idx)
    out = make_labels(df, horizon_bars=3, no_trade_band=0.0, use_costs=True)
    tail = out.iloc[-3:]
    assert tail[["gross_ret", "net_ret", "y_opportunity", "y_direction"]].isna().all().all()


def test_make_labels_requires_cost_est_when_use_costs_true():
    idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(1.0, 1.2, len(idx))}, index=idx)
    with pytest.raises(ValueError, match="cost_est"):
        make_labels(df, horizon_bars=2, no_trade_band=0.0, use_costs=True)
