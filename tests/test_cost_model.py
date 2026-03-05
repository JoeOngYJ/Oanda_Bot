from __future__ import annotations

import numpy as np
import pandas as pd

from oanda_bot.features import CostModel, SpreadTable


def _make_df(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 1.10 + np.linspace(0.0, 0.02, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.001
    low = np.minimum(open_, close) - 0.001
    volume = np.linspace(100.0, 1000.0, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_add_cost_columns_uses_spread_c_when_present():
    df = _make_df()
    df["spread_c"] = 0.00021
    cm = CostModel(min_slip=0.00005, alpha=0.10, commission=0.00003)

    out = cm.add_cost_columns(df, instrument="EUR_USD")
    for c in ["atr", "spread_est", "slippage_est", "cost_est"]:
        assert c in out.columns

    mask = out["atr"].notna()
    assert np.allclose(out.loc[mask, "spread_est"].to_numpy(dtype=float), 0.00021)
    assert np.allclose(
        out.loc[mask, "cost_est"].to_numpy(dtype=float),
        out.loc[mask, "spread_est"].to_numpy(dtype=float)
        + out.loc[mask, "slippage_est"].to_numpy(dtype=float)
        + 0.00003,
    )


def test_add_cost_columns_falls_back_to_spread_table():
    df = _make_df()
    st = SpreadTable(
        table={
            "*": {
                "tokyo": {"low": 0.00030, "mid": 0.00031, "high": 0.00032},
                "london": {"low": 0.00010, "mid": 0.00011, "high": 0.00012},
                "newyork": {"low": 0.00013, "mid": 0.00014, "high": 0.00015},
                "overlap": {"low": 0.00008, "mid": 0.00009, "high": 0.00010},
                "offhours": {"low": 0.00040, "mid": 0.00041, "high": 0.00042},
            }
        }
    )
    cm = CostModel(spread_table=st, min_slip=0.00002, alpha=0.0, commission=0.00001)
    out = cm.add_cost_columns(df, instrument="EUR_USD")

    assert {"atr", "spread_est", "slippage_est", "cost_est"} <= set(out.columns)
    assert (out["spread_est"] > 0.0).all()
    assert np.allclose(out["slippage_est"].to_numpy(dtype=float), 0.00002)
    assert np.allclose(
        out["cost_est"].to_numpy(dtype=float),
        out["spread_est"].to_numpy(dtype=float) + 0.00002 + 0.00001,
    )
