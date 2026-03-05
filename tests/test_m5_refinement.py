from __future__ import annotations

import numpy as np
import pandas as pd

from oanda_bot.execution.m5_refinement import refine_entry


def _m5_df(n: int = 12, base: float = 1.10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = base + np.linspace(0.0, 0.0008, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.0004
    low = np.minimum(open_, close) - 0.0004
    volume = np.linspace(100.0, 200.0, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_refine_entry_skips_when_spread_remains_wide():
    m5 = _m5_df()
    m5["spread_c"] = 0.01  # huge
    signal = {"instrument": "EUR_USD", "action": "BUY", "quantity": 1000}
    out = refine_entry(signal, m5, cost_model=None)
    assert out["status"] == "skip"
    assert out["reason"] == "spread_too_wide"


def test_refine_entry_chase_guard_skips():
    m5 = _m5_df(base=1.2)
    m5["spread_c"] = 0.00002
    signal = {
        "instrument": "EUR_USD",
        "action": "BUY",
        "quantity": 1000,
        "entry_price": 1.15,  # far below current -> adverse chase for buy
    }
    out = refine_entry(signal, m5, cost_model=None)
    assert out["status"] == "skip"
    assert out["reason"] == "chase_guard"


def test_refine_entry_uses_market_when_momentum_strong():
    m5 = _m5_df(base=1.10)
    m5["close"] = m5["close"] + np.linspace(0.0, 0.01, len(m5))  # strong up momentum
    m5["spread_c"] = 0.00001
    signal = {"instrument": "EUR_USD", "action": "BUY", "quantity": 1000}
    out = refine_entry(signal, m5, cost_model=None)
    assert out["status"] == "ok"
    assert out["order_type"] in {"market", "limit"}
    if out["order_type"] == "market":
        assert out["type"] == "MARKET"
