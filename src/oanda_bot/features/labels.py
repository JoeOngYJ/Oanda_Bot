from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    df: pd.DataFrame,
    horizon_bars: int = 8,
    no_trade_band: float = 0.0,
    use_costs: bool = True,
) -> pd.DataFrame:
    """Create horizon labels with optional execution-cost adjustment.

    Output columns:
    - gross_ret: forward return over horizon
    - net_ret: gross_ret minus cost_est (when use_costs=True)
    - y_opportunity: 1 if abs(net_ret) > no_trade_band else 0
    - y_direction: 1 (long), -1 (short), 0 (no-trade-band)

    Last ``horizon_bars`` rows are marked NaN for label columns because
    future data is not available.
    """
    if "close" not in df.columns:
        raise ValueError("make_labels requires 'close' column.")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be > 0.")
    if no_trade_band < 0:
        raise ValueError("no_trade_band must be >= 0.")

    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    gross_ret = (close.shift(-horizon_bars) / close) - 1.0

    if use_costs:
        if "cost_est" not in out.columns:
            raise ValueError("use_costs=True requires 'cost_est' column.")
        cost = pd.to_numeric(out["cost_est"], errors="coerce").astype(float)
        net_ret = gross_ret - cost
    else:
        net_ret = gross_ret.copy()

    y_opportunity = (net_ret.abs() > float(no_trade_band)).astype(float)
    y_direction = np.where(net_ret > float(no_trade_band), 1.0, np.where(net_ret < -float(no_trade_band), -1.0, 0.0))
    y_direction = pd.Series(y_direction, index=out.index, dtype=float)

    out["gross_ret"] = gross_ret
    out["net_ret"] = net_ret
    out["y_opportunity"] = y_opportunity
    out["y_direction"] = y_direction

    # Last horizon bars have no future outcome.
    if len(out) > 0:
        tail_idx = out.index[-min(horizon_bars, len(out)):]
        out.loc[tail_idx, ["gross_ret", "net_ret", "y_opportunity", "y_direction"]] = np.nan

    return out
