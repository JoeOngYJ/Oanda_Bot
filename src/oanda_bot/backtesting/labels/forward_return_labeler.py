from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    df: pd.DataFrame,
    horizon_bars: int = 8,
    no_trade_band: float = 0.0,
    use_costs: bool = True,
) -> pd.DataFrame:
    """Generate forward-return labels for opportunity and direction.

    Definitions:
    - gross_ret = close[t+h] - close[t]
    - net_ret = gross_ret - cost_est (if use_costs)
    - y_opportunity = 1 if abs(net_ret) > no_trade_band else 0
    - y_direction = 1 if net_ret > 0 else 0, only where y_opportunity == 1
      (set NaN when opportunity is 0)
    """
    if "close" not in df.columns:
        raise ValueError("make_labels requires 'close' column.")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be > 0.")
    if no_trade_band < 0:
        raise ValueError("no_trade_band must be >= 0.")

    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    gross_ret = close.shift(-horizon_bars) - close

    if use_costs:
        if "cost_est" not in out.columns:
            raise ValueError("use_costs=True requires 'cost_est' column.")
        cost = pd.to_numeric(out["cost_est"], errors="coerce").astype(float)
        net_ret = gross_ret - cost
    else:
        net_ret = gross_ret.copy()

    opportunity = (net_ret.abs() > float(no_trade_band)).astype(float)
    direction = pd.Series(np.where(net_ret > 0.0, 1.0, 0.0), index=out.index, dtype=float)
    direction = direction.where(opportunity == 1.0, np.nan)

    out["gross_ret"] = gross_ret
    out["net_ret"] = net_ret
    out["y_opportunity"] = opportunity
    out["y_direction"] = direction

    if len(out) > 0:
        tail = out.index[-min(horizon_bars, len(out)):]
        out.loc[tail, ["gross_ret", "net_ret", "y_opportunity", "y_direction"]] = np.nan
    return out
