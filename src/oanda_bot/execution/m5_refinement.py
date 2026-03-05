from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _as_dict(signal: Any) -> Dict[str, Any]:
    if isinstance(signal, dict):
        return signal
    out: Dict[str, Any] = {}
    for k in [
        "instrument",
        "side",
        "action",
        "quantity",
        "entry_price",
        "atr",
        "signal_id",
        "metadata",
    ]:
        if hasattr(signal, k):
            out[k] = getattr(signal, k)
    return out


def _norm_side(sig: Dict[str, Any]) -> str:
    action = str(sig.get("action", "")).upper()
    side = str(sig.get("side", "")).lower()
    if action == "BUY" or side == "buy":
        return "buy"
    if action == "SELL" or side == "sell":
        return "sell"
    raise ValueError("Signal must include side/action as BUY/SELL.")


def _to_mid(df: pd.DataFrame) -> pd.Series:
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce").astype(float)
    if {"bid_c", "ask_c"} <= set(df.columns):
        b = pd.to_numeric(df["bid_c"], errors="coerce").astype(float)
        a = pd.to_numeric(df["ask_c"], errors="coerce").astype(float)
        return (a + b) / 2.0
    raise ValueError("m5_df must include 'close' or ('bid_c','ask_c').")


def _atr14(df: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(df["high"], errors="coerce").astype(float)
    l = pd.to_numeric(df["low"], errors="coerce").astype(float)
    c = _to_mid(df)
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()


def _spread_series(df: pd.DataFrame, cost_model: Any, instrument: str) -> pd.Series:
    if "spread_c" in df.columns:
        return pd.to_numeric(df["spread_c"], errors="coerce").astype(float)

    if cost_model is not None and hasattr(cost_model, "add_cost_columns"):
        enriched = cost_model.add_cost_columns(df.copy(), instrument)
        if "spread_est" in enriched.columns:
            return pd.to_numeric(enriched["spread_est"], errors="coerce").astype(float)

    close = _to_mid(df)
    return close.abs() * 0.0002


def refine_entry(signal: Any, m5_df: pd.DataFrame, cost_model: Any) -> Dict[str, Any]:
    """Refine entry timing and order type from M5 microstructure.

    Policy:
    - if spread too high: wait up to N M5 bars
    - avoid chasing if adverse move > x * ATR from planned entry
    - prefer limit near mid with timeout
    - fallback to market when momentum is strong in signal direction

    Returns an order-request dict compatible with execution payload conventions.
    """
    sig = _as_dict(signal)
    side = _norm_side(sig)
    instrument = str(sig.get("instrument", "EUR_USD"))
    quantity = int(sig.get("quantity", 0))
    if quantity <= 0:
        raise ValueError("signal.quantity must be > 0")

    x = m5_df.copy()
    x.index = pd.to_datetime(x.index, utc=True, errors="coerce")
    x = x.sort_index()
    if x.empty:
        return {
            "status": "skip",
            "reason": "empty_m5",
            "instrument": instrument,
            "side": side,
            "type": "NONE",
            "order_type": "none",
            "quantity": quantity,
        }

    mid = _to_mid(x)
    atr = _atr14(x)
    spread = _spread_series(x, cost_model, instrument)

    cfg = {
        "max_wait_bars": 3,
        "spread_atr_mult_threshold": 0.25,
        "spread_abs_threshold": None,  # optional absolute override
        "chase_atr_mult": 0.5,
        "limit_offset_spread_mult": 0.25,
        "limit_timeout_bars": 2,
        "momentum_lookback": 3,
        "momentum_atr_mult": 0.15,
    }
    cfg.update(dict(sig.get("metadata", {}).get("m5_refine", {})))

    planned_entry = float(sig.get("entry_price", np.nan))
    if not np.isfinite(planned_entry):
        planned_entry = float(mid.iloc[0])

    atr_vals = atr.to_numpy(dtype=float)
    finite_atr = atr_vals[np.isfinite(atr_vals)]
    first_atr = float(atr.iloc[0]) if np.isfinite(atr.iloc[0]) else (float(np.median(finite_atr)) if finite_atr.size else np.nan)
    if not np.isfinite(first_atr) or first_atr <= 0:
        first_atr = max(abs(float(mid.iloc[0])) * 0.001, 1e-6)

    spread_abs_thresh = cfg["spread_abs_threshold"]
    spread_atr_thresh = float(cfg["spread_atr_mult_threshold"]) * first_atr

    idx_sel = 0
    max_wait = int(max(0, cfg["max_wait_bars"]))
    for i in range(min(max_wait + 1, len(x))):
        s_i = float(spread.iloc[i]) if np.isfinite(spread.iloc[i]) else np.inf
        ok_abs = True if spread_abs_thresh is None else (s_i <= float(spread_abs_thresh))
        ok_rel = s_i <= spread_atr_thresh
        if ok_abs and ok_rel:
            idx_sel = i
            break
    else:
        return {
            "status": "skip",
            "reason": "spread_too_wide",
            "instrument": instrument,
            "side": side,
            "type": "NONE",
            "order_type": "none",
            "quantity": quantity,
            "debug": {"spread0": float(spread.iloc[0]), "spread_atr_thresh": spread_atr_thresh},
        }

    px = float(mid.iloc[idx_sel])
    atr_sel = float(atr.iloc[idx_sel]) if np.isfinite(atr.iloc[idx_sel]) else first_atr
    chase_mult = float(cfg["chase_atr_mult"])
    adverse_limit = chase_mult * atr_sel
    if side == "buy":
        adverse_move = px - planned_entry
    else:
        adverse_move = planned_entry - px
    if adverse_move > adverse_limit:
        return {
            "status": "skip",
            "reason": "chase_guard",
            "instrument": instrument,
            "side": side,
            "type": "NONE",
            "order_type": "none",
            "quantity": quantity,
            "debug": {"planned_entry": planned_entry, "current_mid": px, "adverse_move": adverse_move, "adverse_limit": adverse_limit},
        }

    look = int(max(1, cfg["momentum_lookback"]))
    start = max(0, idx_sel - look)
    mom = px - float(mid.iloc[start])
    mom_thr = float(cfg["momentum_atr_mult"]) * atr_sel
    strong_up = mom > mom_thr
    strong_dn = mom < -mom_thr
    momentum_strong_in_side = (side == "buy" and strong_up) or (side == "sell" and strong_dn)

    sp = float(spread.iloc[idx_sel]) if np.isfinite(spread.iloc[idx_sel]) else 0.0
    offset = float(cfg["limit_offset_spread_mult"]) * sp
    if side == "buy":
        limit_price = px - offset
    else:
        limit_price = px + offset

    if momentum_strong_in_side:
        order_type = "market"
        oanda_type = "MARKET"
        price = None
    else:
        order_type = "limit"
        oanda_type = "LIMIT"
        price = float(limit_price)

    return {
        "status": "ok",
        "instrument": instrument,
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "type": oanda_type,
        "price": price,
        "timeInForce": "GTD" if order_type == "limit" else "FOK",
        "limit_timeout_bars": int(cfg["limit_timeout_bars"]),
        "selected_bar_index": int(idx_sel),
        "debug": {
            "planned_entry": planned_entry,
            "selected_mid": px,
            "selected_spread": sp,
            "selected_atr": atr_sel,
            "momentum": mom,
            "momentum_threshold": mom_thr,
            "momentum_strong_in_side": bool(momentum_strong_in_side),
        },
    }
