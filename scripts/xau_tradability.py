from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, col: str, fill: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(fill, index=df.index, dtype=float)


def compute_tradability_score(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> pd.Series:
    """Compute entry-time tradability score in [0, 1] using observable bar-state inputs."""
    c = dict(cfg or {})
    eps = float(c.get("eps", 1e-9))
    close = _series(df, "close", fill=np.nan).abs().clip(lower=eps)
    high = _series(df, "high", fill=np.nan)
    low = _series(df, "low", fill=np.nan)
    spread = _series(df, str(c.get("spread_col", "sf_spread_proxy")), fill=np.nan).abs()
    atr = _series(df, str(c.get("atr_col", "sf_vol_scale_atr14")), fill=np.nan).abs()
    bar_range = _series(df, str(c.get("range_col", "sf_realized_range")), fill=np.nan).abs()
    if bar_range.isna().all():
        bar_range = (high - low).abs()

    default_spread_bps = float(c.get("default_spread_bps", 2.0))
    spread = spread.fillna(default_spread_bps)
    spread_is_bps = bool(c.get("spread_is_bps", True))
    if spread_is_bps:
        spread_price = close * (spread * 1e-4)
    else:
        spread_price = spread.where(spread <= 0.5, close * (spread * 1e-4))
    spread_atr = spread_price / (atr + eps)
    spread_range = spread_price / (bar_range + eps)
    range_spread = bar_range / (spread_price + eps)
    flat_proxy = (bar_range <= float(c.get("flat_range_abs", 1e-8))).astype(float)
    stale_proxy = (_series(df, "close", fill=np.nan).diff().abs() <= float(c.get("stale_close_abs", 1e-10))).astype(float)

    s_spread_atr = np.exp(-spread_atr / max(eps, float(c.get("spread_atr_scale", 0.25))))
    s_spread_range = np.exp(-spread_range / max(eps, float(c.get("spread_range_scale", 0.35))))
    s_range_spread = 1.0 - np.exp(-range_spread / max(eps, float(c.get("range_spread_scale", 5.0))))
    s_flat = 1.0 - flat_proxy
    s_stale = 1.0 - stale_proxy
    score = (
        float(c.get("w_spread_atr", 0.30)) * s_spread_atr
        + float(c.get("w_spread_range", 0.30)) * s_spread_range
        + float(c.get("w_range_spread", 0.25)) * s_range_spread
        + float(c.get("w_flat", 0.10)) * s_flat
        + float(c.get("w_stale", 0.05)) * s_stale
    )
    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index, dtype=float, name="tradability_score")


def build_tradable_mask(df: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Deterministic tradability filter derived only from decision-time observables."""
    c = dict(cfg or {})
    eps = float(c.get("eps", 1e-9))
    close = _series(df, "close", fill=np.nan).abs().clip(lower=eps)
    high = _series(df, "high", fill=np.nan)
    low = _series(df, "low", fill=np.nan)
    spread = _series(df, str(c.get("spread_col", "sf_spread_proxy")), fill=np.nan).abs()
    atr = _series(df, str(c.get("atr_col", "sf_vol_scale_atr14")), fill=np.nan).abs()
    bar_range = _series(df, str(c.get("range_col", "sf_realized_range")), fill=np.nan).abs()
    if bar_range.isna().all():
        bar_range = (high - low).abs()

    default_spread_bps = float(c.get("default_spread_bps", 2.0))
    spread = spread.fillna(default_spread_bps)
    spread_is_bps = bool(c.get("spread_is_bps", True))
    if spread_is_bps:
        spread_price = close * (spread * 1e-4)
    else:
        spread_price = spread.where(spread <= 0.5, close * (spread * 1e-4))
    spread_atr = spread_price / (atr + eps)
    spread_range = spread_price / (bar_range + eps)
    range_spread = bar_range / (spread_price + eps)
    flat = bar_range <= float(c.get("flat_range_abs", 1e-8))
    stale = _series(df, "close", fill=np.nan).diff().abs() <= float(c.get("stale_close_abs", 1e-10))
    ok = (
        spread_atr <= float(c.get("max_spread_atr", 0.35))
    ) & (
        spread_range <= float(c.get("max_spread_range", 0.55))
    ) & (
        range_spread >= float(c.get("min_range_spread", 2.5))
    ) & (~flat) & (~stale)

    score = compute_tradability_score(
        df,
        {
            **c,
            "spread_col": c.get("spread_col", "sf_spread_proxy"),
            "atr_col": c.get("atr_col", "sf_vol_scale_atr14"),
            "range_col": c.get("range_col", "sf_realized_range"),
        },
    )
    min_score = float(c.get("min_tradability_score", 0.25))
    out = pd.DataFrame(index=df.index)
    out["tradable_mask"] = (ok & (score >= min_score)).fillna(False).astype(bool)
    out["tradability_score"] = score.astype(float)
    out["tradability_spread_atr"] = spread_atr.replace([np.inf, -np.inf], np.nan)
    out["tradability_spread_range"] = spread_range.replace([np.inf, -np.inf], np.nan)
    out["tradability_range_spread"] = range_spread.replace([np.inf, -np.inf], np.nan)
    out["tradability_flat_bar"] = flat.fillna(True).astype(bool)
    out["tradability_stale_bar"] = stale.fillna(True).astype(bool)
    return out


def summarize_tradability(mask_df: pd.DataFrame, session: pd.Series | None = None) -> Dict[str, Any]:
    """Summarize retained/excluded counts globally and by session."""
    if "tradable_mask" not in mask_df.columns:
        raise ValueError("mask_df must include 'tradable_mask'.")
    m = mask_df["tradable_mask"].astype(bool)
    total = int(len(m))
    retained = int(m.sum())
    out: Dict[str, Any] = {
        "total_bars": total,
        "retained_bars": retained,
        "excluded_bars": int(total - retained),
        "retained_pct": 0.0 if total <= 0 else float(retained / total),
    }
    if session is not None:
        s = session.astype(str)
        by_session: Dict[str, Dict[str, float | int]] = {}
        for name, g in pd.DataFrame({"m": m, "s": s}).groupby("s", sort=True):
            n = int(len(g))
            r = int(g["m"].sum())
            by_session[str(name)] = {
                "bars": n,
                "retained_bars": r,
                "excluded_bars": int(n - r),
                "retained_pct": 0.0 if n <= 0 else float(r / n),
            }
        out["by_session"] = by_session
    return out
