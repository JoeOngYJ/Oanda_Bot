from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


VALID_SESSIONS = (
    "asia",
    "pre_london",
    "london_open",
    "london_continuation",
    "ny_open",
    "ny_overlap_postdata",
)


@dataclass(frozen=True)
class LabelConfig:
    tp_mult_by_session: Dict[str, float]
    sl_mult_by_session: Dict[str, float]
    horizon_by_session: Dict[str, int]
    neutral_band_by_session: Dict[str, float]
    vol_col: str = "atr14"
    event_col: str = "event_window_flag"
    event_mode: str = "exclude"  # exclude|suspend|event_sleeve|include
    spread_col: Optional[str] = "spread_proxy_bps"
    slippage_bps: float = 0.5
    commission_bps_per_side: float = 0.0
    spread_is_bps: bool = True
    max_spread_for_exec: Optional[float] = None
    min_net_edge: float = 0.0
    max_mae_mult: float = 1.5
    time_bucket_edges: Iterable[int] = field(default_factory=lambda: (4, 8, 16, 32))


def _require(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {ctx}: {miss}")


def _session_get(mapping: Dict[str, float | int], session: str, key: str) -> float:
    if session not in mapping:
        raise ValueError(f"Missing {key} for session '{session}'.")
    return float(mapping[session])


def _time_to_bucket(h: int, edges: Iterable[int]) -> str:
    e = [int(x) for x in edges]
    for x in e:
        if h <= x:
            return f"<= {x}"
    return f"> {e[-1]}"


def _triple_barrier_row(
    close_t: float,
    vol_t: float,
    f_high: np.ndarray,
    f_low: np.ndarray,
    f_close: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    neutral_band: float,
) -> tuple[int, int]:
    if not np.isfinite(close_t) or not np.isfinite(vol_t) or vol_t <= 0:
        return 0, horizon
    h = min(int(horizon), len(f_close))
    if h <= 0:
        return 0, 0

    up = close_t + (tp_mult * vol_t)
    dn = close_t - (sl_mult * vol_t)
    highs = f_high[:h]
    lows = f_low[:h]

    up_hits = np.where(highs >= up)[0]
    dn_hits = np.where(lows <= dn)[0]
    first_up = int(up_hits[0]) if len(up_hits) else None
    first_dn = int(dn_hits[0]) if len(dn_hits) else None

    if first_up is not None and first_dn is not None:
        if first_up < first_dn:
            return 1, first_up + 1
        if first_dn < first_up:
            return -1, first_dn + 1
        return 0, first_up + 1
    if first_up is not None:
        return 1, first_up + 1
    if first_dn is not None:
        return -1, first_dn + 1

    terminal_ret = (float(f_close[h - 1]) - close_t) / (abs(close_t) + 1e-9)
    if abs(terminal_ret) <= neutral_band:
        return 0, h
    return (1 if terminal_ret > 0 else -1), h


def build_meta_exec_label(
    directional_label: pd.Series,
    net_edge: pd.Series,
    mae_mult: pd.Series,
    spread: Optional[pd.Series],
    min_net_edge: float,
    max_mae_mult: float,
    max_spread_for_exec: Optional[float],
) -> pd.Series:
    """Build separate execution-quality meta label conditional on directional candidate."""

    y = pd.to_numeric(directional_label, errors="coerce").fillna(0).astype(int)
    edge_ok = pd.to_numeric(net_edge, errors="coerce").fillna(-np.inf) > float(min_net_edge)
    mae_ok = pd.to_numeric(mae_mult, errors="coerce").fillna(np.inf) <= float(max_mae_mult)
    if spread is None:
        spread_ok = pd.Series(True, index=y.index)
    else:
        s = pd.to_numeric(spread, errors="coerce")
        if max_spread_for_exec is None:
            spread_ok = pd.Series(True, index=y.index)
        else:
            spread_ok = s <= float(max_spread_for_exec)
    out = ((y != 0) & edge_ok & mae_ok & spread_ok).astype(int)
    return out


def build_session_conditioned_labels(df: pd.DataFrame, label_config: LabelConfig) -> pd.DataFrame:
    """Session-conditioned triple-barrier labels with optional meta execution labels."""

    _require(df, ["close", "high", "low", "session_bucket", label_config.vol_col], "build_session_conditioned_labels")
    if label_config.event_col not in df.columns and label_config.event_mode != "include":
        raise ValueError(f"Event column '{label_config.event_col}' not found.")

    x = df.copy()
    c = pd.to_numeric(x["close"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    v = pd.to_numeric(x[label_config.vol_col], errors="coerce")
    s = x["session_bucket"].astype(str)

    n = len(x)
    y_dir = np.zeros(n, dtype=int)
    t_res = np.zeros(n, dtype=int)
    label_state = np.array(["base"] * n, dtype=object)

    event_mask = (
        pd.to_numeric(x.get(label_config.event_col, pd.Series(index=x.index, data=0.0)), errors="coerce")
        .fillna(0.0)
        .astype(float)
        > 0.0
    )
    if label_config.event_mode == "exclude":
        label_state[event_mask.to_numpy()] = "excluded_event"
    elif label_config.event_mode == "suspend":
        label_state[event_mask.to_numpy()] = "suspended_event"
    elif label_config.event_mode == "event_sleeve":
        label_state[event_mask.to_numpy()] = "event_sleeve"
    elif label_config.event_mode == "include":
        pass
    else:
        raise ValueError(f"Unsupported event_mode '{label_config.event_mode}'.")

    highs = h.to_numpy(dtype=float)
    lows = l.to_numpy(dtype=float)
    closes = c.to_numpy(dtype=float)
    vols = v.to_numpy(dtype=float)
    sess = s.to_numpy(dtype=object)

    for i in range(n):
        if label_state[i] == "excluded_event":
            continue
        if label_state[i] == "suspended_event":
            y_dir[i] = 0
            t_res[i] = 0
            continue
        si = str(sess[i])
        if si not in VALID_SESSIONS:
            y_dir[i] = 0
            t_res[i] = 0
            continue
        tp = _session_get(label_config.tp_mult_by_session, si, "tp_mult_by_session")
        sl = _session_get(label_config.sl_mult_by_session, si, "sl_mult_by_session")
        hz = int(_session_get(label_config.horizon_by_session, si, "horizon_by_session"))
        nb = _session_get(label_config.neutral_band_by_session, si, "neutral_band_by_session")
        y, tr = _triple_barrier_row(
            close_t=float(closes[i]),
            vol_t=float(vols[i]),
            f_high=highs[i + 1 :],
            f_low=lows[i + 1 :],
            f_close=closes[i + 1 :],
            tp_mult=tp,
            sl_mult=sl,
            horizon=hz,
            neutral_band=nb,
        )
        y_dir[i] = int(y)
        t_res[i] = int(tr)

    # Path quality and net edge for meta labeling (direction-conditional).
    mae_mult = np.full(n, np.nan, dtype=float)
    net_edge = np.full(n, np.nan, dtype=float)
    spread = pd.to_numeric(x[label_config.spread_col], errors="coerce") if label_config.spread_col and label_config.spread_col in x.columns else None
    spread_s = spread if spread is not None else pd.Series(index=x.index, data=0.0, dtype=float)

    for i in range(n):
        yi = int(y_dir[i])
        if yi == 0:
            mae_mult[i] = np.nan
            net_edge[i] = -np.inf
            continue
        si = str(sess[i])
        hz = int(_session_get(label_config.horizon_by_session, si, "horizon_by_session"))
        hwin = min(hz, n - (i + 1))
        if hwin <= 0:
            mae_mult[i] = np.nan
            net_edge[i] = -np.inf
            continue
        path_h = highs[i + 1 : i + 1 + hwin]
        path_l = lows[i + 1 : i + 1 + hwin]
        path_c = closes[i + 1 : i + 1 + hwin]
        c0 = closes[i]
        vol0 = max(abs(vols[i]), 1e-9)
        if yi > 0:
            mae = max(0.0, c0 - np.nanmin(path_l))
            realized = float(path_c[-1] - c0)
        else:
            mae = max(0.0, np.nanmax(path_h) - c0)
            realized = float(c0 - path_c[-1])
        mae_mult[i] = mae / vol0
        spread_v = float(spread_s.iloc[i]) if np.isfinite(spread_s.iloc[i]) else 0.0
        if label_config.spread_is_bps:
            spread_cost = abs(c0) * (max(0.0, spread_v) * 1e-4)
        else:
            spread_cost = max(0.0, spread_v)
        slip_cost = abs(c0) * (max(0.0, float(label_config.slippage_bps)) * 1e-4)
        comm_cost = abs(c0) * (max(0.0, float(label_config.commission_bps_per_side)) * 2.0 * 1e-4)
        total_cost = spread_cost + slip_cost + comm_cost
        net_edge[i] = realized - total_cost

    y_meta = build_meta_exec_label(
        directional_label=pd.Series(y_dir, index=x.index),
        net_edge=pd.Series(net_edge, index=x.index),
        mae_mult=pd.Series(mae_mult, index=x.index),
        spread=spread,
        min_net_edge=label_config.min_net_edge,
        max_mae_mult=label_config.max_mae_mult,
        max_spread_for_exec=label_config.max_spread_for_exec,
    )

    out = pd.DataFrame(index=x.index)
    out["y_dir"] = pd.Series(y_dir, index=x.index).astype(int)
    out["y_meta_exec"] = pd.Series(y_meta, index=x.index).astype(int)
    out["time_to_resolution_bucket"] = pd.Series(
        [_time_to_bucket(int(z), label_config.time_bucket_edges) if int(z) > 0 else "none" for z in t_res],
        index=x.index,
        dtype="object",
    )
    out["label_state"] = pd.Series(label_state, index=x.index, dtype="object")
    out["label_horizon_bars"] = pd.Series(t_res, index=x.index).astype(int)

    if label_config.event_mode == "exclude":
        m = out["label_state"].eq("excluded_event")
        out.loc[m, ["y_dir", "y_meta_exec"]] = np.nan
        out.loc[m, "time_to_resolution_bucket"] = "excluded"
        out.loc[m, "label_horizon_bars"] = 0
    return out


def compute_session_class_weights(labels_df: pd.DataFrame) -> dict:
    """Compute inverse-frequency class weights within each session."""

    _require(labels_df, ["session_bucket", "y_dir"], "compute_session_class_weights")
    out: dict = {}
    x = labels_df.copy()
    x = x.dropna(subset=["y_dir"])
    x["y_dir"] = pd.to_numeric(x["y_dir"], errors="coerce").astype(int)
    for sess, g in x.groupby("session_bucket", sort=True):
        counts = g["y_dir"].value_counts().to_dict()
        total = float(sum(counts.values()))
        cls_weights: dict[int, float] = {}
        for cls in (-1, 0, 1):
            cnt = float(counts.get(cls, 0.0))
            cls_weights[cls] = 0.0 if cnt <= 0 else total / (3.0 * cnt)
        out[str(sess)] = cls_weights
    return out
