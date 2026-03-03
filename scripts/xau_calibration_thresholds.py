from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)
    acc = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        acc += (np.sum(m) / n) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return float(acc)


def _reliability_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    m = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[m]
    pp = pp[m]
    if len(yy) < 2:
        return np.nan, np.nan
    if float(np.std(pp, ddof=0)) == 0.0:
        return 0.0, float(np.mean(yy))
    x = np.column_stack([np.ones(len(pp)), pp])
    try:
        beta = np.linalg.lstsq(x, yy, rcond=None)[0]
        intercept, slope = float(beta[0]), float(beta[1])
        return slope, intercept
    except Exception:
        return 0.0, float(np.mean(yy))


def _fit_platt(raw_score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # deterministic Newton updates for logistic calibration p = sigmoid(a*s + b)
    s = raw_score.astype(float)
    yb = y.astype(float)
    a = 1.0
    b = 0.0
    for _ in range(50):
        z = a * s + b
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-6, None)
        g_a = np.sum((p - yb) * s)
        g_b = np.sum(p - yb)
        h_aa = np.sum(w * s * s) + 1e-6
        h_ab = np.sum(w * s)
        h_bb = np.sum(w) + 1e-6
        H = np.array([[h_aa, h_ab], [h_ab, h_bb]], dtype=float)
        g = np.array([g_a, g_b], dtype=float)
        step = np.linalg.solve(H, g)
        a -= step[0]
        b -= step[1]
        if np.linalg.norm(step) < 1e-8:
            break
    return float(a), float(b)


def fit_session_calibrator(session_name: str, raw_score: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Fit per-session Platt calibrator with diagnostics."""

    if not session_name:
        raise ValueError("session_name is required.")
    s = np.asarray(raw_score, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if len(s) != len(y):
        raise ValueError("raw_score and y_true length mismatch.")
    if len(s) == 0:
        raise ValueError("Empty calibration sample.")
    y = (y > 0).astype(float)

    a, b = _fit_platt(s, y)
    p = _sigmoid(a * s + b)
    slope, intercept = _reliability_line(y, p)
    cal = {
        "session_name": session_name,
        "type": "platt",
        "a": float(a),
        "b": float(b),
        "diagnostics": {
            "brier": _brier(y, p),
            "ece": _ece(y, p, n_bins=10),
            "reliability_slope": float(slope),
            "reliability_intercept": float(intercept),
            "n": int(len(y)),
        },
    }
    return cal


def predict_calibrated_prob(calibrator: Dict[str, Any], raw_score: np.ndarray) -> np.ndarray:
    s = np.asarray(raw_score, dtype=float)
    if calibrator.get("type") != "platt":
        raise ValueError(f"Unsupported calibrator type: {calibrator.get('type')}")
    a = float(calibrator["a"])
    b = float(calibrator["b"])
    return _sigmoid(a * s + b)


@dataclass(frozen=True)
class ThresholdConfig:
    min_threshold: float = 0.50
    max_threshold: float = 0.90
    step: float = 0.01
    min_trades: int = 20
    min_long_trades: int = 20
    min_short_trades: int = 20
    min_ev: float = 0.0
    smoothness_penalty: float = 0.25
    spike_penalty: float = 0.25
    neighborhood: int = 2
    abstain_value: int = 0
    cost_conditioning_enabled: bool = True
    cost_state_min_samples: int = 80
    cost_state_max_adjustment: float = 0.06
    cost_state_spread_buckets: int = 3
    cost_state_spread_atr_buckets: int = 3
    cost_state_use_calibration_bucket: bool = False
    # Optional fixed edges. When set, they override quantile bucketing.
    # Convention: list of interior cut points; bucket edges become [-inf, cuts..., inf].
    cost_state_spread_edges: Optional[list[float]] = None
    cost_state_spread_atr_edges: Optional[list[float]] = None


def _ev_curve(prob: np.ndarray, ev: np.ndarray, grid: np.ndarray, min_trades: int) -> tuple[np.ndarray, np.ndarray]:
    curve = np.full(len(grid), np.nan, dtype=float)
    counts = np.zeros(len(grid), dtype=int)
    for i, t in enumerate(grid):
        m = prob >= t
        counts[i] = int(np.sum(m))
        if counts[i] >= min_trades:
            curve[i] = float(np.mean(ev[m]))
    return curve, counts


def _local_smoothness(curve: np.ndarray, i: int, nbh: int) -> float:
    lo = max(0, i - nbh)
    hi = min(len(curve), i + nbh + 1)
    w = curve[lo:hi]
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return -1e9
    return float(np.mean(w))


def _spike_measure(curve: np.ndarray, i: int) -> float:
    c = curve[i]
    if not np.isfinite(c):
        return np.inf
    l = curve[i - 1] if i - 1 >= 0 else c
    r = curve[i + 1] if i + 1 < len(curve) else c
    if not np.isfinite(l):
        l = c
    if not np.isfinite(r):
        r = c
    return float(abs((l + r) / 2.0 - c))


def fit_session_threshold(
    session_name: str,
    prob: np.ndarray,
    realized_ev: np.ndarray,
    threshold_config: ThresholdConfig,
    cost_state_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Fit deterministic per-session threshold maximizing stable post-cost EV."""

    p = np.asarray(prob, dtype=float)
    ev = np.asarray(realized_ev, dtype=float)
    if len(p) != len(ev):
        raise ValueError("prob and realized_ev length mismatch.")
    if len(p) == 0:
        raise ValueError("Empty threshold sample.")

    grid = np.arange(
        float(threshold_config.min_threshold),
        float(threshold_config.max_threshold) + 0.5 * float(threshold_config.step),
        float(threshold_config.step),
    )
    min_long_trades = int(max(1, threshold_config.min_long_trades or threshold_config.min_trades))
    min_short_trades = int(max(1, threshold_config.min_short_trades or threshold_config.min_trades))
    min_ev = float(threshold_config.min_ev)
    curve, counts = _ev_curve(p, ev, grid, min_long_trades)
    # Short side uses low p_up region where negative long-edge implies short opportunity.
    short_grid = np.arange(
        max(0.0, 1.0 - float(threshold_config.max_threshold)),
        min(1.0, 1.0 - float(threshold_config.min_threshold)) + 0.5 * float(threshold_config.step),
        float(threshold_config.step),
    )
    short_curve = np.full(len(short_grid), np.nan, dtype=float)
    short_counts = np.zeros(len(short_grid), dtype=int)
    for i, t in enumerate(short_grid):
        m = p <= t
        short_counts[i] = int(np.sum(m))
        if short_counts[i] >= min_short_trades:
            short_curve[i] = float(np.mean(-ev[m]))

    score = np.full(len(grid), -1e18, dtype=float)
    for i, t in enumerate(grid):
        if (not np.isfinite(curve[i])) or (curve[i] < min_ev):
            continue
        smooth = _local_smoothness(curve, i, int(threshold_config.neighborhood))
        spike = _spike_measure(curve, i)
        score[i] = (
            curve[i]
            + float(threshold_config.smoothness_penalty) * smooth
            - float(threshold_config.spike_penalty) * spike
        )

    best_i = int(np.argmax(score)) if len(score) else 0
    long_enabled = bool(len(score) and np.isfinite(score[best_i]))
    best_t = float(grid[best_i]) if long_enabled else float("nan")
    # short side score with same stability penalties
    short_score = np.full(len(short_grid), -1e18, dtype=float)
    for i, _t in enumerate(short_grid):
        if (not np.isfinite(short_curve[i])) or (short_curve[i] < min_ev):
            continue
        smooth = _local_smoothness(short_curve, i, int(threshold_config.neighborhood))
        spike = _spike_measure(short_curve, i)
        short_score[i] = (
            short_curve[i]
            + float(threshold_config.smoothness_penalty) * smooth
            - float(threshold_config.spike_penalty) * spike
        )
    short_i = int(np.argmax(short_score)) if len(short_score) else 0
    short_enabled = bool(len(short_grid) and len(short_score) and np.isfinite(short_score[short_i]))
    short_t = float(short_grid[short_i]) if short_enabled else float("nan")
    out = {
        "session_name": session_name,
        "threshold": best_t,
        "threshold_long": best_t,
        "threshold_short": short_t,
        "long_enabled": long_enabled,
        "short_enabled": short_enabled,
        "abstain_value": int(threshold_config.abstain_value),
        "diagnostics": {
            "grid": grid.tolist(),
            "ev_curve": [None if not np.isfinite(x) else float(x) for x in curve],
            "score_curve": [None if not np.isfinite(x) else float(x) for x in score],
            "trade_count_curve": counts.tolist(),
            "chosen_index": best_i,
            "local_smoothness": _local_smoothness(curve, best_i, int(threshold_config.neighborhood)),
            "local_spike": _spike_measure(curve, best_i),
            "short_grid": short_grid.tolist(),
            "short_ev_curve": [None if not np.isfinite(x) else float(x) for x in short_curve],
            "short_score_curve": [None if not np.isfinite(x) else float(x) for x in short_score],
            "short_trade_count_curve": short_counts.tolist(),
            "short_chosen_index": short_i,
        },
    }
    out["cost_conditioning"] = _fit_cost_conditioned_adjustments(out, p, ev, threshold_config, cost_state_df)
    return out


def _safe_edges(x: np.ndarray, bins: int) -> np.ndarray:
    if len(x) == 0:
        return np.array([0.0, 1.0], dtype=float)
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    e = np.quantile(x, q)
    e = np.unique(e)
    if len(e) < 2:
        z = float(e[0]) if len(e) else 0.0
        return np.array([z - 1e-9, z + 1e-9], dtype=float)
    return np.asarray(e, dtype=float)


def _custom_edges_or_none(values: Optional[list[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.unique(np.sort(arr))
    if len(arr) == 0:
        return None
    # Interior cut points -> explicit finite outer edges (JSON-friendly).
    pad = max(1.0, float(np.max(np.abs(arr))) * 10.0 + 1.0)
    lo = float(arr[0] - pad)
    hi = float(arr[-1] + pad)
    return np.concatenate([np.array([lo], dtype=float), arr, np.array([hi], dtype=float)])


def _digitize_bucket(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) < 2:
        return np.zeros(len(x), dtype=int)
    return np.digitize(x, edges[1:-1], right=True).astype(int)


def _build_state_keys_from_arrays(
    session_arr: np.ndarray,
    spread_arr: np.ndarray,
    spread_atr_arr: np.ndarray,
    cal_drift_arr: Optional[np.ndarray],
    spread_edges: np.ndarray,
    spread_atr_edges: np.ndarray,
) -> np.ndarray:
    sb = _digitize_bucket(spread_arr, spread_edges)
    rab = _digitize_bucket(spread_atr_arr, spread_atr_edges)
    if cal_drift_arr is None:
        cb = np.zeros(len(session_arr), dtype=int)
    else:
        cb = (cal_drift_arr > 0).astype(int)
    return np.array(
        [f"s={str(s)}|sp={int(a)}|sa={int(b)}|cd={int(c)}" for s, a, b, c in zip(session_arr, sb, rab, cb)],
        dtype=object,
    )


def _fit_cost_conditioned_adjustments(
    baseline: Dict[str, Any],
    prob: np.ndarray,
    ev: np.ndarray,
    threshold_config: ThresholdConfig,
    cost_state_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    cfg = threshold_config
    if (not cfg.cost_conditioning_enabled) or (cost_state_df is None) or (len(cost_state_df) != len(prob)):
        return {"enabled": False, "state_thresholds": {}, "fallback_to_baseline": True}
    s = cost_state_df.get("session_bucket", pd.Series("na", index=cost_state_df.index)).astype(str).to_numpy(dtype=object)
    spread = pd.to_numeric(cost_state_df.get("spread_proxy", pd.Series(np.nan, index=cost_state_df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    spread_atr = pd.to_numeric(cost_state_df.get("spread_atr", pd.Series(np.nan, index=cost_state_df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cal_drift = None
    if cfg.cost_state_use_calibration_bucket and "calibration_drift" in cost_state_df.columns:
        cal_drift = pd.to_numeric(cost_state_df["calibration_drift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    spread_edges = _custom_edges_or_none(cfg.cost_state_spread_edges)
    if spread_edges is None:
        spread_edges = _safe_edges(spread, int(cfg.cost_state_spread_buckets))
    spread_atr_edges = _custom_edges_or_none(cfg.cost_state_spread_atr_edges)
    if spread_atr_edges is None:
        spread_atr_edges = _safe_edges(spread_atr, int(cfg.cost_state_spread_atr_buckets))
    keys = _build_state_keys_from_arrays(s, spread, spread_atr, cal_drift, spread_edges, spread_atr_edges)

    base_long = float(baseline["threshold_long"]) if np.isfinite(baseline["threshold_long"]) else np.nan
    base_short = float(baseline["threshold_short"]) if np.isfinite(baseline["threshold_short"]) else np.nan
    max_adj = float(max(0.0, cfg.cost_state_max_adjustment))
    min_samples = int(max(1, cfg.cost_state_min_samples))
    states: Dict[str, Any] = {}

    for k in sorted(set(keys.tolist())):
        m = keys == k
        n = int(np.sum(m))
        if n < min_samples:
            states[k] = {
                "n": n,
                "fallback_to_baseline": True,
                "threshold_long": base_long,
                "threshold_short": base_short,
            }
            continue
        local = fit_session_threshold(
            session_name=str(baseline.get("session_name", "session")),
            prob=prob[m],
            realized_ev=ev[m],
            threshold_config=ThresholdConfig(
                min_threshold=cfg.min_threshold,
                max_threshold=cfg.max_threshold,
                step=cfg.step,
                min_trades=cfg.min_trades,
                min_long_trades=cfg.min_long_trades,
                min_short_trades=cfg.min_short_trades,
                min_ev=cfg.min_ev,
                smoothness_penalty=cfg.smoothness_penalty,
                spike_penalty=cfg.spike_penalty,
                neighborhood=cfg.neighborhood,
                abstain_value=cfg.abstain_value,
                cost_conditioning_enabled=False,
            ),
        )
        lt = float(local.get("threshold_long", np.nan))
        st = float(local.get("threshold_short", np.nan))
        if np.isfinite(base_long) and np.isfinite(lt):
            lt = float(np.clip(lt, base_long - max_adj, base_long + max_adj))
        if np.isfinite(base_short) and np.isfinite(st):
            st = float(np.clip(st, base_short - max_adj, base_short + max_adj))
        states[k] = {
            "n": n,
            "fallback_to_baseline": False,
            "threshold_long": lt,
            "threshold_short": st,
            "long_enabled": bool(local.get("long_enabled", np.isfinite(lt))),
            "short_enabled": bool(local.get("short_enabled", np.isfinite(st))),
        }

    return {
        "enabled": True,
        "spread_edges": spread_edges.tolist(),
        "spread_atr_edges": spread_atr_edges.tolist(),
        "use_calibration_bucket": bool(cfg.cost_state_use_calibration_bucket),
        "state_thresholds": states,
        "fallback_to_baseline": False,
        "max_adjustment": max_adj,
        "min_state_samples": min_samples,
    }


def build_cost_state_keys(
    threshold_obj: Dict[str, Any],
    session_bucket: np.ndarray,
    spread_proxy: np.ndarray,
    spread_atr: np.ndarray,
    calibration_drift: Optional[np.ndarray] = None,
) -> np.ndarray:
    cc = threshold_obj.get("cost_conditioning", {})
    if not cc or not bool(cc.get("enabled", False)):
        return np.array(["baseline"] * len(np.asarray(session_bucket)), dtype=object)
    se = np.asarray(cc.get("spread_edges", [0.0, 1.0]), dtype=float)
    sae = np.asarray(cc.get("spread_atr_edges", [0.0, 1.0]), dtype=float)
    s = np.asarray(session_bucket, dtype=object)
    sp = np.asarray(spread_proxy, dtype=float)
    sa = np.asarray(spread_atr, dtype=float)
    cd = np.asarray(calibration_drift, dtype=float) if calibration_drift is not None else None
    return _build_state_keys_from_arrays(s, sp, sa, cd, se, sae)


def apply_session_threshold(
    session_name: str,
    prob: np.ndarray,
    threshold_obj: Dict[str, Any],
    cost_state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply per-session threshold with optional cost-state threshold shift."""

    p = np.asarray(prob, dtype=float)
    base_t = float(threshold_obj.get("threshold_long", threshold_obj.get("threshold", np.nan)))
    short_t = float(threshold_obj.get("threshold_short", np.nan))
    long_enabled = bool(threshold_obj.get("long_enabled", np.isfinite(base_t)))
    short_enabled = bool(threshold_obj.get("short_enabled", np.isfinite(short_t)))
    abstain_value = int(threshold_obj.get("abstain_value", 0))

    t_long = np.full(len(p), base_t, dtype=float)
    t_short = np.full(len(p), short_t, dtype=float)
    if cost_state is not None:
        cc = threshold_obj.get("cost_conditioning", {})
        if cc and bool(cc.get("enabled", False)):
            session = np.asarray(cost_state.get("session_bucket", np.array([session_name] * len(p))), dtype=object)
            spread = np.asarray(cost_state.get("spread_proxy", np.zeros(len(p))), dtype=float)
            spread_atr = np.asarray(cost_state.get("spread_atr", np.zeros(len(p))), dtype=float)
            cal_drift = cost_state.get("calibration_drift")
            cal_arr = np.asarray(cal_drift, dtype=float) if cal_drift is not None else None
            keys = build_cost_state_keys(
                threshold_obj=threshold_obj,
                session_bucket=session,
                spread_proxy=spread,
                spread_atr=spread_atr,
                calibration_drift=cal_arr,
            )
            state_map = cc.get("state_thresholds", {})
            for i, k in enumerate(keys):
                st = state_map.get(str(k))
                if st is None or bool(st.get("fallback_to_baseline", False)):
                    continue
                if np.isfinite(float(st.get("threshold_long", np.nan))):
                    t_long[i] = float(st["threshold_long"])
                if np.isfinite(float(st.get("threshold_short", np.nan))):
                    t_short[i] = float(st["threshold_short"])
        if "threshold_shift" in cost_state:
            sh = cost_state["threshold_shift"]
            if np.isscalar(sh):
                t_long = t_long + float(sh)
                t_short = t_short - float(sh)
            else:
                arr = np.asarray(sh, dtype=float)
                if len(arr) != len(p):
                    raise ValueError("cost_state['threshold_shift'] length mismatch.")
                t_long = t_long + arr
                t_short = t_short - arr
    sig = np.full(len(p), abstain_value, dtype=int)
    if long_enabled and np.isfinite(base_t):
        sig[p >= t_long] = 1
    if short_enabled and np.isfinite(short_t):
        sig[p <= t_short] = -1
    # deterministic tie-break if overlap due aggressive shifts.
    overlap = (p >= t_long) & (p <= t_short)
    if np.any(overlap):
        mid = 0.5 * (t_long + t_short)
        sig[overlap] = np.where(p[overlap] >= mid[overlap], 1, -1)
    return sig


def ev_by_probability_decile(prob: np.ndarray, realized_ev: np.ndarray) -> pd.DataFrame:
    p = np.asarray(prob, dtype=float)
    ev = np.asarray(realized_ev, dtype=float)
    q = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    df = pd.DataFrame({"decile": q, "prob": p, "ev": ev})
    out = (
        df.groupby("decile", dropna=True)
        .agg(mean_prob=("prob", "mean"), mean_ev=("ev", "mean"), count=("ev", "size"))
        .reset_index()
        .sort_values("decile")
    )
    return out


def threshold_stability_across_folds(thresholds: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(thresholds), dtype=float)
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "range": np.nan, "cv": np.nan}
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    rg = float(np.max(arr) - np.min(arr))
    cv = float(sd / (abs(mu) + 1e-9))
    return {"mean": mu, "std": sd, "range": rg, "cv": cv}


def local_ev_smoothness(threshold_obj: Dict[str, Any]) -> Dict[str, float]:
    d = threshold_obj.get("diagnostics", {})
    return {
        "local_smoothness": float(d.get("local_smoothness", np.nan)),
        "local_spike": float(d.get("local_spike", np.nan)),
    }


def to_jsonable(obj: Any) -> str:
    return str(asdict(obj)) if hasattr(obj, "__dataclass_fields__") else str(obj)
