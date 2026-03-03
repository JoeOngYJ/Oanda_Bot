from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _ensure_tz_index(df: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be DatetimeIndex.")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError(f"{name} index must be timezone-aware.")
    if not bool(idx.is_monotonic_increasing):
        raise ValueError(f"{name} index must be monotonic increasing.")
    if bool(idx.duplicated().any()):
        raise ValueError(f"{name} index contains duplicate timestamps.")
    return idx


def _infer_freq(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if len(idx) < 2:
        raise ValueError("Cannot infer HTF frequency from fewer than 2 rows.")
    d = pd.Series(idx).diff().dropna()
    mode = d.mode()
    if len(mode) == 0:
        raise ValueError("Failed to infer HTF frequency.")
    return pd.Timedelta(mode.iloc[0])


def align_htf_features(m15_df: pd.DataFrame, htf_df: pd.DataFrame, htf_cols: List[str]) -> pd.DataFrame:
    """Backward-only HTF alignment onto M15 rows using only fully completed HTF bars."""

    m15_idx = _ensure_tz_index(m15_df, "m15_df")
    htf_idx = _ensure_tz_index(htf_df, "htf_df")
    miss = [c for c in htf_cols if c not in htf_df.columns]
    if miss:
        raise ValueError(f"Requested HTF columns missing: {miss}")
    if len(htf_df) == 0:
        return pd.DataFrame(index=m15_idx, columns=htf_cols, dtype=float)

    freq = _infer_freq(htf_idx)
    htf = htf_df[htf_cols].copy()
    htf["htf_ts"] = htf_idx
    htf["htf_completed_ts"] = htf["htf_ts"] + freq

    # Exclude incomplete HTF bars relative to M15 horizon.
    max_m15_ts = m15_idx.max()
    htf = htf.loc[htf["htf_completed_ts"] <= max_m15_ts].copy()
    if len(htf) == 0:
        return pd.DataFrame(index=m15_idx, columns=htf_cols, dtype=float)

    left = pd.DataFrame({"ts": m15_idx})
    right = htf.sort_values("htf_completed_ts")[["htf_completed_ts"] + htf_cols]
    right = right.rename(columns={"htf_completed_ts": "ts"})

    merged = pd.merge_asof(left, right, on="ts", direction="backward", allow_exact_matches=True)
    out = merged.set_index("ts")[htf_cols]
    out.index = m15_idx
    return out


def _fit_stats(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    q01 = float(x.quantile(0.01))
    q99 = float(x.quantile(0.99))
    xc = x.clip(lower=q01, upper=q99)
    med = float(np.nanmedian(xc.to_numpy(dtype=float)))
    q75 = float(np.nanquantile(xc.to_numpy(dtype=float), 0.75))
    q25 = float(np.nanquantile(xc.to_numpy(dtype=float), 0.25))
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr == 0.0:
        iqr = 1.0
    return {
        "method": "robust",
        "clip_q01": q01,
        "clip_q99": q99,
        "median": med,
        "iqr": float(iqr),
    }


def _transform_with_stats(s: pd.Series, st: Dict[str, Any]) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    q01 = float(st["clip_q01"])
    q99 = float(st["clip_q99"])
    med = float(st["median"])
    iqr = float(st["iqr"]) if float(st["iqr"]) != 0.0 else 1.0
    return x.clip(lower=q01, upper=q99).sub(med).div(iqr)


def fit_feature_scalers(train_df: pd.DataFrame, feature_groups: Dict[str, List[str]], by_session: bool) -> Dict[str, Any]:
    """Fit robust scalers on train data only with optional session-conditional scaling."""

    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("train_df must be a DataFrame.")
    if "global" not in feature_groups:
        raise ValueError("feature_groups must include 'global'.")

    global_cols = list(feature_groups.get("global", []))
    session_cols = list(feature_groups.get("session_sensitive", []))
    for c in global_cols + session_cols:
        if c not in train_df.columns:
            raise ValueError(f"Feature '{c}' not found in train_df.")

    bundle: Dict[str, Any] = {
        "version": 1,
        "column_order": list(train_df.columns),
        "by_session": bool(by_session),
        "feature_groups": {"global": global_cols, "session_sensitive": session_cols},
        "global_scalers": {},
        "session_scalers": {},
    }

    for c in global_cols:
        bundle["global_scalers"][c] = _fit_stats(train_df[c])

    if by_session:
        if "session_bucket" not in train_df.columns:
            raise ValueError("session_bucket column required when by_session=True.")
        for sess, g in train_df.groupby(train_df["session_bucket"].astype(str), sort=True):
            sess_map: Dict[str, Any] = {}
            for c in session_cols:
                sess_map[c] = _fit_stats(g[c])
            bundle["session_scalers"][str(sess)] = sess_map
    else:
        for c in session_cols:
            bundle["global_scalers"][c] = _fit_stats(train_df[c])
    return bundle


def transform_feature_scalers(df: pd.DataFrame, scaler_bundle: Dict[str, Any]) -> pd.DataFrame:
    """Apply frozen scaler state while preserving exact training column order."""

    assert_inference_schema(list(scaler_bundle["column_order"]), list(df.columns))
    out = df.copy()
    global_scalers = scaler_bundle.get("global_scalers", {})
    for c, st in global_scalers.items():
        out[c] = _transform_with_stats(out[c], st)

    if bool(scaler_bundle.get("by_session", False)):
        if "session_bucket" not in out.columns:
            raise ValueError("session_bucket column required for session-conditional transform.")
        sess_map = scaler_bundle.get("session_scalers", {})
        for sess, st_map in sess_map.items():
            mask = out["session_bucket"].astype(str).eq(str(sess))
            if not bool(mask.any()):
                continue
            for c, st in st_map.items():
                out.loc[mask, c] = _transform_with_stats(out.loc[mask, c], st)

    return out.loc[:, scaler_bundle["column_order"]]


def save_feature_schema(path: str, column_order: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"column_order": list(column_order)}, indent=2), encoding="utf-8")


def load_feature_schema(path: str) -> List[str]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "column_order" not in obj or not isinstance(obj["column_order"], list):
        raise ValueError("Invalid schema file: missing 'column_order'.")
    return [str(c) for c in obj["column_order"]]


def save_scaler_metadata(path: str, scaler_bundle: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scaler_bundle, indent=2), encoding="utf-8")


def load_scaler_metadata(path: str) -> Dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "column_order" not in obj:
        raise ValueError("Invalid scaler metadata: missing 'column_order'.")
    if "global_scalers" not in obj:
        raise ValueError("Invalid scaler metadata: missing 'global_scalers'.")
    return obj


def assert_inference_schema(train_columns: List[str], infer_columns: List[str]) -> None:
    if list(train_columns) != list(infer_columns):
        raise ValueError(
            "Inference schema mismatch: expected columns "
            f"{list(train_columns)} but got {list(infer_columns)}"
        )
