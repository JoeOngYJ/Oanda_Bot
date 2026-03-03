from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int
    val_size: int
    max_label_horizon_bars: int
    embargo_bars: int = 0
    min_train_size: int = 200
    label_end_col: str = "label_end_ts"
    split_mode: str = "tail"  # tail|rolling|anchored_yearly
    rolling_train_size: Optional[int] = None
    anchor_train_start: Optional[str] = None
    validation_years: Optional[List[int]] = None


@dataclass(frozen=True)
class LinearModel:
    feature_order: List[str]
    mean: List[float]
    std: List[float]
    weights: List[float]
    intercept: float
    seed: int


@dataclass(frozen=True)
class TrainingSliceMeta:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    purge_start: str
    purge_end: str
    embargo_start: str
    embargo_end: str
    train_rows: int
    validation_rows: int


@dataclass(frozen=True)
class ModelArtifacts:
    feature_schema: Dict[str, List[str]]
    trunk_model: Dict[str, Any]
    session_heads: Dict[str, Dict[str, Any]]
    training_slice: Dict[str, Any]
    seeds: Dict[str, int]
    training_config_snapshot: Dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _ensure_2d_numeric(df: pd.DataFrame, feature_order: Optional[List[str]] = None) -> tuple[np.ndarray, List[str]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected DataFrame input.")
    cols = list(df.columns) if feature_order is None else list(feature_order)
    if feature_order is not None and cols != list(df.columns):
        # reorder for deterministic schema match if same set, else fail.
        missing = [c for c in cols if c not in df.columns]
        extra = [c for c in df.columns if c not in cols]
        if missing or extra:
            raise ValueError(f"Feature schema mismatch. missing={missing} extra={extra}")
        df = df.loc[:, cols]
    arr = df.to_numpy(dtype=float)
    return arr, cols


def _fit_linear_prob(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    sample_weight: Optional[pd.Series] = None,
) -> LinearModel:
    x, cols = _ensure_2d_numeric(X)
    yb = pd.to_numeric(y, errors="coerce").fillna(0).to_numpy(dtype=float)
    yb = (yb > 0).astype(float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    z = (x - mu) / sd

    wv = np.ones(len(z), dtype=float)
    if sample_weight is not None:
        wv = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        wv[~np.isfinite(wv) | (wv <= 0)] = 1.0
    sw = np.sqrt(wv)
    Zw = z * sw[:, None]
    yw = yb * sw

    ridge = 1e-3
    A = Zw.T @ Zw + ridge * np.eye(Zw.shape[1])
    b = Zw.T @ yw
    beta = np.linalg.solve(A, b)
    # deterministic intercept from weighted residual mean.
    p0 = _sigmoid(z @ beta)
    eps = 1e-6
    p0 = np.clip(p0, eps, 1 - eps)
    logit_p0 = np.log(p0 / (1 - p0))
    intercept = float(np.average(yb - logit_p0, weights=wv))

    return LinearModel(
        feature_order=cols,
        mean=mu.tolist(),
        std=sd.tolist(),
        weights=beta.tolist(),
        intercept=intercept,
        seed=int(seed),
    )


def _predict_linear_raw(model: LinearModel, X: pd.DataFrame) -> np.ndarray:
    _assert_schema(model.feature_order, list(X.columns))
    x, _ = _ensure_2d_numeric(X, model.feature_order)
    mu = np.array(model.mean, dtype=float)
    sd = np.array(model.std, dtype=float)
    beta = np.array(model.weights, dtype=float)
    z = (x - mu) / sd
    raw = z @ beta + float(model.intercept)
    return raw


def _assert_schema(expected: List[str], actual: List[str]) -> None:
    if list(expected) != list(actual):
        raise ValueError(
            "Feature schema mismatch: expected "
            f"{list(expected)} but got {list(actual)}"
        )


def make_purged_walk_forward_splits(df: pd.DataFrame, split_config: SplitConfig) -> List[Dict[str, Any]]:
    """Create contiguous forward validation splits with purge+embargo metadata."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be DatetimeIndex.")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError("df index must be timezone-aware.")
    if not bool(idx.is_monotonic_increasing):
        raise ValueError("df index must be monotonic increasing.")
    if len(df) < split_config.val_size + split_config.min_train_size:
        return []

    n = len(df)
    val_size = int(split_config.val_size)
    n_splits = int(split_config.n_splits)
    purge = int(split_config.max_label_horizon_bars)
    embargo = int(split_config.embargo_bars)
    mode = str(split_config.split_mode or "tail").lower()
    anchor_start_idx = 0
    if split_config.anchor_train_start:
        ts = pd.to_datetime(split_config.anchor_train_start, utc=True, errors="coerce")
        if pd.notna(ts):
            anchor_start_idx = int(np.searchsorted(idx.view("i8"), pd.Timestamp(ts).value, side="left"))

    splits: List[Dict[str, Any]] = []
    val_windows: List[tuple[int, int]] = []
    if mode == "anchored_yearly":
        years = pd.Series(idx.year, index=np.arange(len(idx)))
        uniq_years = sorted([int(y) for y in years.unique().tolist()])
        val_years = split_config.validation_years if split_config.validation_years else uniq_years[1:]
        for y in val_years:
            m = years.eq(int(y)).to_numpy()
            if not np.any(m):
                continue
            pos = np.where(m)[0]
            val_windows.append((int(pos[0]), int(pos[-1] + 1)))
        if split_config.n_splits > 0 and len(val_windows) > split_config.n_splits:
            val_windows = val_windows[-int(split_config.n_splits) :]
    else:
        for k in range(n_splits):
            val_start = n - (n_splits - k) * val_size
            val_end = val_start + val_size
            if val_start < 0 or val_end > n:
                continue
            val_windows.append((int(val_start), int(val_end)))

    for k, (val_start, val_end) in enumerate(val_windows):
        purge_start = max(anchor_start_idx, val_start - purge)
        purge_end = val_start
        train_end = purge_start
        train_start = anchor_start_idx
        if mode == "rolling" and split_config.rolling_train_size is not None:
            train_start = max(train_start, train_end - int(split_config.rolling_train_size))
        if (train_end - train_start) < split_config.min_train_size:
            continue
        train_idx = np.arange(train_start, train_end, dtype=int)
        val_idx = np.arange(val_start, val_end, dtype=int)

        # Optional stricter leakage prevention when label_end_ts is present.
        if split_config.label_end_col in df.columns and len(train_idx) > 0:
            label_end = pd.to_datetime(df.iloc[train_idx][split_config.label_end_col], utc=True, errors="coerce")
            keep = (label_end.isna()) | (label_end < idx[val_start])
            train_idx = train_idx[keep.to_numpy()]

        embargo_start = val_end
        embargo_end = min(n, val_end + embargo)
        meta = {
            "fold": int(k + 1),
            "train_start": idx[train_idx[0]].isoformat() if len(train_idx) else None,
            "train_end": idx[train_idx[-1]].isoformat() if len(train_idx) else None,
            "purge_window": {
                "start": idx[purge_start].isoformat() if purge_start < n else None,
                "end": idx[purge_end - 1].isoformat() if purge_end - 1 < n and purge_end - 1 >= 0 else None,
                "bars": int(max(0, purge_end - purge_start)),
            },
            "embargo_window": {
                "start": idx[embargo_start].isoformat() if embargo_start < n else None,
                "end": idx[embargo_end - 1].isoformat() if embargo_end - 1 < n and embargo_end - 1 >= 0 else None,
                "bars": int(max(0, embargo_end - embargo_start)),
            },
            "validation_start": idx[val_idx[0]].isoformat(),
            "validation_end": idx[val_idx[-1]].isoformat(),
            "train_indices": train_idx.tolist(),
            "validation_indices": val_idx.tolist(),
            "serializable": True,
            "split_mode": mode,
        }
        splits.append(meta)
    return splits


def fit_shared_trunk(
    X_shared: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit deterministic shared trunk model."""
    np.random.seed(int(seed))
    model = _fit_linear_prob(X_shared.copy(), y.copy(), seed=int(seed), sample_weight=sample_weight)
    return asdict(model)


def transform_shared_trunk(trunk_model: Dict[str, Any], X_shared: pd.DataFrame) -> pd.DataFrame:
    """Transform shared features into latent trunk representation Z."""
    model = LinearModel(**trunk_model)
    raw = _predict_linear_raw(model, X_shared.copy())
    z = pd.DataFrame(index=X_shared.index)
    z["z_trunk_raw"] = raw
    z["z_trunk_prob"] = _sigmoid(raw)
    return z


def fit_session_head(
    session_name: str,
    Z: pd.DataFrame,
    X_session: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit deterministic session head model over [Z, X_session]."""
    if not isinstance(session_name, str) or not session_name:
        raise ValueError("session_name must be a non-empty string.")
    X = pd.concat([Z.copy(), X_session.copy()], axis=1)
    np.random.seed(int(seed))
    model = _fit_linear_prob(X, y.copy(), seed=int(seed), sample_weight=sample_weight)
    d = asdict(model)
    d["session_name"] = session_name
    return d


def predict_session_head(head_model: Dict[str, Any], Z: pd.DataFrame, X_session: pd.DataFrame) -> pd.Series:
    """Predict deterministic raw score from session head over [Z, X_session]."""
    model_dict = dict(head_model)
    model_dict.pop("session_name", None)
    model = LinearModel(**model_dict)
    X = pd.concat([Z.copy(), X_session.copy()], axis=1)
    raw = _predict_linear_raw(model, X)
    return pd.Series(raw, index=X.index, dtype=float, name="raw_score")


def build_artifact_container(
    feature_schema: Dict[str, List[str]],
    trunk_model: Dict[str, Any],
    session_heads: Dict[str, Dict[str, Any]],
    training_slice: TrainingSliceMeta,
    seeds: Dict[str, int],
    training_config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Build serializable artifact container for fold persistence."""
    art = ModelArtifacts(
        feature_schema=feature_schema,
        trunk_model=trunk_model,
        session_heads=session_heads,
        training_slice=asdict(training_slice),
        seeds={k: int(v) for k, v in seeds.items()},
        training_config_snapshot=training_config_snapshot,
    )
    return asdict(art)


def assert_model_schema(model: Dict[str, Any], X: pd.DataFrame) -> None:
    _assert_schema(list(model["feature_order"]), list(X.columns))


def to_jsonable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
