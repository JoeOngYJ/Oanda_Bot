#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import scripts.pipeline_compat_adapter as adapter


SESSIONS = [
    "asia",
    "pre_london",
    "london_open",
    "london_continuation",
    "ny_open",
    "ny_overlap_post_data",
]


@dataclass(frozen=True)
class SessionBoundary:
    start_hm: str
    end_hm: str


@dataclass
class SessionConfig:
    tz: str = "Europe/London"
    boundaries: Dict[str, SessionBoundary] = field(
        default_factory=lambda: {
            "asia": SessionBoundary("00:00", "06:00"),
            "pre_london": SessionBoundary("06:00", "08:00"),
            "london_open": SessionBoundary("08:00", "09:00"),
            "london_continuation": SessionBoundary("09:00", "13:00"),
            "ny_open": SessionBoundary("13:00", "15:00"),
            "ny_overlap_post_data": SessionBoundary("15:00", "24:00"),
        }
    )


@dataclass
class LabelSessionParams:
    up_mult: float
    dn_mult: float
    min_horizon_bars: int
    max_horizon_bars: int


@dataclass
class PipelineConfig:
    symbol: str = "XAU_USD"
    ltf: str = "M15"
    h1: str = "H1"
    d1: str = "D1"
    seed: int = 42
    n_splits: int = 6
    test_size: int = 1500
    embargo_bars: int = 96
    threshold_grid: List[float] = field(default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70])
    min_ev: float = 0.0
    corr_suppress_threshold: float = 0.95
    min_session_train_rows: int = 150
    output_dir: str = "data/research/xau_multi_session_pipeline"
    session_cfg: SessionConfig = field(default_factory=SessionConfig)
    label_params: Dict[str, LabelSessionParams] = field(
        default_factory=lambda: {
            "asia": LabelSessionParams(1.8, 1.8, 12, 32),
            "pre_london": LabelSessionParams(1.7, 1.7, 10, 28),
            "london_open": LabelSessionParams(1.3, 1.3, 6, 20),
            "london_continuation": LabelSessionParams(1.4, 1.4, 8, 24),
            "ny_open": LabelSessionParams(1.5, 1.5, 8, 24),
            "ny_overlap_post_data": LabelSessionParams(1.6, 1.6, 8, 26),
        }
    )


@dataclass
class LinearProbModel:
    feature_names: List[str]
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.feature_names].to_numpy(dtype=float)
        z = (arr - self.mu) / self.sd
        xb = np.hstack([np.ones((len(z), 1)), z]) @ self.w
        p = 1.0 / (1.0 + np.exp(-np.clip(xb, -40, 40)))
        return np.column_stack([1.0 - p, p])


def _parse_hm(hm: str) -> int:
    hh, mm = hm.split(":")
    return int(hh) * 60 + int(mm)


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    adapter.set_deterministic_seed(int(seed))


def load_ohlcv_bundle(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_m15 = adapter.load_ohlcv(cfg.symbol, cfg.ltf)
    df_h1 = adapter.load_ohlcv(cfg.symbol, cfg.h1)
    df_d1 = adapter.load_ohlcv(cfg.symbol, cfg.d1)
    return df_m15, df_h1, df_d1


def _exclude_incomplete_htf(df_htf: pd.DataFrame) -> pd.DataFrame:
    if len(df_htf) <= 1:
        return df_htf.iloc[0:0]
    return df_htf.iloc[:-1].copy()


def segment_sessions(index_utc: pd.DatetimeIndex, session_cfg: SessionConfig) -> pd.Series:
    idx = pd.DatetimeIndex(index_utc).tz_convert(session_cfg.tz)
    minute = (idx.hour * 60) + idx.minute
    out = pd.Series("unknown", index=index_utc, dtype="object")
    for s in SESSIONS:
        b = session_cfg.boundaries[s]
        a = _parse_hm(b.start_hm)
        z = _parse_hm(b.end_hm)
        if z <= a:
            mask = (minute >= a) | (minute < z)
        else:
            mask = (minute >= a) & (minute < z)
        out.loc[mask] = s
    return out


def compute_developing_session_range_features(
    df_m15: pd.DataFrame,
    session_series: pd.Series,
) -> pd.DataFrame:
    x = df_m15.copy()
    high = pd.to_numeric(x["high"], errors="coerce")
    low = pd.to_numeric(x["low"], errors="coerce")
    close = pd.to_numeric(x["close"], errors="coerce")
    atr14 = pd.to_numeric(x.get("atr14", (high - low).rolling(14, min_periods=14).mean()), errors="coerce")

    date_key = pd.DatetimeIndex(x.index).tz_convert("Europe/London").normalize()
    grp = pd.Series([f"{d.isoformat()}::{s}" for d, s in zip(date_key, session_series)], index=x.index, dtype="object")

    # developing-only range at t: use bars up to and including t.
    dev_high = high.groupby(grp).cummax()
    dev_low = low.groupby(grp).cummin()
    dev_range = dev_high - dev_low
    dev_mid = 0.5 * (dev_high + dev_low)

    out = pd.DataFrame(index=x.index)
    out["sess_dev_high"] = dev_high
    out["sess_dev_low"] = dev_low
    out["sess_dev_range"] = dev_range
    out["sess_dev_mid"] = dev_mid
    out["sess_dev_range_atr"] = dev_range / (atr14 + 1e-9)
    out["dist_to_sess_dev_high_atr"] = (close - dev_high) / (atr14 + 1e-9)
    out["dist_to_sess_dev_low_atr"] = (close - dev_low) / (atr14 + 1e-9)
    out["dist_to_sess_dev_mid_atr"] = (close - dev_mid) / (atr14 + 1e-9)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_feature_trunk(
    df_m15: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_d1: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    m15 = adapter._ensure_utc_index(df_m15)
    h1 = adapter._ensure_utc_index(_exclude_incomplete_htf(df_h1))
    d1 = adapter._ensure_utc_index(_exclude_incomplete_htf(df_d1))

    session_series = segment_sessions(pd.DatetimeIndex(m15.index), cfg.session_cfg)
    local_cfg = {"timezone": cfg.session_cfg.tz, "_df_h1": h1, "_df_d1": d1}
    feat = adapter.make_features(m15, cfg.ltf, local_cfg)
    sess_dev = compute_developing_session_range_features(m15.join(feat[["atr14"]], how="left"), session_series)
    sess_onehot = pd.get_dummies(session_series, prefix="session", dtype=float)
    out = pd.concat([feat, sess_dev, sess_onehot], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, session_series


def generate_session_conditioned_labels(
    df_m15: pd.DataFrame,
    feature_df: pd.DataFrame,
    session_series: pd.Series,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    x = adapter._ensure_utc_index(df_m15)
    sigma = pd.to_numeric(feature_df.get("atr_pct", pd.Series(index=x.index, data=np.nan)), errors="coerce")
    sigma = sigma.ffill().bfill().fillna(0.0)

    out = pd.DataFrame(index=x.index)
    out["label_side"] = 0
    out["label_end_ts"] = pd.Series(pd.DatetimeIndex([pd.NaT] * len(out), tz="UTC"), index=out.index)
    out["label_horizon_bars"] = 0
    for s in SESSIONS:
        params = cfg.label_params[s]
        idx = session_series[session_series == s].index
        if len(idx) == 0:
            continue
        tmp = x.loc[idx].copy()
        tmp["sigma"] = sigma.loc[idx]
        labels = adapter.compute_vol_scaled_triple_barrier_labels(
            tmp,
            sigma_col="sigma",
            up_mult=params.up_mult,
            dn_mult=params.dn_mult,
            min_horizon_bars=params.min_horizon_bars,
            max_horizon_bars=params.max_horizon_bars,
        )
        out.loc[idx, "label_side"] = labels["label_side"].to_numpy()
        out.loc[idx, "label_end_ts"] = pd.to_datetime(labels["label_end_ts"], utc=True, errors="coerce").to_numpy()
        out.loc[idx, "label_horizon_bars"] = labels["label_horizon_bars"].to_numpy()
    out["label_side"] = pd.to_numeric(out["label_side"], errors="coerce").fillna(0).astype(int)
    out["label_end_ts"] = pd.to_datetime(out["label_end_ts"], utc=True, errors="coerce")
    out["label_horizon_bars"] = pd.to_numeric(out["label_horizon_bars"], errors="coerce").fillna(0).astype(int)
    out["session"] = session_series.reindex(out.index)
    return out


def _fit_transform_train(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X_t, state = adapter.select_and_transform_features(X_train, list(X_train.columns))
    return X_t, state


def _apply_transform(X: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    cols = list(state.get("selected_columns", []))
    scalers = state.get("scalers", {})
    out = pd.DataFrame(index=X.index)
    for c in cols:
        s = pd.to_numeric(X.get(c, pd.Series(index=X.index, data=np.nan)), errors="coerce")
        sc = scalers.get(c, {})
        method = sc.get("method", "none")
        lo = sc.get("clip_q01", None)
        hi = sc.get("clip_q99", None)
        if lo is not None and hi is not None:
            s = s.clip(float(lo), float(hi))
        if method == "robust":
            med = float(sc.get("median", 0.0))
            iqr = float(sc.get("iqr", 1.0)) if float(sc.get("iqr", 1.0)) != 0 else 1.0
            s = (s - med) / iqr
        elif method == "zscore":
            mu = float(sc.get("mean", 0.0))
            sd = float(sc.get("std", 1.0)) if float(sc.get("std", 1.0)) != 0 else 1.0
            s = (s - mu) / sd
        out[c] = s
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_logit(X: pd.DataFrame, y: pd.Series, seed: int) -> LinearProbModel:
    _ = seed
    fn = list(X.columns)
    arr = X.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd[sd == 0] = 1.0
    z = (arr - mu) / sd
    yv = pd.to_numeric(y, errors="coerce").fillna(0).to_numpy(dtype=float)
    yb = (yv > 0).astype(float)
    Xd = np.hstack([np.ones((len(z), 1)), z])
    eye = np.eye(Xd.shape[1])
    eye[0, 0] = 0.0
    ridge = 1e-3
    a = (Xd.T @ Xd) + (ridge * eye)
    b = Xd.T @ yb
    w = np.linalg.solve(a, b)
    return LinearProbModel(feature_names=fn, mu=mu, sd=sd, w=w)


def _predict_prob(model: LinearProbModel, X: pd.DataFrame) -> pd.Series:
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=X.index, dtype=float)


def _fit_threshold_on_train_slice(
    p: pd.Series,
    y: pd.Series,
    gross_win: pd.Series,
    gross_loss: pd.Series,
    expected_cost: pd.Series,
    grid: List[float],
    min_ev: float,
) -> float:
    best_t = float(grid[0])
    best_score = -1e18
    ev = adapter.compute_expected_value(p, gross_win, gross_loss, expected_cost)
    tmp = pd.DataFrame({"p": p, "ev": ev}, index=p.index)
    for t in grid:
        g = adapter.apply_trade_gating(tmp, p_col="p", ev_col="ev", min_ev=min_ev, base_p_threshold=float(t))
        if int(g.sum()) == 0:
            continue
        score = float((ev[g > 0]).mean())
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _dependence_matrix(prob_by_session: Dict[str, pd.Series]) -> pd.DataFrame:
    if not prob_by_session:
        return pd.DataFrame()
    mat = pd.DataFrame(prob_by_session).corr(method="spearman")
    return mat


def _portfolio_weights(dep: pd.DataFrame, corr_suppress_threshold: float) -> Tuple[pd.Series, Dict[str, Any]]:
    if dep.empty:
        return pd.Series(dtype=float), {"suppressed": []}
    sessions = list(dep.columns)
    avg_corr = dep.abs().mean(axis=1).fillna(0.0)
    w = (1.0 - avg_corr).clip(lower=0.05)
    suppressed = []
    for i in sessions:
        for j in sessions:
            if i >= j:
                continue
            if float(abs(dep.loc[i, j])) >= float(corr_suppress_threshold):
                # suppress lower weight sleeve for fake diversification control
                loser = i if float(w.loc[i]) < float(w.loc[j]) else j
                w.loc[loser] = 0.0
                suppressed.append({"pair": [i, j], "suppressed": loser, "corr": float(dep.loc[i, j])})
    if float(w.sum()) <= 0:
        w[:] = 1.0
    w = w / w.sum()
    return w, {"suppressed": suppressed}


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def train_walkforward(cfg: PipelineConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df_m15, df_h1, df_d1 = load_ohlcv_bundle(cfg)
    feat, session_series = compute_feature_trunk(df_m15, df_h1, df_d1, cfg)
    labels = generate_session_conditioned_labels(df_m15, feat, session_series, cfg)
    y = (labels["label_side"] > 0).astype(int)
    label_end_ts = pd.to_datetime(labels["label_end_ts"], utc=True, errors="coerce").reindex(feat.index)

    valid = y.notna()
    feat = feat.loc[valid].copy()
    y = y.loc[valid].astype(int)
    session_series = session_series.reindex(feat.index)
    label_end_ts = label_end_ts.reindex(feat.index)

    splits = adapter.generate_purged_walkforward_splits(
        feat.index,
        label_end_ts,
        n_splits=cfg.n_splits,
        test_size=cfg.test_size,
        embargo_bars=cfg.embargo_bars,
    )
    summary_rows: List[Dict[str, Any]] = []
    fold_dirs: List[str] = []

    for k, (tr_idx, te_idx) in enumerate(splits, start=1):
        if len(tr_idx) < 200 or len(te_idx) < 50:
            continue
        fold_dir = out_root / f"fold_{k:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_dirs.append(str(fold_dir))

        X_tr = feat.iloc[tr_idx].copy()
        X_te = feat.iloc[te_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_te = y.iloc[te_idx].copy()
        s_tr = session_series.iloc[tr_idx].copy()
        s_te = session_series.iloc[te_idx].copy()

        # train/cal split within train only (prevents threshold fit on test slice).
        split_pt = int(len(X_tr) * 0.8)
        X_fit = X_tr.iloc[:split_pt].copy()
        X_cal = X_tr.iloc[split_pt:].copy()
        y_fit = y_tr.iloc[:split_pt].copy()
        y_cal = y_tr.iloc[split_pt:].copy()
        s_fit = s_tr.iloc[:split_pt].copy()
        s_cal = s_tr.iloc[split_pt:].copy()

        X_fit_t, scaler_state = _fit_transform_train(X_fit)
        X_cal_t = _apply_transform(X_cal, scaler_state)
        X_te_t = _apply_transform(X_te, scaler_state)

        # shared trunk
        trunk = _fit_logit(X_fit_t.fillna(0.0), y_fit, cfg.seed)
        trunk_fit_p = _predict_prob(trunk, X_fit_t.fillna(0.0))
        trunk_cal_p = _predict_prob(trunk, X_cal_t.fillna(0.0))
        trunk_te_p = _predict_prob(trunk, X_te_t.fillna(0.0))

        # specialist heads (train on trunk prob + selected core features).
        head_models: Dict[str, Any] = {}
        calibrators: Dict[str, Any] = {}
        thresholds: Dict[str, float] = {}
        te_probs_by_session: Dict[str, pd.Series] = {}
        sleeve_decisions: Dict[str, pd.Series] = {}

        core_cols = [c for c in X_fit_t.columns if c.startswith("regime_") or c.startswith("sess_dev_") or c.startswith("sweep_")]
        for s in SESSIONS:
            idx_fit_s = s_fit[s_fit == s].index.intersection(X_fit_t.index)
            idx_cal_s = s_cal[s_cal == s].index.intersection(X_cal_t.index)
            idx_te_s = s_te[s_te == s].index.intersection(X_te_t.index)
            if len(idx_fit_s) < cfg.min_session_train_rows:
                continue
            Xh_fit = pd.concat(
                [
                    trunk_fit_p.reindex(idx_fit_s).rename("trunk_p"),
                    X_fit_t.reindex(idx_fit_s, columns=core_cols),
                ],
                axis=1,
            ).fillna(0.0)
            yh_fit = y_fit.reindex(idx_fit_s).astype(int)
            head = _fit_logit(Xh_fit, yh_fit, cfg.seed + k + len(s))
            head_models[s] = head

            Xh_cal = pd.concat(
                [
                    trunk_cal_p.reindex(idx_cal_s).rename("trunk_p"),
                    X_cal_t.reindex(idx_cal_s, columns=core_cols),
                ],
                axis=1,
            ).fillna(0.0)
            Xh_te = pd.concat(
                [
                    trunk_te_p.reindex(idx_te_s).rename("trunk_p"),
                    X_te_t.reindex(idx_te_s, columns=core_cols),
                ],
                axis=1,
            ).fillna(0.0)

            p_cal_raw = _predict_prob(head, Xh_cal) if len(Xh_cal) else pd.Series(dtype=float)
            p_te_raw = _predict_prob(head, Xh_te) if len(Xh_te) else pd.Series(dtype=float)
            y_cal_s = y_cal.reindex(idx_cal_s).fillna(0).astype(int)

            if len(p_cal_raw) > 0:
                calibrator = adapter.fit_probability_calibrator(
                    p_cal_raw.to_numpy(dtype=float),
                    y_cal_s.to_numpy(dtype=int),
                    isotonic_min_samples=1000,
                    bucket_min_samples=300,
                )
                p_cal = pd.Series(
                    adapter.apply_probability_calibrator(calibrator, p_cal_raw.to_numpy(dtype=float)),
                    index=p_cal_raw.index,
                    dtype=float,
                )
                p_te = pd.Series(
                    adapter.apply_probability_calibrator(calibrator, p_te_raw.to_numpy(dtype=float)),
                    index=p_te_raw.index,
                    dtype=float,
                )
                calibrators[s] = calibrator
            else:
                p_cal = p_cal_raw
                p_te = p_te_raw
                calibrators[s] = {"type": "none"}

            gw = pd.Series(1.0, index=p_cal.index, dtype=float)
            gl = pd.Series(1.0, index=p_cal.index, dtype=float)
            ec = pd.Series(0.0001, index=p_cal.index, dtype=float)
            thr = _fit_threshold_on_train_slice(p_cal, y_cal_s, gw, gl, ec, cfg.threshold_grid, cfg.min_ev) if len(p_cal) else 0.55
            thresholds[s] = float(thr)

            if len(p_te) > 0:
                ev_te = adapter.compute_expected_value(
                    p_te,
                    pd.Series(1.0, index=p_te.index),
                    pd.Series(1.0, index=p_te.index),
                    pd.Series(0.0001, index=p_te.index),
                )
                gate_df = pd.DataFrame({"p": p_te, "ev": ev_te}, index=p_te.index)
                gate = adapter.apply_trade_gating(gate_df, p_col="p", ev_col="ev", min_ev=cfg.min_ev, base_p_threshold=thr)
                te_probs_by_session[s] = p_te
                sleeve_decisions[s] = gate.astype(int)

        dep = _dependence_matrix(te_probs_by_session)
        weights, controller_state = _portfolio_weights(dep, cfg.corr_suppress_threshold)

        combined = pd.Series(0.0, index=X_te_t.index, dtype=float)
        combined_gate = pd.Series(0, index=X_te_t.index, dtype=int)
        for s, p_s in te_probs_by_session.items():
            w = float(weights.get(s, 0.0))
            g = sleeve_decisions[s].reindex(X_te_t.index).fillna(0).astype(int)
            pp = p_s.reindex(X_te_t.index).fillna(0.0)
            combined += w * pp
            combined_gate = np.maximum(combined_gate, (g > 0).astype(int))

        y_te_full = y_te.reindex(X_te_t.index).fillna(0).astype(int)
        brier = float(((combined - y_te_full) ** 2).mean()) if len(combined) else np.nan
        trade_count = int(combined_gate.sum())

        _save_pickle(fold_dir / "scaler_state.pkl", scaler_state)
        _save_pickle(fold_dir / "shared_trunk.pkl", trunk)
        _save_pickle(fold_dir / "session_heads.pkl", head_models)
        _save_json(fold_dir / "calibrators.json", calibrators)
        _save_json(fold_dir / "thresholds.json", thresholds)
        dep.to_csv(fold_dir / "dependence_matrix.csv")
        _save_json(fold_dir / "portfolio_controller_state.json", {"weights": weights.to_dict(), **controller_state})
        pd.DataFrame(
            {
                "timestamp": X_te_t.index,
                "session": s_te.reindex(X_te_t.index).astype(str).values,
                "p_combined": combined.values,
                "trade_gate": combined_gate.values,
                "y_true": y_te_full.values,
            }
        ).to_csv(fold_dir / "portfolio_decisions.csv", index=False)

        summary_rows.append(
            {
                "fold": int(k),
                "train_rows": int(len(X_tr)),
                "test_rows": int(len(X_te)),
                "n_session_heads": int(len(head_models)),
                "brier": brier,
                "trade_count": trade_count,
                "avg_p": float(combined.mean()) if len(combined) else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_root / "fold_summary.csv"
    summary.to_csv(summary_path, index=False)
    run_manifest = {
        "config": asdict(cfg),
        "n_folds_saved": int(len(summary)),
        "fold_dirs": fold_dirs,
        "summary_csv": str(summary_path),
    }
    _save_json(out_root / "run_manifest.json", run_manifest)
    return run_manifest


def _load_cfg(path: Optional[str]) -> PipelineConfig:
    if not path:
        return PipelineConfig()
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    cfg = PipelineConfig()
    for k, v in obj.items():
        if k == "session_cfg" and isinstance(v, dict):
            b = v.get("boundaries", {})
            bb = {}
            for sk, sv in b.items():
                bb[sk] = SessionBoundary(start_hm=sv["start_hm"], end_hm=sv["end_hm"])
            cfg.session_cfg = SessionConfig(tz=v.get("tz", cfg.session_cfg.tz), boundaries=bb or cfg.session_cfg.boundaries)
        elif k == "label_params" and isinstance(v, dict):
            lp = {}
            for sk, sv in v.items():
                lp[sk] = LabelSessionParams(
                    up_mult=float(sv["up_mult"]),
                    dn_mult=float(sv["dn_mult"]),
                    min_horizon_bars=int(sv["min_horizon_bars"]),
                    max_horizon_bars=int(sv["max_horizon_bars"]),
                )
            cfg.label_params = lp
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic XAU multi-session pipeline")
    p.add_argument("--config", default="", help="JSON config path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_cfg(args.config or None)
    res = train_walkforward(cfg)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
