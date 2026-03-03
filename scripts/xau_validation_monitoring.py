from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

from scripts.xau_calibration_thresholds import (
    ThresholdConfig,
    apply_session_threshold,
    ev_by_probability_decile,
    fit_session_calibrator,
    fit_session_threshold,
    predict_calibrated_prob,
    threshold_stability_across_folds,
)
from scripts.xau_wfo_models import SplitConfig, make_purged_walk_forward_splits


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=True)
    n = len(p)
    acc = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        acc += float(np.sum(m) / n) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
    return float(acc)


def _cal_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(p)), p])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1]), float(beta[0])  # slope, intercept


def _drawdown_stats(pnl: pd.Series) -> tuple[pd.Series, float]:
    eq = pd.to_numeric(pnl, errors="coerce").fillna(0.0).cumsum()
    dd = eq - eq.cummax()
    return dd, float(dd.min())


def evaluate_sleeve_metrics(preds: pd.DataFrame, labels: pd.DataFrame, pnl: pd.Series) -> Dict[str, Any]:
    """Sleeve-level validation metrics."""
    req_p = {"prob"}
    req_l = {"y_true"}
    if not req_p.issubset(preds.columns):
        raise ValueError("preds must include 'prob'.")
    if not req_l.issubset(labels.columns):
        raise ValueError("labels must include 'y_true'.")

    p = pd.to_numeric(preds["prob"], errors="coerce")
    y = pd.to_numeric(labels["y_true"], errors="coerce")
    ix = p.index.intersection(y.index).intersection(pnl.index)
    p = p.reindex(ix)
    y = y.reindex(ix)
    x = pd.concat([p.rename("p"), y.rename("y"), pd.to_numeric(pnl.reindex(ix), errors="coerce").rename("pnl")], axis=1).dropna()
    if len(x) == 0:
        return {}

    signal = pd.to_numeric(preds.get("signal", pd.Series((x["p"] >= 0.5).astype(int), index=preds.index)), errors="coerce").reindex(x.index).fillna(0)
    abstain = (signal == 0).astype(float)
    trigger = (signal != 0).astype(float)
    slope, intercept = _cal_line(x["y"].to_numpy(dtype=float), x["p"].to_numpy(dtype=float))

    thr_stab = {}
    if {"fold", "threshold"}.issubset(preds.columns):
        t_by_fold = preds[["fold", "threshold"]].dropna().groupby("fold")["threshold"].mean().tolist()
        thr_stab = threshold_stability_across_folds(t_by_fold)

    regime_attr = {}
    regime_col = "regime_bucket"
    if regime_col in labels.columns:
        rr = pd.concat([labels[[regime_col]].reindex(x.index), x["pnl"]], axis=1).dropna()
        regime_attr = rr.groupby(regime_col)["pnl"].mean().to_dict()

    return {
        "log_loss": _log_loss(x["y"].to_numpy(dtype=float), x["p"].to_numpy(dtype=float)),
        "brier_score": _brier(x["y"].to_numpy(dtype=float), x["p"].to_numpy(dtype=float)),
        "ece": _ece(x["y"].to_numpy(dtype=float), x["p"].to_numpy(dtype=float)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expectancy_net_cost": float(x["pnl"].mean()),
        "trigger_rate": float(trigger.mean()),
        "abstain_rate": float(abstain.mean()),
        "threshold_stability_by_fold": thr_stab,
        "ev_by_score_decile": ev_by_probability_decile(x["p"].to_numpy(dtype=float), x["pnl"].to_numpy(dtype=float)).to_dict(orient="records"),
        "regime_bucket_attribution": regime_attr,
    }


def evaluate_portfolio_metrics(portfolio_pnl: pd.Series, sleeve_pnl_map: Mapping[str, pd.Series]) -> Dict[str, Any]:
    """Portfolio-level evaluation and sleeve attribution."""
    p = pd.to_numeric(portfolio_pnl, errors="coerce").fillna(0.0)
    dd, max_dd = _drawdown_stats(p)
    daily = p.resample("1D").sum(min_count=1) if isinstance(p.index, pd.DatetimeIndex) else p
    sleeve_mean = {k: float(pd.to_numeric(v.reindex(p.index), errors="coerce").fillna(0.0).mean()) for k, v in sleeve_pnl_map.items()}
    full_util = float(p.mean() - 0.5 * p.std(ddof=0))
    loo = {}
    for k, v in sleeve_pnl_map.items():
        vv = pd.to_numeric(v.reindex(p.index), errors="coerce").fillna(0.0)
        wo = p - vv
        util_wo = float(wo.mean() - 0.5 * wo.std(ddof=0))
        loo[k] = float(full_util - util_wo)

    co_dd = {}
    dd_mask = dd < 0
    if dd_mask.any():
        for k, v in sleeve_pnl_map.items():
            vv = pd.to_numeric(v.reindex(p.index), errors="coerce").fillna(0.0)
            co_dd[k] = float(vv[dd_mask].sum())

    # dependence drift proxy by fixed fold chunks
    dep_drift = {}
    if len(sleeve_pnl_map) >= 2 and len(p) >= 40:
        pnl_mat = pd.DataFrame({k: pd.to_numeric(v.reindex(p.index), errors="coerce").fillna(0.0) for k, v in sleeve_pnl_map.items()}, index=p.index)
        chunk = max(20, len(pnl_mat) // 4)
        corr_seq: List[pd.DataFrame] = []
        for i in range(0, len(pnl_mat), chunk):
            block = pnl_mat.iloc[i : i + chunk]
            if len(block) >= 5:
                corr_seq.append(block.corr())
        if len(corr_seq) >= 2:
            vals = np.array([c.to_numpy() for c in corr_seq], dtype=float)
            std_list: List[float] = []
            for i in range(vals.shape[1]):
                for j in range(vals.shape[2]):
                    v = vals[:, i, j]
                    vf = v[np.isfinite(v)]
                    if len(vf) >= 2:
                        std_list.append(float(np.std(vf, ddof=0)))
            if len(std_list):
                dep_drift["corr_std_mean"] = float(np.mean(std_list))

    if len(dd) <= 200:
        drawdown_obj: Any = {str(k): float(v) for k, v in dd.items()}
    else:
        drawdown_obj = {"n": int(len(dd)), "min": float(dd.min()), "last": float(dd.iloc[-1])}

    return {
        "expectancy": float(p.mean()),
        "drawdown": drawdown_obj,
        "max_drawdown": max_dd,
        "trade_volatility": float(p.std(ddof=0)),
        "daily_volatility": float(daily.std(ddof=0)) if len(daily) > 1 else np.nan,
        "marginal_contribution_by_sleeve": sleeve_mean,
        "leave_one_out_contribution": loo,
        "co_drawdown_attribution": co_dd,
        "dependence_drift_over_folds": dep_drift,
    }


def compute_calibration_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, session_col: str = "session_bucket") -> Dict[str, Any]:
    out = {}
    for sess in sorted(set(ref_df.get(session_col, pd.Series(dtype=str)).dropna().astype(str)).union(set(cur_df.get(session_col, pd.Series(dtype=str)).dropna().astype(str)))):
        r = ref_df[ref_df[session_col].astype(str) == sess]
        c = cur_df[cur_df[session_col].astype(str) == sess]
        if len(r) == 0 or len(c) == 0:
            continue
        rb = _brier(pd.to_numeric(r["y_true"], errors="coerce").fillna(0).to_numpy(), pd.to_numeric(r["prob"], errors="coerce").fillna(0.5).to_numpy())
        cb = _brier(pd.to_numeric(c["y_true"], errors="coerce").fillna(0).to_numpy(), pd.to_numeric(c["prob"], errors="coerce").fillna(0.5).to_numpy())
        out[sess] = {"brier_ref": rb, "brier_cur": cb, "delta_brier": float(cb - rb)}
    return out


def compute_threshold_drift(ref_thr: pd.DataFrame, cur_thr: pd.DataFrame, session_col: str = "session_bucket") -> Dict[str, Any]:
    out = {}
    sessions = sorted(set(ref_thr[session_col].astype(str)).union(set(cur_thr[session_col].astype(str))))
    for s in sessions:
        r = ref_thr.loc[ref_thr[session_col].astype(str) == s, "threshold"]
        c = cur_thr.loc[cur_thr[session_col].astype(str) == s, "threshold"]
        if len(r) == 0 or len(c) == 0:
            continue
        out[s] = {"ref_mean": float(r.mean()), "cur_mean": float(c.mean()), "delta": float(c.mean() - r.mean())}
    return out


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r = pd.to_numeric(ref, errors="coerce").dropna()
    c = pd.to_numeric(cur, errors="coerce").dropna()
    if len(r) == 0 or len(c) == 0:
        return np.nan
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    rc, _ = np.histogram(r, bins=edges)
    cc, _ = np.histogram(c, bins=edges)
    rp = np.clip(rc / max(1, rc.sum()), 1e-6, None)
    cp = np.clip(cc / max(1, cc.sum()), 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def compute_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    feature_cols: List[str],
    session_col: str = "session_bucket",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"global": {}, "by_session": {}}
    for c in feature_cols:
        out["global"][c] = _psi(ref_df[c], cur_df[c])
    sessions = sorted(set(ref_df[session_col].astype(str)).union(set(cur_df[session_col].astype(str))))
    for s in sessions:
        rs = ref_df[ref_df[session_col].astype(str) == s]
        cs = cur_df[cur_df[session_col].astype(str) == s]
        if len(rs) == 0 or len(cs) == 0:
            continue
        out["by_session"][s] = {c: _psi(rs[c], cs[c]) for c in feature_cols}
    return out


def compute_dependence_drift(ref_corr: pd.DataFrame, cur_corr: pd.DataFrame) -> Dict[str, Any]:
    a = ref_corr.reindex(index=cur_corr.index, columns=cur_corr.columns)
    d = (cur_corr - a).abs()
    return {"mean_abs_delta": float(np.nanmean(d.to_numpy(dtype=float))), "max_abs_delta": float(np.nanmax(d.to_numpy(dtype=float)))}


def build_live_monitor_snapshot(current_preds: pd.DataFrame, recent_perf: Dict[str, Any], dependence_mats: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Build live audit snapshot for monitoring/alerts."""
    cal_drift = recent_perf.get("calibration_drift", {})
    thr_drift = recent_perf.get("threshold_drift", {})
    feat_drift = recent_perf.get("feature_drift", {})
    dep_drift = recent_perf.get("dependence_drift", {})

    redundancy_alerts = []
    sc = dependence_mats.get("score_corr")
    if sc is not None and not sc.empty:
        for i, a in enumerate(sc.index):
            for b in sc.columns[i + 1 :]:
                v = float(abs(sc.loc[a, b]))
                if v >= float(recent_perf.get("redundancy_corr_alert", 0.9)):
                    redundancy_alerts.append({"a": str(a), "b": str(b), "score_corr": v})

    drawdown_attr = recent_perf.get("portfolio_drawdown_attribution", {})
    return {
        "calibration_drift_by_session": cal_drift,
        "threshold_drift_by_session": thr_drift,
        "feature_drift_by_session": feat_drift.get("by_session", {}),
        "correlation_drift_between_sleeves": dep_drift,
        "sleeve_redundancy_alerts": redundancy_alerts,
        "portfolio_drawdown_attribution": drawdown_attr,
        "timestamp_count": int(len(current_preds)),
    }


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def run_walk_forward_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic walk-forward runner with per-fold artifacts."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df index must be DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("df index must be timezone-aware.")

    output_dir = Path(config.get("output_dir", "data/research/xau_validation_wfo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    sleeves = list(config.get("sleeves", []))
    if not sleeves:
        sleeves = [c.replace("score_", "") for c in df.columns if c.startswith("score_")]
    split_cfg = SplitConfig(
        n_splits=int(config.get("n_splits", 3)),
        val_size=int(config.get("val_size", 200)),
        max_label_horizon_bars=int(config.get("max_label_horizon_bars", 8)),
        embargo_bars=int(config.get("embargo_bars", 0)),
        min_train_size=int(config.get("min_train_size", 200)),
        label_end_col=str(config.get("label_end_col", "label_end_ts")),
    )
    splits = make_purged_walk_forward_splits(df, split_cfg)

    sleeve_metrics_by_fold: List[Dict[str, Any]] = []
    fold_portfolio_series: List[pd.Series] = []
    dep_by_fold: List[pd.DataFrame] = []

    thr_cfg = ThresholdConfig(
        min_threshold=float(config.get("min_threshold", 0.5)),
        max_threshold=float(config.get("max_threshold", 0.9)),
        step=float(config.get("threshold_step", 0.01)),
        min_trades=int(config.get("min_trades", 10)),
    )

    for f in splits:
        fold_id = int(f["fold"])
        tr = np.array(f["train_indices"], dtype=int)
        va = np.array(f["validation_indices"], dtype=int)
        train = df.iloc[tr].copy()
        valid = df.iloc[va].copy()
        fold_dir = output_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_sleeve_rows = []
        val_portfolio = pd.Series(0.0, index=valid.index)
        val_prob_map: Dict[str, pd.Series] = {}
        val_sleeve_pnl: Dict[str, pd.Series] = {}

        for s in sleeves:
            score_col = f"score_{s}"
            pnl_col = f"pnl_{s}"
            if score_col not in train.columns or pnl_col not in train.columns:
                continue
            cal = fit_session_calibrator(s, train[score_col].to_numpy(dtype=float), train["y_true"].to_numpy(dtype=float))
            p_tr = predict_calibrated_prob(cal, train[score_col].to_numpy(dtype=float))
            p_va = predict_calibrated_prob(cal, valid[score_col].to_numpy(dtype=float))
            th = fit_session_threshold(s, p_tr, train[pnl_col].to_numpy(dtype=float), thr_cfg)
            sig_va = apply_session_threshold(s, p_va, th, cost_state=None)
            sleeve_pnl_va = pd.Series(sig_va.astype(float), index=valid.index) * pd.to_numeric(valid[pnl_col], errors="coerce").fillna(0.0)

            preds = pd.DataFrame(
                {
                    "prob": p_va,
                    "signal": sig_va,
                    "threshold": float(th["threshold"]),
                    "fold": fold_id,
                },
                index=valid.index,
            )
            lbl = valid[["y_true"]].copy()
            if "regime_bucket" in valid.columns:
                lbl["regime_bucket"] = valid["regime_bucket"]
            sm = evaluate_sleeve_metrics(preds, lbl, sleeve_pnl_va)
            sm["session"] = s
            sm["fold"] = fold_id
            fold_sleeve_rows.append(sm)

            val_prob_map[s] = pd.Series(p_va, index=valid.index)
            val_sleeve_pnl[s] = sleeve_pnl_va
            val_portfolio = val_portfolio.add(sleeve_pnl_va, fill_value=0.0)

            _save_json(fold_dir / f"calibrator_{s}.json", cal)
            _save_json(fold_dir / f"threshold_{s}.json", th)

        if val_prob_map:
            dep = pd.DataFrame(val_prob_map).corr()
            dep_by_fold.append(dep)
            dep.to_csv(fold_dir / "dependence_corr.csv")
        fold_portfolio_series.append(val_portfolio)
        _save_json(fold_dir / "sleeve_metrics.json", fold_sleeve_rows)
        pd.DataFrame({"timestamp": valid.index, "portfolio_pnl": val_portfolio.values}).to_csv(fold_dir / "portfolio_pnl.csv", index=False)
        sleeve_metrics_by_fold.extend(fold_sleeve_rows)

    if len(fold_portfolio_series) == 0:
        return {"sleeve_summary": {}, "portfolio_summary": {}, "folds": 0}

    portfolio_pnl = pd.concat(fold_portfolio_series).sort_index()
    # aggregate sleeve pnl map from all folds
    sleeve_map_all: Dict[str, pd.Series] = {}
    for s in sleeves:
        col = f"pnl_{s}"
        if col in df.columns:
            sleeve_map_all[s] = pd.Series(0.0, index=portfolio_pnl.index)
    for f in splits:
        va = np.array(f["validation_indices"], dtype=int)
        valid = df.iloc[va].copy()
        for s in sleeves:
            col = f"pnl_{s}"
            if col in valid.columns:
                sleeve_map_all[s].loc[valid.index] = pd.to_numeric(valid[col], errors="coerce").fillna(0.0).to_numpy()

    portfolio_summary = evaluate_portfolio_metrics(portfolio_pnl, sleeve_map_all)
    if len(dep_by_fold) >= 2:
        portfolio_summary["dependence_drift_over_folds"] = compute_dependence_drift(dep_by_fold[0], dep_by_fold[-1])

    sleeve_summary_df = pd.DataFrame(sleeve_metrics_by_fold)
    sleeve_summary = sleeve_summary_df.to_dict(orient="records")
    _save_json(output_dir / "sleeve_summary.json", sleeve_summary)
    _save_json(output_dir / "portfolio_summary.json", portfolio_summary)

    return {
        "sleeve_summary": sleeve_summary,
        "portfolio_summary": portfolio_summary,
        "folds": len(splits),
        "output_dir": str(output_dir),
    }
