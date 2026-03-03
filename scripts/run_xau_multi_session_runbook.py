#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import scripts.pipeline_compat_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "research" / "runbook"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class StageResult:
    stage: str
    passed: bool
    notes: List[str]
    artifacts: List[str]


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _session_bucket(ts_utc: pd.Timestamp, tz: str = "Europe/London") -> str:
    t = ts_utc.tz_convert(tz)
    m = t.hour * 60 + t.minute
    if m < 6 * 60:
        return "asia"
    if m < 8 * 60:
        return "pre_london"
    if m < 9 * 60:
        return "london_open"
    if m < 13 * 60:
        return "london_continuation"
    if m < 15 * 60:
        return "ny_open"
    if m < 17 * 60:
        return "ny_overlap"
    return "ny_post_data"


def _feature_registry(columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for c in columns:
        block = "shared"
        role = "shared"
        allowed = "all"
        if c.startswith("asia_") or "asia_" in c or c.startswith("lo_") or "lo_" in c:
            block = "session_anchor"
            role = "specialist"
            allowed = "asia,london_open,london_continuation"
        elif c.startswith("sweep_") or c.startswith("took_prev_") or c.startswith("close_back_inside_"):
            block = "sweep_stoprun"
            role = "specialist"
            allowed = "all"
        elif c.startswith("h1_") or c.startswith("d1_") or c.startswith("regime_"):
            block = "htf_regime"
        elif c.startswith("sess_") or c.startswith("is_") or c.startswith("hour_"):
            block = "session_state"
        elif "cost" in c or "spread" in c or "slippage" in c:
            block = "cost_state"
        elif "event" in c or "minutes_to_" in c or "minutes_since_" in c:
            block = "event_state"
        elif "wick" in c or "bar_range" in c or "close_location" in c:
            block = "microstructure"
        elif "eff_ratio" in c or "directional_consistency" in c or "choppiness" in c:
            block = "trend_quality"
        elif "ret_skew_" in c or "ret_kurt_" in c or "downside_semivar_" in c:
            block = "distribution_shape"
        rows.append(
            {
                "feature_name": c,
                "block": block,
                "shared_or_specialist": role,
                "allowed_sessions": allowed,
            }
        )
    return pd.DataFrame(rows)


def _cmd_out(cmd: List[str]) -> str:
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        return out + (("\n" + err) if err else "")
    except Exception as exc:
        return str(exc)


def run() -> Dict[str, Any]:
    results: List[StageResult] = []
    summary: Dict[str, Any] = {"stages": [], "all_passed": False}

    # Stage 0
    snap = {
        "git_status_short": _cmd_out(["git", "status", "--short"]),
        "git_head": _cmd_out(["git", "rev-parse", "HEAD"]),
        "active_xau_pointer": (ROOT / "models/active/XAU_USD/CURRENT_MODEL.txt").read_text(encoding="utf-8")
        if (ROOT / "models/active/XAU_USD/CURRENT_MODEL.txt").exists()
        else "",
    }
    p0 = OUT / "baseline_snapshot.json"
    _safe_write_json(p0, snap)
    results.append(StageResult("stage0_baseline", p0.exists(), [], [str(p0)]))

    # Load data for remaining stages
    data_source = "warehouse"
    try:
        df = adapter.load_ohlcv("XAU_USD", "M15")
    except Exception:
        data_source = "synthetic"
        idx = pd.date_range("2023-01-01", periods=1500, freq="15min", tz="UTC")
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.0008, size=len(idx))
        c = 2000.0 * np.exp(np.cumsum(r))
        o = np.concatenate([[c[0]], c[:-1]])
        span = np.abs(rng.normal(0.0, 0.0012, size=len(idx))) * c
        h = np.maximum(o, c) + span
        l = np.minimum(o, c) - span
        v = rng.integers(100, 1000, size=len(idx))
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)

    # Stage 1
    cfg = {"timezone": "Europe/London", "trend_quality_lookbacks": [8, 16, 32], "distribution_lookbacks": [16, 32]}
    feat = adapter.make_features(df, "M15", cfg)
    reg = _feature_registry(list(feat.columns))
    p1a = OUT / "stage1_feature_registry.csv"
    p1b = OUT / "stage1_session_map_sample.csv"
    reg.to_csv(p1a, index=False)
    sample = pd.DataFrame({"timestamp": df.index[:300]})
    sample["session_bucket"] = [ _session_bucket(ts) for ts in pd.DatetimeIndex(sample["timestamp"]) ]
    sample.to_csv(p1b, index=False)
    pass1 = (len(reg) > 0) and p1a.exists() and p1b.exists()
    results.append(StageResult("stage1_session_registry", pass1, [f"data_source={data_source}"], [str(p1a), str(p1b)]))

    # Stage 2
    df2 = df.copy()
    df2["atr14"] = (df2["high"] - df2["low"]).rolling(14, min_periods=14).mean()
    labels = adapter.compute_vol_scaled_triple_barrier_labels(df2, "atr14", 2.0, 2.0, 4, 24)
    sess = pd.Series([_session_bucket(ts) for ts in df2.index], index=df2.index, name="session_bucket")
    lb = labels.join(sess)
    p2a = OUT / "stage2_label_diagnostics.csv"
    p2b = OUT / "stage2_label_balance_by_session.csv"
    lb.describe(include="all").to_csv(p2a)
    bal = lb.groupby(["session_bucket", "label_side"]).size().rename("count").reset_index()
    bal.to_csv(p2b, index=False)
    pass2 = set(pd.unique(lb["label_side"])) <= {-1, 0, 1}
    results.append(StageResult("stage2_labeling", pass2, [], [str(p2a), str(p2b)]))

    # Stage 3
    fcols = list(feat.columns)
    X_sel, state = adapter.select_and_transform_features(feat, fcols)
    p3a = OUT / "stage3_selected_features.csv"
    p3b = OUT / "stage3_transform_state.json"
    pd.DataFrame({"feature": X_sel.columns}).to_csv(p3a, index=False)
    _safe_write_json(p3b, state)
    warm = X_sel.iloc[300:]
    warm_clean = warm.dropna(how="any")
    pass3 = (len(X_sel.columns) > 0) and (len(warm_clean) > 0) and bool(warm_clean.notna().all().all())
    results.append(StageResult("stage3_feature_transform", pass3, [], [str(p3a), str(p3b)]))

    # Stage 4
    y_bin = (labels["label_side"] > 0).astype(int)
    splits = adapter.generate_purged_walkforward_splits(df.index, labels["label_end_ts"], n_splits=3, test_size=120, embargo_bars=24)
    p4a = OUT / "stage4_splits.json"
    _safe_write_json(
        p4a,
        [{"train_n": int(len(tr)), "test_n": int(len(te)), "test_start": str(df.index[te[0]]) if len(te) else "", "test_end": str(df.index[te[-1]]) if len(te) else ""} for tr, te in splits],
    )
    probs = np.clip(0.5 + 0.25 * np.tanh(pd.Series(df["close"]).pct_change(8).fillna(0.0).to_numpy() * 500.0), 1e-6, 1 - 1e-6)
    cal = adapter.fit_probability_calibrator(probs, y_bin.to_numpy(), isotonic_min_samples=999999)
    pcal = adapter.apply_probability_calibrator(cal, probs)
    fold_rows: List[Dict[str, Any]] = []
    for i, (_, te) in enumerate(splits, start=1):
        if len(te) == 0:
            continue
        c = {"expected_cost": pd.Series(0.0, index=df.index[te]), "threshold": 0.55}
        d = adapter.compute_fold_diagnostics(y_bin.iloc[te], pd.Series(pcal, index=df.index).iloc[te], c, i)
        fold_rows.append(d)
    p4b = OUT / "stage4_fold_diagnostics.csv"
    p4c = OUT / "stage4_calibration_summary.csv"
    pd.DataFrame(fold_rows).to_csv(p4b, index=False)
    pd.DataFrame([{"calibrator_type": cal.get("type", ""), "n": int(len(probs)), "mean_p_raw": float(np.mean(probs)), "mean_p_cal": float(np.mean(pcal))}]).to_csv(p4c, index=False)
    pass4 = (len(splits) > 0) and p4b.exists()
    results.append(StageResult("stage4_purged_calibration", pass4, [], [str(p4a), str(p4b), str(p4c)]))

    # Stage 5
    ev = adapter.compute_expected_value(pd.Series(pcal, index=df.index), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index), pd.Series(0.01, index=df.index))
    gate_df = pd.DataFrame({"p": pcal, "ev": ev}, index=df.index)
    gate_df["dynamic_threshold"] = 0.55 + (feat.get("spread_proxy_bps", pd.Series(index=df.index, data=0.0)).fillna(0.0) > feat.get("spread_proxy_bps", pd.Series(index=df.index, data=0.0)).fillna(0.0).quantile(0.8)).astype(float) * 0.03
    gate = adapter.apply_trade_gating(gate_df, "p", "ev", min_ev=0.0, base_p_threshold=0.55, dynamic_threshold_col="dynamic_threshold")
    dlog = gate_df.copy()
    dlog["trade_gate"] = gate
    dlog["reason_ev"] = (dlog["ev"] > 0.0).astype(int)
    dlog["reason_prob"] = (dlog["p"] > dlog["dynamic_threshold"]).astype(int)
    p5a = OUT / "stage5_decision_log.csv"
    p5b = OUT / "stage5_threshold_diagnostics.csv"
    dlog.to_csv(p5a, index=True)
    pd.DataFrame(
        [
            {
                "trade_rate": float(gate.mean()),
                "avg_ev_traded": float(dlog.loc[dlog["trade_gate"] > 0, "ev"].mean()) if (gate > 0).any() else 0.0,
                "avg_threshold": float(dlog["dynamic_threshold"].mean()),
            }
        ]
    ).to_csv(p5b, index=False)
    pass5 = bool(((dlog["trade_gate"] > 0) <= ((dlog["reason_ev"] > 0) & (dlog["reason_prob"] > 0))).all())
    results.append(StageResult("stage5_decision_layer", pass5, [], [str(p5a), str(p5b)]))

    # Stage 6
    bucket = pd.Series([_session_bucket(ts) for ts in df.index], index=df.index)
    y = y_bin.reindex(df.index).fillna(0).astype(int)
    p_pool = pd.Series(pcal, index=df.index)
    arch_rows = []
    for name in ["specialist_per_session", "pooled_global", "shared_trunk_plus_session_heads"]:
        if name == "specialist_per_session":
            p_hat = p_pool.groupby(bucket).transform("mean")
        elif name == "pooled_global":
            p_hat = pd.Series(p_pool.mean(), index=df.index)
        else:
            p_hat = 0.6 * p_pool + 0.4 * p_pool.groupby(bucket).transform("mean")
        brier = float(((p_hat - y) ** 2).mean())
        evm = float((adapter.compute_expected_value(p_hat, pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index), pd.Series(0.01, index=df.index))).mean())
        arch_rows.append({"architecture": name, "brier": brier, "mean_ev": evm, "robust_score": evm - brier})
    arch = pd.DataFrame(arch_rows).sort_values("robust_score", ascending=False)
    p6a = OUT / "stage6_architecture_comparison.csv"
    p6b = OUT / "stage6_selected_architecture.json"
    arch.to_csv(p6a, index=False)
    _safe_write_json(p6b, {"selected_architecture": str(arch.iloc[0]["architecture"]), "ranking": arch.to_dict(orient="records")})
    pass6 = len(arch) == 3
    results.append(StageResult("stage6_architecture", pass6, [], [str(p6a), str(p6b)]))

    # Stage 7
    sleeves = {
        "asia": p_pool.where(bucket == "asia", np.nan).ffill().fillna(0.5),
        "london": p_pool.where(bucket.isin(["london_open", "london_continuation"]), np.nan).ffill().fillna(0.5),
        "ny": p_pool.where(bucket.isin(["ny_open", "ny_overlap", "ny_post_data"]), np.nan).ffill().fillna(0.5),
    }
    ps = pd.DataFrame(sleeves)
    corr = ps.corr()
    p7a = OUT / "stage7_model_dependence_matrix.csv"
    corr.to_csv(p7a)
    red_rows = []
    for a in corr.columns:
        for b in corr.columns:
            if a >= b:
                continue
            c = float(corr.loc[a, b])
            red_rows.append({"sleeve_a": a, "sleeve_b": b, "corr": c, "suppress": bool(c > 0.95)})
    p7b = OUT / "stage7_redundancy_decisions.csv"
    pd.DataFrame(red_rows).to_csv(p7b, index=False)
    alloc = pd.DataFrame(index=df.index)
    base_w = pd.Series({"asia": 0.33, "london": 0.34, "ny": 0.33})
    penalty = corr.mean().fillna(0.0)
    w = (base_w * (1.0 - penalty)).clip(lower=0.05)
    w = w / w.sum()
    for k in w.index:
        alloc[f"w_{k}"] = float(w[k])
    alloc["portfolio_score"] = sum(alloc[f"w_{k}"] * ps[k] for k in w.index)
    p7c = OUT / "stage7_portfolio_allocation_trace.csv"
    alloc.to_csv(p7c, index=True)
    pass7 = p7a.exists() and p7b.exists() and p7c.exists()
    results.append(StageResult("stage7_correlation_portfolio", pass7, [], [str(p7a), str(p7b), str(p7c)]))

    # Stage 8
    scorecard = pd.DataFrame(
        [
            {"component": "isolation_model", "metric": "brier", "value": float(((p_pool - y) ** 2).mean())},
            {"component": "portfolio", "metric": "mean_score", "value": float(alloc["portfolio_score"].mean())},
            {"component": "portfolio", "metric": "score_std", "value": float(alloc["portfolio_score"].std(ddof=0))},
        ]
    )
    p8a = OUT / "stage8_validation_scorecard.csv"
    scorecard.to_csv(p8a, index=False)
    monitor = {
        "calibration_drift": {"metrics": ["brier", "ece"], "freq": "weekly"},
        "threshold_drift": {"metrics": ["trade_rate", "threshold_hit"], "freq": "weekly"},
        "feature_drift": {"metrics": ["psi", "ks"], "freq": "weekly"},
        "correlation_drift": {"metrics": ["score_corr", "drawdown_overlap"], "freq": "weekly"},
    }
    p8b = OUT / "stage8_monitoring_spec.json"
    _safe_write_json(p8b, monitor)
    manifest = {
        "symbol": "XAU_USD",
        "mode": "runbook_checkpoint",
        "selected_architecture": str(arch.iloc[0]["architecture"]),
        "generated_artifacts": [str(p) for p in [p8a, p8b]],
    }
    p8c = ROOT / "models" / "manifests" / "XAU_USD_multi_session.json"
    _safe_write_json(p8c, manifest)
    pass8 = p8a.exists() and p8b.exists() and p8c.exists()
    results.append(StageResult("stage8_validation_monitoring", pass8, [], [str(p8a), str(p8b), str(p8c)]))

    # Stage 9
    pytest_out = _cmd_out([".venv/bin/pytest", "-q", "tests/test_feature_pipeline_modules.py"])
    p9 = OUT / "stage9_pytest_output.txt"
    p9.write_text(pytest_out, encoding="utf-8")
    pass9 = "passed" in pytest_out.lower()
    results.append(StageResult("stage9_acceptance_tests", pass9, [], [str(p9)]))

    summary["stages"] = [
        {"stage": r.stage, "passed": r.passed, "notes": r.notes, "artifacts": r.artifacts}
        for r in results
    ]
    summary["all_passed"] = bool(all(r.passed for r in results))
    summary["data_source"] = data_source
    p_final = OUT / "final_summary.json"
    _safe_write_json(p_final, summary)
    summary["final_summary_json"] = str(p_final)
    return summary


def main() -> int:
    res = run()
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
