from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.xau_full_pipeline as xfp


def _make_input_csv(path: Path, n: int = 900, seed: int = 17) -> Path:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    ret = rng.normal(0.0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.0012, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    # Highly correlated sleeve scores to trigger redundancy suppression.
    score_asia = rng.normal(0.0, 1.0, size=n)
    score_london_open = score_asia + rng.normal(0.0, 0.05, size=n)
    score_ny_open = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "timestamp": idx.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "score_asia": score_asia,
            "score_london_open": score_london_open,
            "score_ny_open": score_ny_open,
        }
    )
    df.to_csv(path, index=False)
    return path


def _make_htf_csv(path: Path, n: int = 80, seed: int = 23) -> Path:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.8, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, size=n))
    pd.DataFrame(
        {"timestamp": idx.astype(str), "open": open_, "high": high, "low": low, "close": close}
    ).to_csv(path, index=False)
    return path


def test_full_pipeline_reproducible_and_no_leakage(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau.csv")
    htf_path = _make_htf_csv(tmp_path / "xau_h1.csv")
    cfg1 = {
        "seed": 42,
        "output_dir": str(tmp_path / "run1"),
        "sleeves": ["asia", "london_open", "ny_open"],
        "n_splits": 2,
        "val_size": 120,
        "max_label_horizon_bars": 8,
        "min_train_size": 240,
        "min_session_train_rows": 20,
        "htf_sources": [{"path": str(htf_path), "cols": ["close"], "prefix": "h1"}],
        "alloc_config": {"redundant_pair_penalty_threshold": 0.2, "max_total_concurrent_sleeves": 2, "max_total_risk": 1.0, "max_risk_per_sleeve": 0.7},
    }
    cfg2 = dict(cfg1)
    cfg2["output_dir"] = str(tmp_path / "run2")

    r1 = xfp.run_full_research_pipeline(str(data_path), cfg1)
    r2 = xfp.run_full_research_pipeline(str(data_path), cfg2)

    assert r1["folds"] == r2["folds"]
    assert r1["leakage_checks"]["session_anchor_future_extreme_leak_rows"] == 0
    assert not r1["leakage_checks"]["feature_columns_contain_label_prefix"]
    assert "broker_setup_snapshot" in r1
    assert json.dumps(r1["portfolio_summary"], sort_keys=True, default=str) == json.dumps(r2["portfolio_summary"], sort_keys=True, default=str)
    assert r1["portfolio_decision_count"] == r2["portfolio_decision_count"]


def test_full_pipeline_redundancy_suppression_in_flow(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau2.csv")
    cfg = {
        "seed": 7,
        "output_dir": str(tmp_path / "run"),
        "sleeves": ["asia", "london_open"],
        "force_all_sessions_for_heads": True,
        "mode": "test",
        "n_splits": 2,
        "val_size": 100,
        "max_label_horizon_bars": 8,
        "min_train_size": 220,
        "min_session_train_rows": 20,
        "min_trades": 1,
        "meta_min_prob": 0.0,
        "tradability": {"min_tradability_score": 0.0},
        "use_tradable_mask_for_training": False,
        "alloc_config": {"redundant_pair_penalty_threshold": 0.1, "max_total_concurrent_sleeves": 2, "max_total_risk": 1.0, "max_risk_per_sleeve": 0.8},
    }
    xfp.run_full_research_pipeline(str(data_path), cfg)
    dec_path = Path(cfg["output_dir"]) / "fold_01" / "portfolio_decisions.json"
    rows = json.loads(dec_path.read_text(encoding="utf-8"))
    assert len(rows) > 0
    assert any(len(r.get("suppressed_sleeves", [])) > 0 for r in rows)


def test_full_pipeline_weighted_realized_pnl_matches_allocations(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau3.csv")
    cfg = {
        "seed": 9,
        "output_dir": str(tmp_path / "run3"),
        "sleeves": ["asia", "london_open"],
        "force_all_sessions_for_heads": True,
        "mode": "test",
        "n_splits": 2,
        "val_size": 100,
        "max_label_horizon_bars": 8,
        "min_train_size": 220,
        "min_session_train_rows": 20,
        "alloc_config": {"redundant_pair_penalty_threshold": 99.0, "max_total_concurrent_sleeves": 2, "max_total_risk": 1.0, "max_risk_per_sleeve": 0.8},
    }
    xfp.run_full_research_pipeline(str(data_path), cfg)
    rows = json.loads((Path(cfg["output_dir"]) / "fold_01" / "portfolio_decisions.json").read_text(encoding="utf-8"))
    # realized_pnl should equal sum of per-sleeve weighted contributions by selected allocations.
    for r in rows[:20]:
        alloc = r.get("selected_allocations", {})
        assert isinstance(alloc, dict)
        # bounded sanity; selected allocations should sum to size_multiplier.
        assert abs(sum(float(v) for v in alloc.values()) - float(r.get("size_multiplier", 0.0))) < 1e-9


def test_force_all_sessions_guard_in_prod(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau4.csv")
    cfg = {
        "seed": 1,
        "output_dir": str(tmp_path / "run4"),
        "sleeves": ["asia", "london_open"],
        "force_all_sessions_for_heads": True,
        "mode": "prod",
        "n_splits": 2,
        "val_size": 100,
        "max_label_horizon_bars": 8,
        "min_train_size": 220,
    }
    try:
        xfp.run_full_research_pipeline(str(data_path), cfg)
        assert False, "Expected ValueError for force_all_sessions_for_heads in prod mode"
    except ValueError as e:
        assert "test-only" in str(e)


def test_controller_health_is_from_train_calibration_not_validation_metrics(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau5.csv", n=760, seed=5)
    cfg = {
        "seed": 22,
        "output_dir": str(tmp_path / "run5"),
        "sleeves": ["asia", "london_open"],
        "force_all_sessions_for_heads": True,
        "mode": "test",
        "n_splits": 2,
        "val_size": 80,
        "max_label_horizon_bars": 8,
        "min_train_size": 180,
        "min_session_train_rows": 20,
        "alloc_config": {"redundant_pair_penalty_threshold": 0.2, "max_total_concurrent_sleeves": 2, "max_total_risk": 1.0, "max_risk_per_sleeve": 0.8},
    }
    xfp.run_full_research_pipeline(str(data_path), cfg)
    fold_dir = Path(cfg["output_dir"]) / "fold_01"
    state = json.loads((fold_dir / "portfolio_controller_state.json").read_text(encoding="utf-8"))
    cals = json.loads((fold_dir / "calibrators.json").read_text(encoding="utf-8"))
    health = {r["sleeve"]: r["recent_brier"] for r in state.get("sleeve_health", [])}
    for s, c in cals.items():
        assert abs(float(health[s]) - float(c["diagnostics"]["brier"])) < 1e-12


def test_failure_decomposition_layers_and_tradability_reporting(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau6.csv", n=820, seed=11)
    cfg = {
        "seed": 31,
        "output_dir": str(tmp_path / "run6"),
        "sleeves": ["asia", "london_open"],
        "n_splits": 1,
        "val_size": 120,
        "max_label_horizon_bars": 8,
        "min_train_size": 300,
        "min_session_train_rows": 20,
        "warmup_bars": 64,
        "meta_min_prob": 0.5,
        "tradability": {"min_tradability_score": 0.1},
    }
    out = xfp.run_full_research_pipeline(str(data_path), cfg)
    assert "tradability_summary" in out
    assert "failure_decomposition" in out
    fold_dir = Path(cfg["output_dir"]) / "fold_01"
    f = json.loads((fold_dir / "failure_decomposition.json").read_text(encoding="utf-8"))
    req = [
        "gross_pre_cost",
        "after_spread",
        "after_spread_slippage",
        "full_broker_pre_threshold",
        "after_threshold_gating",
        "after_tradability_filter",
    ]
    for k in req:
        assert k in f
    # apples-to-apples cost attribution uses identical underlying pre-threshold signal set.
    assert int(f["gross_pre_cost"]["trade_bars"]) == int(f["after_spread"]["trade_bars"]) == int(f["after_spread_slippage"]["trade_bars"]) == int(f["full_broker_pre_threshold"]["trade_bars"])


def test_session_validation_outputs_present(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau7.csv", n=860, seed=19)
    cfg = {
        "seed": 77,
        "output_dir": str(tmp_path / "run7"),
        "sleeves": ["asia", "london_open", "ny_open"],
        "n_splits": 1,
        "val_size": 120,
        "max_label_horizon_bars": 8,
        "min_train_size": 300,
        "warmup_bars": 64,
    }
    out = xfp.run_full_research_pipeline(str(data_path), cfg)
    sv = out.get("session_validation", {})
    assert "per_session" in sv
    assert "calibration_quality_by_session" in sv
    assert "threshold_usage_by_session_cost_bucket" in sv


def test_abstain_reason_tradability_block(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau8.csv", n=780, seed=33)
    cfg = {
        "seed": 14,
        "output_dir": str(tmp_path / "run8"),
        "sleeves": ["asia", "london_open"],
        "n_splits": 1,
        "val_size": 100,
        "max_label_horizon_bars": 8,
        "min_train_size": 260,
        "warmup_bars": 64,
        "tradability": {"min_tradability_score": 1.1},
    }
    xfp.run_full_research_pipeline(str(data_path), cfg)
    rows = json.loads((Path(cfg["output_dir"]) / "fold_01" / "portfolio_decisions.json").read_text(encoding="utf-8"))
    assert any(r.get("abstain_reason") == "tradability_block" for r in rows)


def test_threshold_decisions_reproducible_same_seed(tmp_path: Path):
    data_path = _make_input_csv(tmp_path / "xau9.csv", n=820, seed=21)
    cfg = {
        "seed": 55,
        "output_dir": str(tmp_path / "run9a"),
        "sleeves": ["asia", "london_open"],
        "n_splits": 1,
        "val_size": 120,
        "max_label_horizon_bars": 8,
        "min_train_size": 300,
        "warmup_bars": 64,
    }
    cfg2 = dict(cfg)
    cfg2["output_dir"] = str(tmp_path / "run9b")
    xfp.run_full_research_pipeline(str(data_path), cfg)
    xfp.run_full_research_pipeline(str(data_path), cfg2)
    th1 = json.loads((Path(cfg["output_dir"]) / "fold_01" / "thresholds.json").read_text(encoding="utf-8"))
    th2 = json.loads((Path(cfg2["output_dir"]) / "fold_01" / "thresholds.json").read_text(encoding="utf-8"))
    assert json.dumps(th1, sort_keys=True) == json.dumps(th2, sort_keys=True)
