from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_portfolio_control as xpc


def _dep_mats() -> dict:
    sleeves = ["asia", "london", "ny"]
    score = pd.DataFrame(
        [[1.0, 0.95, 0.2], [0.95, 1.0, 0.25], [0.2, 0.25, 1.0]],
        index=sleeves,
        columns=sleeves,
    )
    overlap = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.15], [0.1, 0.15, 1.0]],
        index=sleeves,
        columns=sleeves,
    )
    pnl = pd.DataFrame(
        [[1.0, 0.85, 0.2], [0.85, 1.0, 0.2], [0.2, 0.2, 1.0]],
        index=sleeves,
        columns=sleeves,
    )
    coloss = pd.DataFrame(
        [[1.0, 0.7, 0.05], [0.7, 1.0, 0.05], [0.05, 0.05, 1.0]],
        index=sleeves,
        columns=sleeves,
    )
    return {"score_corr": score, "trigger_overlap": overlap, "pnl_corr": pnl, "coloss_freq": coloss}


def test_higher_utility_redundant_sleeve_survives():
    cand = pd.DataFrame(
        {
            "sleeve": ["asia", "london"],
            "signal": [1, 1],
            "prob": [0.7, 0.65],
            "payoff_estimate": [1.2, 0.9],
            "cost_estimate": [0.1, 0.1],
        }
    )
    health = pd.DataFrame({"sleeve": ["asia", "london"], "recent_brier": [0.10, 0.08]})
    cfg = {"redundant_pair_penalty_threshold": 1.0}
    out = xpc.apply_redundancy_suppression(cand, _dep_mats(), health, cfg)
    kept = out.loc[~out["suppressed"], "sleeve"].tolist()
    assert kept == ["asia"]


def test_lower_utility_redundant_sleeve_suppressed():
    cand = pd.DataFrame(
        {
            "sleeve": ["asia", "london"],
            "signal": [1, 1],
            "prob": [0.55, 0.75],
            "payoff_estimate": [0.8, 1.3],
            "cost_estimate": [0.1, 0.1],
        }
    )
    health = pd.DataFrame({"sleeve": ["asia", "london"], "recent_brier": [0.10, 0.10]})
    cfg = {"redundant_pair_penalty_threshold": 1.0}
    out = xpc.apply_redundancy_suppression(cand, _dep_mats(), health, cfg)
    suppressed = out.loc[out["suppressed"], "sleeve"].tolist()
    assert suppressed == ["asia"]


def test_risk_caps_enforced():
    cand = pd.DataFrame(
        {
            "sleeve": ["asia", "london", "ny"],
            "signal": [1, 1, 1],
            "expected_utility": [0.20, 0.18, 0.05],
            "suppressed": [False, False, False],
            "cluster": ["eu", "eu", "us"],
        }
    )
    alloc = xpc.allocate_sleeve_risk(
        cand,
        portfolio_state={"drawdown_pct": 0.02},
        dependence_mats={},
        alloc_config={
            "max_risk_per_sleeve": 0.30,
            "max_risk_per_cluster": 0.45,
            "max_total_concurrent_sleeves": 3,
            "max_total_risk": 0.8,
            "dd_soft_cap_pct": 0.08,
            "dd_hard_cap_pct": 0.15,
        },
    )
    assert float(alloc["risk_alloc"].sum()) <= 0.8 + 1e-9
    assert float(alloc.loc[alloc["sleeve"].isin(["asia", "london"]), "risk_alloc"].sum()) <= 0.45 + 1e-9
    assert float(alloc["risk_alloc"].max()) <= 0.30 + 1e-9


def test_conflicting_sleeves_handled_deterministically():
    sleeves = pd.DataFrame(
        {
            "sleeve": ["asia", "ny"],
            "signal": [1, -1],
            "prob": [0.75, 0.72],
            "payoff_estimate": [1.0, 0.9],
            "cost_estimate": [0.1, 0.1],
            "cluster": ["eu", "us"],
        }
    )
    ctrl = {
        "dependence_mats": _dep_mats(),
        "alloc_config": {
            "redundant_pair_penalty_threshold": 99.0,
            "max_risk_per_sleeve": 0.5,
            "max_risk_per_cluster": 0.5,
            "max_total_concurrent_sleeves": 2,
            "max_total_risk": 1.0,
        },
        "sleeve_health": pd.DataFrame({"sleeve": ["asia", "ny"], "recent_brier": [0.1, 0.1]}),
        "portfolio_state": {"drawdown_pct": 0.02},
    }
    d1 = xpc.combine_session_outputs("2025-01-01T10:00:00Z", sleeves, ctrl)
    d2 = xpc.combine_session_outputs("2025-01-01T10:00:00Z", sleeves, ctrl)
    assert d1["final_action"] in {"LONG", "SHORT", "NO_TRADE"}
    assert d1 == d2


def test_final_payload_has_audit_fields():
    sleeves = pd.DataFrame(
        {
            "sleeve": ["asia", "london"],
            "signal": [1, 1],
            "prob": [0.8, 0.7],
            "payoff_estimate": [1.0, 0.9],
            "cost_estimate": [0.1, 0.1],
            "cluster": ["eu", "eu"],
        }
    )
    ctrl = {
        "dependence_mats": _dep_mats(),
        "alloc_config": {
            "redundant_pair_penalty_threshold": 1.0,
            "max_risk_per_sleeve": 0.4,
            "max_risk_per_cluster": 0.4,
            "max_total_concurrent_sleeves": 2,
            "max_total_risk": 0.6,
        },
        "sleeve_health": pd.DataFrame({"sleeve": ["asia", "london"], "recent_brier": [0.1, 0.2]}),
        "portfolio_state": {"drawdown_pct": 0.02},
    }
    d = xpc.combine_session_outputs("2025-01-01T10:00:00Z", sleeves, ctrl)
    for k in ["final_action", "selected_sleeves", "size_multiplier", "suppressed_sleeves", "suppression_reasons", "dependence_snapshot", "decision_metrics"]:
        assert k in d
