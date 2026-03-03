from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def compute_sleeve_expected_utility(prob: Any, payoff_estimate: Any, cost_estimate: Any) -> float:
    """Deterministic net expected utility: E[payoff] - cost."""
    p = np.asarray(prob, dtype=float)
    pay = np.asarray(payoff_estimate, dtype=float)
    c = np.asarray(cost_estimate, dtype=float)
    val = p * pay - c
    return float(np.nanmean(val))


def _dep_lookup(mat: pd.DataFrame | None, a: str, b: str) -> float:
    if mat is None or mat.empty:
        return np.nan
    if a not in mat.index or b not in mat.columns:
        return np.nan
    return float(mat.loc[a, b])


def compute_dependence_penalty(
    active_sleeves: Iterable[str],
    dependence_mats: Dict[str, pd.DataFrame],
    alloc_config: Dict[str, float],
) -> Dict[str, Any]:
    """Compute per-sleeve dependence penalties from provided dependence matrices."""
    sleeves = list(dict.fromkeys([str(s) for s in active_sleeves]))
    if len(sleeves) == 0:
        return {"per_sleeve_penalty": {}, "pair_penalties": []}

    w_score = float(alloc_config.get("w_score_corr", 1.0))
    w_overlap = float(alloc_config.get("w_trigger_overlap", 1.0))
    w_pnl = float(alloc_config.get("w_pnl_corr", 1.0))
    w_coloss = float(alloc_config.get("w_coloss", 1.0))

    score_mat = dependence_mats.get("score_corr")
    overlap_mat = dependence_mats.get("trigger_overlap")
    pnl_mat = dependence_mats.get("pnl_corr")
    coloss_mat = dependence_mats.get("coloss_freq")

    per: Dict[str, float] = {s: 0.0 for s in sleeves}
    pair_rows: List[Dict[str, Any]] = []
    for i, a in enumerate(sleeves):
        for b in sleeves[i + 1 :]:
            sc = abs(_dep_lookup(score_mat, a, b))
            ov = _dep_lookup(overlap_mat, a, b)
            pc = abs(_dep_lookup(pnl_mat, a, b))
            cl = _dep_lookup(coloss_mat, a, b)
            sc = 0.0 if not np.isfinite(sc) else sc
            ov = 0.0 if not np.isfinite(ov) else ov
            pc = 0.0 if not np.isfinite(pc) else pc
            cl = 0.0 if not np.isfinite(cl) else cl
            pen = (w_score * sc) + (w_overlap * ov) + (w_pnl * pc) + (w_coloss * cl)
            per[a] += pen
            per[b] += pen
            pair_rows.append({"a": a, "b": b, "pair_penalty": float(pen), "score_corr": sc, "trigger_overlap": ov, "pnl_corr": pc, "coloss": cl})
    return {"per_sleeve_penalty": per, "pair_penalties": pair_rows}


def apply_redundancy_suppression(
    candidate_signals: pd.DataFrame,
    dependence_mats: Dict[str, pd.DataFrame],
    sleeve_health: pd.DataFrame,
    alloc_config: Dict[str, float],
) -> pd.DataFrame:
    """Suppress redundant sleeves by utility, calibration quality, and dependence penalty."""
    req = {"sleeve", "signal", "prob", "payoff_estimate", "cost_estimate"}
    miss = [c for c in req if c not in candidate_signals.columns]
    if miss:
        raise ValueError(f"candidate_signals missing columns: {miss}")
    out = candidate_signals.copy()
    out["suppressed"] = False
    out["suppression_reason"] = ""

    # expected utility per sleeve
    out["expected_utility"] = [
        compute_sleeve_expected_utility(p, pe, ce) for p, pe, ce in zip(out["prob"], out["payoff_estimate"], out["cost_estimate"])
    ]
    # merge health (lower brier is better)
    h = sleeve_health.copy() if sleeve_health is not None else pd.DataFrame(columns=["sleeve", "recent_brier"])
    if "recent_brier" not in h.columns:
        h["recent_brier"] = np.nan
    out = out.merge(h[["sleeve", "recent_brier"]], on="sleeve", how="left")

    active = out.loc[out["signal"] != 0, "sleeve"].astype(str).tolist()
    dep = compute_dependence_penalty(active, dependence_mats, alloc_config)
    pen_map = dep["per_sleeve_penalty"]
    out["dependence_penalty"] = out["sleeve"].map(pen_map).fillna(0.0)

    redundant_threshold = float(alloc_config.get("redundant_pair_penalty_threshold", 1.5))
    pairs = dep["pair_penalties"]
    for r in pairs:
        if float(r["pair_penalty"]) < redundant_threshold:
            continue
        a = str(r["a"])
        b = str(r["b"])
        ia = out.index[out["sleeve"] == a]
        ib = out.index[out["sleeve"] == b]
        if len(ia) == 0 or len(ib) == 0:
            continue
        ia = ia[0]
        ib = ib[0]
        if out.at[ia, "signal"] == 0 or out.at[ib, "signal"] == 0:
            continue

        # ranking: higher utility -> better calibration (lower brier) -> lower dependence penalty -> lexical sleeve name
        key_a = (
            float(out.at[ia, "expected_utility"]),
            -float(out.at[ia, "recent_brier"]) if np.isfinite(out.at[ia, "recent_brier"]) else -np.inf,
            -float(out.at[ia, "dependence_penalty"]),
            str(out.at[ia, "sleeve"]),
        )
        key_b = (
            float(out.at[ib, "expected_utility"]),
            -float(out.at[ib, "recent_brier"]) if np.isfinite(out.at[ib, "recent_brier"]) else -np.inf,
            -float(out.at[ib, "dependence_penalty"]),
            str(out.at[ib, "sleeve"]),
        )
        loser = ib if key_a >= key_b else ia
        winner = ia if loser == ib else ib
        out.at[loser, "suppressed"] = True
        out.at[loser, "suppression_reason"] = f"redundant_vs_{out.at[winner, 'sleeve']}"
        out.at[loser, "signal"] = 0
    return out


def allocate_sleeve_risk(
    candidate_signals: pd.DataFrame,
    portfolio_state: Dict[str, Any],
    dependence_mats: Dict[str, pd.DataFrame],
    alloc_config: Dict[str, float],
) -> pd.DataFrame:
    """Allocate per-sleeve risk with caps and drawdown-aware downweighting."""
    req = {"sleeve", "signal", "expected_utility"}
    miss = [c for c in req if c not in candidate_signals.columns]
    if miss:
        raise ValueError(f"candidate_signals missing columns: {miss}")
    out = candidate_signals.copy()
    out["risk_alloc"] = 0.0

    max_risk_per_sleeve = float(alloc_config.get("max_risk_per_sleeve", 0.3))
    max_concurrent = int(alloc_config.get("max_total_concurrent_sleeves", 3))
    max_risk_total = float(alloc_config.get("max_total_risk", 1.0))
    max_risk_per_cluster = float(alloc_config.get("max_risk_per_cluster", 0.5))
    dd = float(portfolio_state.get("drawdown_pct", 0.0))
    dd_soft = float(alloc_config.get("dd_soft_cap_pct", 0.08))
    dd_hard = float(alloc_config.get("dd_hard_cap_pct", 0.15))

    if dd >= dd_hard:
        dd_mult = 0.0
    elif dd <= dd_soft:
        dd_mult = 1.0
    else:
        dd_mult = max(0.0, 1.0 - ((dd - dd_soft) / max(1e-9, (dd_hard - dd_soft))))

    active = out[(out["signal"] != 0) & (~out.get("suppressed", False).astype(bool))].copy()
    if len(active) == 0 or dd_mult <= 0:
        return out

    # deterministic sort
    active = active.sort_values(["expected_utility", "sleeve"], ascending=[False, True]).head(max_concurrent)
    util = active["expected_utility"].clip(lower=0.0)
    if float(util.sum()) <= 0:
        util = pd.Series(1.0, index=active.index)
    w = util / float(util.sum())
    risk = (w * max_risk_total * dd_mult).clip(upper=max_risk_per_sleeve)

    # cluster cap
    if "cluster" in active.columns:
        for cl, idxs in active.groupby("cluster").groups.items():
            s = float(risk.loc[list(idxs)].sum())
            if s > max_risk_per_cluster:
                scale = max_risk_per_cluster / s
                risk.loc[list(idxs)] = risk.loc[list(idxs)] * scale

    out.loc[risk.index, "risk_alloc"] = risk.astype(float)
    return out


def _build_matrix_snapshot(mats: Dict[str, pd.DataFrame], sleeves: List[str]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for k, m in mats.items():
        if m is None or m.empty:
            snap[k] = {}
            continue
        sub = m.reindex(index=sleeves, columns=sleeves)
        snap[k] = sub.to_dict()
    return snap


def combine_session_outputs(timestamp: Any, sleeve_outputs: pd.DataFrame, portfolio_controller: Dict[str, Any]) -> Dict[str, Any]:
    """Final deterministic decision combiner with full audit payload."""
    dependence_mats = portfolio_controller.get("dependence_mats", {})
    alloc_config = portfolio_controller.get("alloc_config", {})
    sleeve_health = portfolio_controller.get("sleeve_health", pd.DataFrame(columns=["sleeve", "recent_brier"]))
    portfolio_state = portfolio_controller.get("portfolio_state", {})

    suppressed = apply_redundancy_suppression(sleeve_outputs, dependence_mats, sleeve_health, alloc_config)
    alloc = allocate_sleeve_risk(suppressed, portfolio_state, dependence_mats, alloc_config)
    selected = alloc[(alloc["signal"] != 0) & (alloc["risk_alloc"] > 0)].copy()

    if len(selected) == 0:
        action = "NO_TRADE"
        size_mult = 0.0
        selected_sleeves: List[str] = []
        selected_allocations: Dict[str, float] = {}
        selected_signals: Dict[str, int] = {}
    else:
        net_signed_alloc = float((selected["risk_alloc"] * selected["signal"]).sum())
        if net_signed_alloc > 0:
            action = "LONG"
        elif net_signed_alloc < 0:
            action = "SHORT"
        else:
            action = "NO_TRADE"
        size_mult = float(selected["risk_alloc"].sum()) if action != "NO_TRADE" else 0.0
        selected_sleeves = selected["sleeve"].astype(str).tolist()
        selected_allocations = {
            str(r["sleeve"]): float(r["risk_alloc"])
            for _, r in selected.iterrows()
        }
        selected_signals = {
            str(r["sleeve"]): int(r["signal"])
            for _, r in selected.iterrows()
        }

    suppressed_rows = alloc[alloc.get("suppressed", False).astype(bool)][["sleeve", "suppression_reason"]]
    dep_pen = compute_dependence_penalty(selected_sleeves, dependence_mats, alloc_config)
    payload = {
        "timestamp": str(timestamp),
        "final_action": action,
        "selected_sleeves": selected_sleeves,
        "selected_allocations": selected_allocations,
        "selected_signals": selected_signals,
        "size_multiplier": size_mult,
        "suppressed_sleeves": suppressed_rows["sleeve"].astype(str).tolist(),
        "suppression_reasons": dict(zip(suppressed_rows["sleeve"].astype(str), suppressed_rows["suppression_reason"].astype(str))),
        "dependence_snapshot": _build_matrix_snapshot(dependence_mats, list(dict.fromkeys(list(alloc["sleeve"].astype(str))))),
        "decision_metrics": {
            "selected_count": int(len(selected_sleeves)),
            "dependence_penalty_selected": dep_pen["per_sleeve_penalty"],
        },
    }
    return payload
