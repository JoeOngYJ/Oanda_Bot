from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _pairs(cols: List[str]) -> Iterable[Tuple[str, str]]:
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            yield a, b


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 2:
        return np.nan
    if float(x.iloc[:, 0].std(ddof=0)) == 0.0 or float(x.iloc[:, 1].std(ddof=0)) == 0.0:
        return np.nan
    return float(x.iloc[:, 0].corr(x.iloc[:, 1], method=method))


def compute_signal_dependence(pred_df: pd.DataFrame, high_conf_quantile: float = 0.8) -> Dict[str, Any]:
    """Compute score dependence across sleeves."""
    cols = list(pred_df.columns)
    out: Dict[str, Any] = {"pairs": []}
    for a, b in _pairs(cols):
        xa = pd.to_numeric(pred_df[a], errors="coerce")
        xb = pd.to_numeric(pred_df[b], errors="coerce")
        x = pd.concat([xa, xb], axis=1).dropna()
        if len(x) == 0:
            out["pairs"].append({"a": a, "b": b, "pearson": np.nan, "spearman": np.nan, "high_conf_corr": np.nan})
            continue
        q_a = x.iloc[:, 0].quantile(high_conf_quantile)
        q_b = x.iloc[:, 1].quantile(high_conf_quantile)
        high = x[(x.iloc[:, 0] >= q_a) | (x.iloc[:, 1] >= q_b)]
        out["pairs"].append(
            {
                "a": a,
                "b": b,
                "pearson": _safe_corr(x.iloc[:, 0], x.iloc[:, 1], "pearson"),
                "spearman": _safe_corr(x.iloc[:, 0], x.iloc[:, 1], "spearman"),
                "high_conf_corr": _safe_corr(high.iloc[:, 0], high.iloc[:, 1], "pearson") if len(high) > 1 else np.nan,
                "n": int(len(x)),
                "n_high_conf": int(len(high)),
            }
        )
    return out


def compute_trade_overlap(signals_df: pd.DataFrame, high_conf_df: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Compute trigger overlap metrics across sleeves."""
    cols = list(signals_df.columns)
    out: Dict[str, Any] = {"pairs": []}
    bin_df = signals_df.fillna(0).astype(int).ne(0).astype(int)
    high_df = None
    if high_conf_df is not None:
        high_df = high_conf_df.fillna(0).astype(int).ne(0).astype(int)
    for a, b in _pairs(cols):
        xa = bin_df[a]
        xb = bin_df[b]
        inter = int(((xa == 1) & (xb == 1)).sum())
        union = int(((xa == 1) | (xb == 1)).sum())
        j = float(inter / union) if union > 0 else np.nan
        rec = {"a": a, "b": b, "jaccard": j, "overlap_count": inter, "union_count": union}
        if high_df is not None:
            ha = high_df[a]
            hb = high_df[b]
            h_inter = int(((ha == 1) & (hb == 1)).sum())
            h_union = int(((ha == 1) | (hb == 1)).sum())
            rec["high_conf_jaccard"] = float(h_inter / h_union) if h_union > 0 else np.nan
            rec["high_conf_overlap_count"] = h_inter
        out["pairs"].append(rec)
    return out


def compute_residual_error_correlation(pred_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    """Residual correlation where residual = realized outcome - model expectation."""
    if "outcome" not in outcomes_df.columns:
        raise ValueError("outcomes_df must contain 'outcome' column.")
    y = pd.to_numeric(outcomes_df["outcome"], errors="coerce")
    res = pd.DataFrame(index=pred_df.index)
    for c in pred_df.columns:
        p = pd.to_numeric(pred_df[c], errors="coerce")
        res[f"res_{c}"] = y - p
    cols = list(res.columns)
    out: Dict[str, Any] = {"pairs": []}
    for a, b in _pairs(cols):
        out["pairs"].append(
            {
                "a": a.replace("res_", ""),
                "b": b.replace("res_", ""),
                "pearson": _safe_corr(res[a], res[b], "pearson"),
                "spearman": _safe_corr(res[a], res[b], "spearman"),
            }
        )
    return out


def compute_pnl_dependence(pnl_df: pd.DataFrame, rolling_windows: Iterable[int] = (20, 60)) -> Dict[str, Any]:
    """Compute per-trade, daily, and rolling PnL dependence."""
    cols = list(pnl_df.columns)
    out: Dict[str, Any] = {"pairs": []}
    daily = pnl_df.resample("1D").sum(min_count=1) if isinstance(pnl_df.index, pd.DatetimeIndex) else pnl_df.copy()
    for a, b in _pairs(cols):
        rec: Dict[str, Any] = {
            "a": a,
            "b": b,
            "trade_pearson": _safe_corr(pnl_df[a], pnl_df[b], "pearson"),
            "trade_spearman": _safe_corr(pnl_df[a], pnl_df[b], "spearman"),
            "daily_pearson": _safe_corr(daily[a], daily[b], "pearson"),
            "daily_spearman": _safe_corr(daily[a], daily[b], "spearman"),
            "rolling": {},
        }
        for w in rolling_windows:
            ra = pd.to_numeric(pnl_df[a], errors="coerce").rolling(int(w), min_periods=int(w)).sum()
            rb = pd.to_numeric(pnl_df[b], errors="coerce").rolling(int(w), min_periods=int(w)).sum()
            rec["rolling"][str(int(w))] = _safe_corr(ra, rb, "pearson")
        out["pairs"].append(rec)
    return out


def _drawdown_series(pnl: pd.Series) -> pd.Series:
    eq = pd.to_numeric(pnl, errors="coerce").fillna(0.0).cumsum()
    peak = eq.cummax()
    return eq - peak


def _episodes(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    idx = pd.DatetimeIndex(mask.index)
    m = mask.fillna(False).to_numpy(dtype=bool)
    ep: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = idx[i]
        if (not v) and start is not None:
            ep.append((start, idx[i - 1]))
            start = None
    if start is not None:
        ep.append((start, idx[-1]))
    return ep


def compute_codrawdown_metrics(pnl_df: pd.DataFrame, tail_q: float = 0.1) -> Dict[str, Any]:
    """Compute co-drawdown and tail-loss dependence metrics."""
    cols = list(pnl_df.columns)
    out: Dict[str, Any] = {"pairs": []}
    for a, b in _pairs(cols):
        pa = pd.to_numeric(pnl_df[a], errors="coerce")
        pb = pd.to_numeric(pnl_df[b], errors="coerce")
        xa = pa.dropna()
        xb = pb.dropna()
        q_a = xa.quantile(tail_q) if len(xa) else np.nan
        q_b = xb.quantile(tail_q) if len(xb) else np.nan
        tail_a = pa <= q_a
        tail_b = pb <= q_b
        tail_joint = int((tail_a & tail_b).sum())
        tail_union = int((tail_a | tail_b).sum())
        tail_freq = float(tail_joint / tail_union) if tail_union > 0 else np.nan

        dda = _drawdown_series(pa) < 0
        ddb = _drawdown_series(pb) < 0
        overlap = int((dda & ddb).sum())
        union = int((dda | ddb).sum())
        overlap_ratio = float(overlap / union) if union > 0 else np.nan

        ep_a = _episodes(dda)
        ep_b = _episodes(ddb)
        starts_a = {e[0] for e in ep_a}
        starts_b = {e[0] for e in ep_b}
        ends_a = {e[1] for e in ep_a}
        ends_b = {e[1] for e in ep_b}
        start_coin = float(len(starts_a & starts_b) / max(1, len(starts_a | starts_b)))
        end_coin = float(len(ends_a & ends_b) / max(1, len(ends_a | ends_b)))

        out["pairs"].append(
            {
                "a": a,
                "b": b,
                "simultaneous_worst_decile_freq": tail_freq,
                "drawdown_overlap_ratio": overlap_ratio,
                "drawdown_start_coincidence": start_coin,
                "drawdown_end_coincidence": end_coin,
            }
        )
    return out


@dataclass(frozen=True)
class AdditivityConfig:
    cost_per_trade: float = 0.0
    utility_risk_aversion: float = 0.5


def evaluate_sleeve_additivity(
    base_portfolio_df: pd.DataFrame,
    candidate_sleeve_df: pd.DataFrame,
    config: AdditivityConfig,
) -> Dict[str, Any]:
    """Evaluate candidate sleeve marginal contribution vs base portfolio."""
    for c in ["pnl", "signal"]:
        if c not in base_portfolio_df.columns or c not in candidate_sleeve_df.columns:
            raise ValueError("Both base and candidate dataframes must contain 'pnl' and 'signal'.")
    base_pnl = pd.to_numeric(base_portfolio_df["pnl"], errors="coerce").fillna(0.0)
    cand_pnl_raw = pd.to_numeric(candidate_sleeve_df["pnl"], errors="coerce").fillna(0.0)
    cand_sig = pd.to_numeric(candidate_sleeve_df["signal"], errors="coerce").fillna(0).astype(int).ne(0)
    base_sig = pd.to_numeric(base_portfolio_df["signal"], errors="coerce").fillna(0).astype(int).ne(0)

    cand_pnl = cand_pnl_raw - float(config.cost_per_trade) * cand_sig.astype(float)
    comb_pnl = base_pnl + cand_pnl

    def utility(x: pd.Series) -> float:
        return float(x.mean() - float(config.utility_risk_aversion) * x.std(ddof=0))

    base_util = utility(base_pnl)
    comb_util = utility(comb_pnl)
    marginal_util = comb_util - base_util
    marginal_expectancy = float(comb_pnl.mean() - base_pnl.mean())
    base_dd = float((_drawdown_series(base_pnl)).min())
    comb_dd = float((_drawdown_series(comb_pnl)).min())
    marginal_dd = comb_dd - base_dd
    unique_trade_share = float((cand_sig & ~base_sig).sum() / max(1, cand_sig.sum()))

    frozen = candidate_sleeve_df["pnl_frozen"] if "pnl_frozen" in candidate_sleeve_df.columns else cand_pnl_raw
    frozen = pd.to_numeric(frozen, errors="coerce").fillna(0.0) - float(config.cost_per_trade) * cand_sig.astype(float)
    frozen_comb = base_pnl + frozen
    survives_frozen = utility(frozen_comb) > base_util

    return {
        "marginal_expectancy": marginal_expectancy,
        "marginal_drawdown": marginal_dd,
        "unique_trade_share": unique_trade_share,
        "marginal_utility": marginal_util,
        "improves_utility_after_costs": bool(marginal_util > 0),
        "benefit_survives_frozen_threshold": bool(survives_frozen),
    }


def flag_redundant_sleeves(
    dependence_metrics: Dict[str, Any],
    additivity_metrics: Dict[str, Dict[str, Any]],
    config: Dict[str, float],
) -> pd.DataFrame:
    """Flag redundant sleeves using configurable thresholds."""
    recs: List[Dict[str, Any]] = []
    sig_pairs = dependence_metrics.get("signal", {}).get("pairs", [])
    trg_pairs = dependence_metrics.get("trigger", {}).get("pairs", [])
    pnl_pairs = dependence_metrics.get("pnl", {}).get("pairs", [])
    co_pairs = dependence_metrics.get("codrawdown", {}).get("pairs", [])

    def _find(pairs: List[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
        for r in pairs:
            if {r.get("a"), r.get("b")} == {a, b}:
                return r
        return {}

    sleeves = set()
    for plist in [sig_pairs, trg_pairs, pnl_pairs, co_pairs]:
        for r in plist:
            sleeves.add(r.get("a"))
            sleeves.add(r.get("b"))
    sleeves = sorted([s for s in sleeves if s is not None])

    for s in sleeves:
        reasons: List[str] = []
        for o in sleeves:
            if o == s:
                continue
            sig = _find(sig_pairs, s, o)
            trg = _find(trg_pairs, s, o)
            pnl = _find(pnl_pairs, s, o)
            co = _find(co_pairs, s, o)
            if abs(float(sig.get("pearson", np.nan))) >= float(config.get("high_score_corr", 0.9)):
                reasons.append(f"high_score_corr_vs_{o}")
            if float(trg.get("jaccard", np.nan)) >= float(config.get("high_trigger_overlap", 0.7)):
                reasons.append(f"high_trigger_overlap_vs_{o}")
            if abs(float(pnl.get("trade_pearson", np.nan))) >= float(config.get("high_pnl_corr", 0.8)):
                reasons.append(f"high_pnl_corr_vs_{o}")
            if float(co.get("simultaneous_worst_decile_freq", np.nan)) >= float(config.get("high_coloss_freq", 0.5)):
                reasons.append(f"high_coloss_vs_{o}")

        add = additivity_metrics.get(s, {})
        if float(add.get("marginal_utility", 0.0)) <= float(config.get("min_marginal_utility", 0.0)):
            reasons.append("non_positive_marginal_utility")
        if float(add.get("unique_trade_share", 1.0)) <= float(config.get("min_unique_trade_share", 0.05)):
            reasons.append("low_unique_trade_share")

        recs.append(
            {
                "sleeve": s,
                "is_redundant": bool(len(reasons) > 0),
                "reasons": ";".join(sorted(set(reasons))),
            }
        )
    return pd.DataFrame(recs).sort_values("sleeve").reset_index(drop=True)
