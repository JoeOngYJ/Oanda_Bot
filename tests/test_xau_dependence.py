from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.xau_dependence as xd


def test_signal_dependence_toy_correlations():
    idx = pd.date_range("2025-01-01", periods=100, freq="15min", tz="UTC")
    a = np.linspace(0, 1, 100)
    b = a * 2.0 + 0.1
    c = np.linspace(1, 0, 100)
    pred = pd.DataFrame({"asia": a, "london": b, "ny": c}, index=idx)
    m = xd.compute_signal_dependence(pred)
    ab = [r for r in m["pairs"] if {r["a"], r["b"]} == {"asia", "london"}][0]
    assert ab["pearson"] > 0.99
    assert ab["spearman"] > 0.99


def test_trade_overlap_jaccard_toy():
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    s = pd.DataFrame(
        {
            "a": [1, 0, 1, 0, 0, 1],
            "b": [1, 1, 0, 0, 0, 1],
        },
        index=idx,
    )
    m = xd.compute_trade_overlap(s)
    r = m["pairs"][0]
    # intersection = {0,5}=2; union={0,1,2,5}=4 => 0.5
    assert abs(r["jaccard"] - 0.5) < 1e-9
    assert r["overlap_count"] == 2


def test_identical_pnl_series_max_dependence():
    idx = pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC")
    x = np.sin(np.linspace(0, 6, 200))
    pnl = pd.DataFrame({"a": x, "b": x.copy()}, index=idx)
    p = xd.compute_pnl_dependence(pnl)
    c = p["pairs"][0]
    assert c["trade_pearson"] > 0.999
    assert c["daily_pearson"] > 0.999
    co = xd.compute_codrawdown_metrics(pnl)["pairs"][0]
    assert co["drawdown_overlap_ratio"] > 0.99


def test_independent_random_low_dependence():
    rng = np.random.default_rng(123)
    idx = pd.date_range("2025-01-01", periods=1000, freq="15min", tz="UTC")
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    pred = pd.DataFrame({"a": a, "b": b}, index=idx)
    m = xd.compute_signal_dependence(pred)
    r = m["pairs"][0]
    assert abs(r["pearson"]) < 0.15
    assert abs(r["spearman"]) < 0.15


def test_redundant_candidate_flagged():
    idx = pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC")
    base_signal = pd.Series(([1, 0] * 100), index=idx)
    cand_signal = base_signal.copy()
    base_pnl = pd.Series(np.where(base_signal == 1, 0.02, 0.0), index=idx)
    cand_pnl = pd.Series(np.where(cand_signal == 1, 0.005, 0.0), index=idx)

    dep = {
        "signal": xd.compute_signal_dependence(pd.DataFrame({"base": base_signal, "cand": cand_signal}, index=idx)),
        "trigger": xd.compute_trade_overlap(pd.DataFrame({"base": base_signal, "cand": cand_signal}, index=idx)),
        "pnl": xd.compute_pnl_dependence(pd.DataFrame({"base": base_pnl, "cand": cand_pnl}, index=idx)),
        "codrawdown": xd.compute_codrawdown_metrics(pd.DataFrame({"base": base_pnl, "cand": cand_pnl}, index=idx)),
    }
    add_base = {"marginal_utility": 1.0, "unique_trade_share": 1.0}
    add_cand = xd.evaluate_sleeve_additivity(
        base_portfolio_df=pd.DataFrame({"pnl": base_pnl, "signal": base_signal}, index=idx),
        candidate_sleeve_df=pd.DataFrame({"pnl": cand_pnl, "signal": cand_signal}, index=idx),
        config=xd.AdditivityConfig(cost_per_trade=0.01, utility_risk_aversion=0.2),
    )
    flags = xd.flag_redundant_sleeves(
        dependence_metrics=dep,
        additivity_metrics={"base": add_base, "cand": add_cand},
        config={
            "high_score_corr": 0.95,
            "high_trigger_overlap": 0.95,
            "high_pnl_corr": 0.80,
            "high_coloss_freq": 0.50,
            "min_marginal_utility": 0.0,
            "min_unique_trade_share": 0.05,
        },
    )
    row = flags.loc[flags["sleeve"] == "cand"].iloc[0]
    assert bool(row["is_redundant"])
