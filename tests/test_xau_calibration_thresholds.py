from __future__ import annotations

import numpy as np

import scripts.xau_calibration_thresholds as xct


def test_calibrators_differ_by_session_for_different_score_distributions():
    rng = np.random.default_rng(7)
    s_asia = rng.normal(-0.5, 0.6, size=400)
    y_asia = (rng.uniform(size=400) < (1.0 / (1.0 + np.exp(-s_asia)))).astype(int)
    s_ny = rng.normal(0.8, 0.7, size=400)
    y_ny = (rng.uniform(size=400) < (1.0 / (1.0 + np.exp(-s_ny * 1.4)))).astype(int)

    c1 = xct.fit_session_calibrator("asia", s_asia, y_asia)
    c2 = xct.fit_session_calibrator("ny_open", s_ny, y_ny)
    assert (c1["a"], c1["b"]) != (c2["a"], c2["b"])
    assert "brier" in c1["diagnostics"] and "ece" in c1["diagnostics"]


def test_thresholding_can_emit_abstain():
    p = np.array([0.52, 0.56, 0.71, 0.49, 0.68], dtype=float)
    ev = np.array([0.0, 0.02, 0.03, -0.01, 0.01], dtype=float)
    cfg = xct.ThresholdConfig(min_threshold=0.55, max_threshold=0.80, step=0.05, min_trades=1, abstain_value=0)
    th = xct.fit_session_threshold("asia", p, ev, cfg)
    sig = xct.apply_session_threshold("asia", p, th, cost_state=None)
    assert (sig == 0).any()
    assert ((sig == 1) | (sig == -1)).any()


def test_fragile_spike_penalized_vs_broad_plateau():
    grid = np.arange(0.50, 0.91, 0.01)
    # Construct p values on grid-like support.
    p = np.repeat(grid, 10)
    # EV landscape: broad plateau around 0.65 and one sharp spike at 0.80.
    ev_map = np.full_like(grid, 0.02, dtype=float)
    ev_map[(grid >= 0.62) & (grid <= 0.69)] = 0.06
    spike_i = np.argmin(np.abs(grid - 0.80))
    ev_map[spike_i] = 0.09
    ev = np.repeat(ev_map, 10)

    cfg = xct.ThresholdConfig(
        min_threshold=0.50,
        max_threshold=0.90,
        step=0.01,
        min_trades=1,
        smoothness_penalty=0.4,
        spike_penalty=0.8,
        neighborhood=2,
    )
    th = xct.fit_session_threshold("london_open", p, ev, cfg)
    # should prefer plateau zone instead of isolated spike
    assert 0.60 <= th["threshold"] <= 0.70


def test_threshold_selection_is_deterministic():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.45, 0.9, size=600)
    ev = 0.04 * (p - 0.55) + rng.normal(0.0, 0.005, size=600)
    cfg = xct.ThresholdConfig(min_threshold=0.5, max_threshold=0.85, step=0.01, min_trades=10)
    t1 = xct.fit_session_threshold("ny_open", p, ev, cfg)
    t2 = xct.fit_session_threshold("ny_open", p, ev, cfg)
    assert t1["threshold"] == t2["threshold"]
    assert t1["diagnostics"]["chosen_index"] == t2["diagnostics"]["chosen_index"]


def test_calibration_reliability_fallback_on_constant_scores():
    s = np.zeros(200, dtype=float)
    y = np.array([0, 1] * 100, dtype=int)
    c = xct.fit_session_calibrator("asia", s, y)
    assert np.isfinite(c["diagnostics"]["reliability_intercept"])


def test_cost_conditioned_thresholds_are_bounded_and_deterministic():
    rng = np.random.default_rng(12)
    n = 900
    p = rng.uniform(0.45, 0.9, size=n)
    ev = 0.03 * (p - 0.55) + rng.normal(0.0, 0.004, size=n)
    st = np.where(np.arange(n) % 2 == 0, "asia", "london_open")
    spread = rng.uniform(1.0, 4.0, size=n)
    spread_atr = rng.uniform(0.01, 0.25, size=n)
    cdf = __import__("pandas").DataFrame({"session_bucket": st, "spread_proxy": spread, "spread_atr": spread_atr})
    cfg = xct.ThresholdConfig(
        min_threshold=0.5,
        max_threshold=0.85,
        step=0.01,
        min_trades=20,
        cost_conditioning_enabled=True,
        cost_state_min_samples=50,
        cost_state_max_adjustment=0.04,
    )
    t1 = xct.fit_session_threshold("asia", p, ev, cfg, cost_state_df=cdf)
    t2 = xct.fit_session_threshold("asia", p, ev, cfg, cost_state_df=cdf)
    assert t1["threshold"] == t2["threshold"]
    cc = t1["cost_conditioning"]
    assert cc["enabled"] is True
    base = float(t1["threshold_long"])
    for _, v in cc["state_thresholds"].items():
        if v.get("fallback_to_baseline", False):
            continue
        if np.isfinite(v.get("threshold_long", np.nan)):
            assert abs(float(v["threshold_long"]) - base) <= 0.0400001


def test_cost_conditioned_threshold_fallback_on_low_samples():
    p = np.linspace(0.45, 0.8, 40)
    ev = np.linspace(-0.01, 0.02, 40)
    cdf = __import__("pandas").DataFrame(
        {
            "session_bucket": ["asia"] * 40,
            "spread_proxy": np.linspace(1.0, 3.0, 40),
            "spread_atr": np.linspace(0.01, 0.2, 40),
        }
    )
    cfg = xct.ThresholdConfig(
        min_threshold=0.5,
        max_threshold=0.85,
        step=0.01,
        min_trades=5,
        cost_conditioning_enabled=True,
        cost_state_min_samples=1000,
    )
    th = xct.fit_session_threshold("asia", p, ev, cfg, cost_state_df=cdf)
    cc = th["cost_conditioning"]
    assert cc["enabled"] is True
    assert all(bool(v.get("fallback_to_baseline", False)) for v in cc["state_thresholds"].values())


def test_apply_threshold_respects_disabled_side_and_abstain():
    p = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    th = {
        "threshold": 0.6,
        "threshold_long": 0.6,
        "threshold_short": 0.3,
        "long_enabled": False,
        "short_enabled": True,
        "abstain_value": 0,
    }
    sig = xct.apply_session_threshold("asia", p, th, cost_state=None)
    assert (sig[p > 0.6] == 0).all()
    assert (sig[p <= 0.3] == -1).all()


def test_cost_conditioning_uses_fixed_custom_edges_when_provided():
    rng = np.random.default_rng(99)
    n = 600
    p = rng.uniform(0.45, 0.9, size=n)
    ev = 0.03 * (p - 0.55) + rng.normal(0.0, 0.004, size=n)
    cdf = __import__("pandas").DataFrame(
        {
            "session_bucket": ["london_open"] * n,
            "spread_proxy": rng.uniform(0.1, 0.8, size=n),
            "spread_atr": rng.uniform(0.01, 0.25, size=n),
        }
    )
    cfg = xct.ThresholdConfig(
        min_threshold=0.5,
        max_threshold=0.85,
        step=0.01,
        min_trades=10,
        cost_conditioning_enabled=True,
        cost_state_min_samples=20,
        cost_state_spread_edges=[0.2, 0.3, 0.5],
    )
    th = xct.fit_session_threshold("london_open", p, ev, cfg, cost_state_df=cdf)
    cc = th["cost_conditioning"]
    edges = np.asarray(cc["spread_edges"], dtype=float)
    # Expect custom interior cuts to be preserved.
    assert np.any(np.isclose(edges, 0.2))
    assert np.any(np.isclose(edges, 0.3))
    assert np.any(np.isclose(edges, 0.5))
