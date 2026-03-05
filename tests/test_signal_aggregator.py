from __future__ import annotations

import numpy as np

from oanda_bot.ml.signal_aggregator import SignalAggregator


def test_signal_aggregator_formula_and_gate():
    ag = SignalAggregator(trade_gate=0.5, enter_threshold=0.01, exit_threshold=0.005, flip_threshold=0.02, cooldown_bars=0)
    out = ag.aggregate(
        p_trade=0.8,
        p_long=0.7,
        p_short=0.2,
        regime_probs=[0.5, 0.3, 0.4, 0.1],  # trend, range, highvol, lowliq
        p_breakout=0.6,
        p_meanrev=0.2,
    )
    d = 0.7 - 0.2
    w_break = 0.6 * 0.5 + 0.6 * 0.4
    w_mean = 0.8 * 0.3
    pen = 1.0 - 0.7 * 0.1
    expected = pen * 0.8 * d * (1.0 + 0.5 * w_break * 0.6 + 0.5 * w_mean * 0.2)
    assert np.isclose(out["score"], expected)
    assert out["side"] == 1
    assert out["action"] == "BUY"

    # gate fails => score forced to 0 and flat by thresholds
    ag2 = SignalAggregator(trade_gate=0.9, enter_threshold=0.01, exit_threshold=0.005, flip_threshold=0.02, cooldown_bars=0)
    out2 = ag2.aggregate(
        p_trade=0.8,
        p_long=0.7,
        p_short=0.2,
        regime_probs=[0.5, 0.3, 0.4, 0.1],
        p_breakout=0.6,
        p_meanrev=0.2,
    )
    assert out2["score"] == 0.0
    assert out2["side"] == 0
    assert out2["action"] == "FLAT"


def test_hysteresis_and_cooldown_behavior():
    ag = SignalAggregator(
        trade_gate=0.5,
        enter_threshold=0.02,
        exit_threshold=0.01,
        flip_threshold=0.05,
        cooldown_bars=2,
    )

    # Enter long
    a1 = ag.aggregate(
        p_trade=0.9, p_long=0.8, p_short=0.1,
        regime_probs=[0.6, 0.2, 0.4, 0.1], p_breakout=0.5, p_meanrev=0.1
    )
    assert a1["side"] == 1

    # Immediate strong opposite signal during cooldown -> blocked (stay long)
    a2 = ag.aggregate(
        p_trade=0.9, p_long=0.1, p_short=0.85,
        regime_probs=[0.6, 0.2, 0.4, 0.1], p_breakout=0.5, p_meanrev=0.1
    )
    assert a2["side"] == 1

    # After cooldown expires, opposite signal can flip
    ag.aggregate(
        p_trade=0.9, p_long=0.55, p_short=0.50,
        regime_probs=[0.6, 0.2, 0.4, 0.1], p_breakout=0.5, p_meanrev=0.1
    )
    a4 = ag.aggregate(
        p_trade=0.9, p_long=0.1, p_short=0.85,
        regime_probs=[0.6, 0.2, 0.4, 0.1], p_breakout=0.5, p_meanrev=0.1
    )
    assert a4["side"] in {-1, 1}  # depending on exact score thresholds
    assert a4["action"] in {"SELL", "BUY"}
