from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _regime_to_dict(regime_probs: Dict[str, float] | Iterable[float]) -> Dict[str, float]:
    if isinstance(regime_probs, dict):
        trend = float(regime_probs.get("trend", 0.0))
        range_ = float(regime_probs.get("range", 0.0))
        highvol = float(regime_probs.get("highvol", 0.0))
        lowliq = float(regime_probs.get("lowliq", 0.0))
    else:
        vals = list(regime_probs)
        trend = float(vals[0]) if len(vals) > 0 else 0.0
        range_ = float(vals[1]) if len(vals) > 1 else 0.0
        highvol = float(vals[2]) if len(vals) > 2 else 0.0
        lowliq = float(vals[3]) if len(vals) > 3 else 0.0
    return {
        "trend": _clip01(trend),
        "range": _clip01(range_),
        "highvol": _clip01(highvol),
        "lowliq": _clip01(lowliq),
    }


@dataclass
class SignalAggregator:
    """Rule-based probability aggregator with hysteresis and cooldown."""

    trade_gate: float = 0.50
    enter_threshold: float = 0.05
    exit_threshold: float = 0.02
    flip_threshold: float = 0.08
    cooldown_bars: int = 3

    side: int = 0
    bars_since_change: int = field(default=10_000)

    def reset(self) -> None:
        self.side = 0
        self.bars_since_change = 10_000

    def aggregate(
        self,
        *,
        p_trade: float,
        p_long: float,
        p_short: float,
        regime_probs: Dict[str, float] | Iterable[float],
        p_breakout: float,
        p_meanrev: float,
    ) -> Dict[str, Any]:
        self.bars_since_change += 1

        p_trade = _clip01(p_trade)
        p_long = _clip01(p_long)
        p_short = _clip01(p_short)
        p_breakout = _clip01(p_breakout)
        p_meanrev = _clip01(p_meanrev)
        r = _regime_to_dict(regime_probs)

        d = p_long - p_short
        w_break = (0.6 * r["trend"]) + (0.6 * r["highvol"])
        w_mean = 0.8 * r["range"]
        pen = 1.0 - (0.7 * r["lowliq"])
        score_raw = pen * p_trade * d * (
            1.0 + (0.5 * w_break * p_breakout) + (0.5 * w_mean * p_meanrev)
        )

        gate_pass = p_trade >= self.trade_gate
        score = score_raw if gate_pass else 0.0

        desired = self._desired_side(score)
        side_next = self._apply_cooldown(desired)

        if side_next != self.side:
            self.side = side_next
            self.bars_since_change = 0

        action = "BUY" if self.side > 0 else ("SELL" if self.side < 0 else "FLAT")
        return {
            "score": float(score),
            "side": int(self.side),
            "action": action,
            "debug": {
                "gate_pass": bool(gate_pass),
                "components": {
                    "p_trade": p_trade,
                    "p_long": p_long,
                    "p_short": p_short,
                    "d": d,
                    "trend": r["trend"],
                    "range": r["range"],
                    "highvol": r["highvol"],
                    "lowliq": r["lowliq"],
                    "w_break": w_break,
                    "w_mean": w_mean,
                    "pen": pen,
                    "p_breakout": p_breakout,
                    "p_meanrev": p_meanrev,
                    "score_raw": score_raw,
                },
                "state": {
                    "prev_side": int(self.side),
                    "bars_since_change": int(self.bars_since_change),
                    "desired_side": int(desired),
                    "cooldown_bars": int(self.cooldown_bars),
                },
            },
        }

    def _desired_side(self, score: float) -> int:
        sgn = 1 if score > 0 else (-1 if score < 0 else 0)
        abs_s = abs(score)
        if self.side == 0:
            return sgn if abs_s >= self.enter_threshold else 0
        if sgn == self.side:
            return self.side
        if abs_s <= self.exit_threshold:
            return 0
        if sgn != 0 and abs_s >= self.flip_threshold:
            return sgn
        return self.side

    def _apply_cooldown(self, desired: int) -> int:
        # During cooldown, allow flattening for safety, but block fresh entries/flips.
        if desired == self.side:
            return desired
        if self.bars_since_change < self.cooldown_bars and desired != 0:
            return self.side
        return desired
