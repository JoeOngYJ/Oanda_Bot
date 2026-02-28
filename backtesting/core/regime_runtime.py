"""Runtime regime feature engineering and centroid-based regime prediction."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from backtesting.core.backtester import FeatureEngineer, RegimePredictor
from backtesting.data.models import OHLCVBar


@dataclass
class RegimeModel:
    feature_columns: List[str]
    train_mean: Dict[str, float]
    train_std: Dict[str, float]
    centers: np.ndarray
    regime_to_strategy: Dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "RegimeModel":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_columns=list(obj["feature_columns"]),
            train_mean={k: float(v) for k, v in obj["train_mean"].items()},
            train_std={k: float(v) for k, v in obj["train_std"].items()},
            centers=np.asarray(obj["centers"], dtype=np.float64),
            regime_to_strategy={str(k): str(v) for k, v in obj.get("regime_to_strategy", {}).items()},
        )


class RegimeFeatureEngineer(FeatureEngineer):
    """
    Streaming version of research feature set:
    ret_1, ret_4, vol_20, range_pct, range_ma_20, trend_strength, atr_pct
    """

    def __init__(self):
        self._close: Deque[float] = deque(maxlen=256)
        self._high: Deque[float] = deque(maxlen=256)
        self._low: Deque[float] = deque(maxlen=256)
        self._range_pct: Deque[float] = deque(maxlen=256)
        self._tr: Deque[float] = deque(maxlen=256)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._alpha_fast = 2.0 / (20.0 + 1.0)
        self._alpha_slow = 2.0 / (50.0 + 1.0)

    def compute(self, bar: OHLCVBar, state: Dict[str, Any]) -> Dict[str, Any]:
        c = float(bar.close)
        h = float(bar.high)
        l = float(bar.low)

        prev_close = self._close[-1] if self._close else c
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        self._tr.append(tr)

        self._close.append(c)
        self._high.append(h)
        self._low.append(l)

        range_pct = (h - l) / c if c else 0.0
        self._range_pct.append(range_pct)

        if self._ema_fast is None:
            self._ema_fast = c
            self._ema_slow = c
        else:
            self._ema_fast = self._alpha_fast * c + (1.0 - self._alpha_fast) * self._ema_fast
            self._ema_slow = self._alpha_slow * c + (1.0 - self._alpha_slow) * self._ema_slow

        if len(self._close) < 21:
            return {}

        arr_close = np.asarray(self._close, dtype=np.float64)
        ret_1 = (arr_close[-1] / arr_close[-2]) - 1.0 if arr_close[-2] else 0.0
        ret_4 = (arr_close[-1] / arr_close[-5]) - 1.0 if arr_close[-5] else 0.0

        recent_returns = np.diff(arr_close[-21:]) / np.where(arr_close[-21:-1] == 0, 1.0, arr_close[-21:-1])
        vol_20 = float(np.std(recent_returns))
        range_ma_20 = float(np.mean(list(self._range_pct)[-20:])) if len(self._range_pct) >= 20 else 0.0
        trend_strength = float((self._ema_fast - self._ema_slow) / c) if c else 0.0
        atr_pct = (float(np.mean(list(self._tr)[-14:])) / c) if len(self._tr) >= 14 and c else 0.0

        return {
            "ret_1": ret_1,
            "ret_4": ret_4,
            "vol_20": vol_20,
            "range_pct": range_pct,
            "range_ma_20": range_ma_20,
            "trend_strength": trend_strength,
            "atr_pct": atr_pct,
        }


class MultiTimeframeRegimeFeatureEngineer(FeatureEngineer):
    """Compute runtime features for M15/H1/H4-prefixed trained models."""

    def __init__(self):
        self._close: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._high: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._low: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))

    def compute(self, bar: OHLCVBar, state: Dict[str, Any]) -> Dict[str, Any]:
        tf_name = str(bar.timeframe.name).lower()
        if tf_name not in {"m15", "h1", "h4"}:
            return {}
        self._close[tf_name].append(float(bar.close))
        self._high[tf_name].append(float(bar.high))
        self._low[tf_name].append(float(bar.low))

        out = {}
        for key in ("m15", "h1", "h4"):
            c = np.asarray(self._close[key], dtype=np.float64)
            h = np.asarray(self._high[key], dtype=np.float64)
            l = np.asarray(self._low[key], dtype=np.float64)
            if len(c) < 30:
                continue
            prev_c = np.roll(c, 1)
            prev_c[0] = c[0]
            tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
            atr = np.mean(tr[-14:])
            ret1 = (c[-1] / c[-2]) - 1.0 if c[-2] else 0.0
            ret4 = (c[-1] / c[-5]) - 1.0 if len(c) >= 5 and c[-5] else 0.0
            ema20 = self._ema(c, 20)
            ema50 = self._ema(c, 50)
            trend = (ema20 - ema50) / c[-1] if c[-1] else 0.0
            sma20 = np.mean(c[-20:])
            std20 = np.std(c[-20:])
            bbw = (2.0 * 2.0 * std20 / sma20) if sma20 else 0.0
            out[f"{key}_ret1"] = float(ret1)
            out[f"{key}_ret4"] = float(ret4)
            out[f"{key}_atr_pct"] = float(atr / c[-1]) if c[-1] else 0.0
            out[f"{key}_trend"] = float(trend)
            out[f"{key}_bbw"] = float(bbw)
        return out

    @staticmethod
    def _ema(a: np.ndarray, n: int) -> float:
        alpha = 2.0 / (n + 1.0)
        ema = a[0]
        for x in a[1:]:
            ema = alpha * x + (1.0 - alpha) * ema
        return float(ema)


class KMeansRegimePredictor(RegimePredictor):
    """Predict regime by nearest centroid from exported research model."""

    def __init__(self, model: RegimeModel):
        self.model = model
        self.regime_counts: Dict[str, int] = {}
        self.last_probabilities: Dict[str, float] = {}

    def predict(self, bar: OHLCVBar, features: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        if not features:
            self.last_probabilities = {}
            return None
        vec = []
        for col in self.model.feature_columns:
            if col not in features:
                self.last_probabilities = {}
                return None
            value = float(features[col])
            mu = self.model.train_mean.get(col, 0.0)
            sd = self.model.train_std.get(col, 1.0) or 1.0
            vec.append((value - mu) / sd)
        x = np.asarray(vec, dtype=np.float64)
        d = np.sum((self.model.centers - x[None, :]) ** 2, axis=1)
        # Softmax over negative distances as pseudo-probabilities.
        scaled = -d
        scaled -= np.max(scaled)
        expv = np.exp(scaled)
        probs = expv / np.sum(expv)
        self.last_probabilities = {str(i): float(p) for i, p in enumerate(probs.tolist())}
        regime = str(int(np.argmin(d)))
        self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1
        return regime
