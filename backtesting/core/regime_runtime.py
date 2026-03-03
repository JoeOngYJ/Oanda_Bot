"""Runtime regime feature engineering and centroid-based regime prediction."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    _CUPY_AVAILABLE = False

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

    def __init__(self, use_gpu: bool = False):
        self._close: Deque[float] = deque(maxlen=256)
        self._high: Deque[float] = deque(maxlen=256)
        self._low: Deque[float] = deque(maxlen=256)
        self._range_pct: Deque[float] = deque(maxlen=256)
        self._tr: Deque[float] = deque(maxlen=256)
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._alpha_fast = 2.0 / (20.0 + 1.0)
        self._alpha_slow = 2.0 / (50.0 + 1.0)
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)

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

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)
        self._xp = cp if self.use_gpu else np
        self._close: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._high: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._low: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._open: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._vol: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self._ema20: Dict[str, Optional[float]] = defaultdict(lambda: None)
        self._ema50: Dict[str, Optional[float]] = defaultdict(lambda: None)
        self._alpha20 = 2.0 / 21.0
        self._alpha50 = 2.0 / 51.0

    def on_market_bar(self, bar: OHLCVBar, state: Dict[str, Any]) -> None:
        tf_name = str(bar.timeframe.name).lower()
        if tf_name not in {"m15", "h1", "h4", "d1"}:
            return
        self._close[tf_name].append(float(bar.close))
        self._high[tf_name].append(float(bar.high))
        self._low[tf_name].append(float(bar.low))
        self._open[tf_name].append(float(bar.open))
        self._vol[tf_name].append(float(bar.volume))
        c = float(bar.close)
        if self._ema20[tf_name] is None:
            self._ema20[tf_name] = c
        else:
            self._ema20[tf_name] = self._alpha20 * c + (1.0 - self._alpha20) * float(self._ema20[tf_name])
        if self._ema50[tf_name] is None:
            self._ema50[tf_name] = c
        else:
            self._ema50[tf_name] = self._alpha50 * c + (1.0 - self._alpha50) * float(self._ema50[tf_name])

    def compute(self, bar: OHLCVBar, state: Dict[str, Any]) -> Dict[str, Any]:
        tf_name = str(bar.timeframe.name).lower()
        if tf_name != "m15":
            return {}

        xp = self._xp
        out = {}
        for key in ("m15", "h1", "h4", "d1"):
            c = xp.asarray(self._close[key], dtype=xp.float64)
            h = xp.asarray(self._high[key], dtype=xp.float64)
            l = xp.asarray(self._low[key], dtype=xp.float64)
            o = xp.asarray(self._open[key], dtype=xp.float64)
            v = xp.asarray(self._vol[key], dtype=xp.float64)
            if len(c) < 5:
                continue
            prev_c = xp.roll(c, 1)
            prev_c[0] = c[0]
            tr = xp.maximum(xp.maximum(h - l, xp.abs(h - prev_c)), xp.abs(l - prev_c))
            atr_lookback = min(14, len(tr))
            atr = xp.mean(tr[-atr_lookback:])
            c_last = float(c[-1])
            c_prev1 = float(c[-2])
            c_prev4 = float(c[-5]) if len(c) >= 5 else 0.0
            o_last = float(o[-1])
            h_last = float(h[-1])
            l_last = float(l[-1])
            v_last = float(v[-1])
            ret1 = (c_last / c_prev1) - 1.0 if c_prev1 else 0.0
            ret4 = (c_last / c_prev4) - 1.0 if c_prev4 else 0.0
            ema20 = self._ema20[key]
            ema50 = self._ema50[key]
            trend = ((ema20 - ema50) / c_last) if (ema20 is not None and ema50 is not None and c_last) else 0.0
            band_lookback = min(20, len(c))
            sma20 = float(xp.mean(c[-band_lookback:]))
            std20 = float(xp.std(c[-band_lookback:]))
            bbw = (2.0 * 2.0 * std20 / sma20) if sma20 else 0.0
            body_pct = ((c_last - o_last) / o_last) if o_last else 0.0
            range_pct = ((h_last - l_last) / c_last) if c_last else 0.0
            vol_lookback = min(20, len(v))
            vol_recent = v[-vol_lookback:]
            vol_std = float(xp.std(vol_recent))
            vol_mean = float(xp.mean(vol_recent))
            vol_z = 0.0 if vol_std == 0.0 else float((v_last - vol_mean) / vol_std)
            out[f"{key}_ret1"] = float(ret1)
            out[f"{key}_ret4"] = float(ret4)
            out[f"{key}_atr_pct"] = float(atr / c_last) if c_last else 0.0
            out[f"{key}_trend"] = float(trend)
            out[f"{key}_bbw"] = float(bbw)
            out[f"{key}_body_pct"] = float(body_pct)
            out[f"{key}_range_pct"] = float(range_pct)
            out[f"{key}_vol_z"] = float(vol_z)
            if key == "m15":
                hour = float(bar.timestamp.hour)
                wday = float(bar.timestamp.weekday())
                out[f"{key}_sess_asia"] = float(1.0 if 0 <= hour < 7 else 0.0)
                out[f"{key}_sess_europe"] = float(1.0 if 7 <= hour < 13 else 0.0)
                out[f"{key}_sess_us"] = float(1.0 if 13 <= hour < 22 else 0.0)
                out[f"{key}_sess_eu_us_overlap"] = float(1.0 if 13 <= hour < 17 else 0.0)
                out[f"{key}_hour_sin"] = float(np.sin((2.0 * np.pi * hour) / 24.0))
                out[f"{key}_hour_cos"] = float(np.cos((2.0 * np.pi * hour) / 24.0))
                out[f"{key}_wday_sin"] = float(np.sin((2.0 * np.pi * wday) / 7.0))
                out[f"{key}_wday_cos"] = float(np.cos((2.0 * np.pi * wday) / 7.0))
        return out


class KMeansRegimePredictor(RegimePredictor):
    """Predict regime by nearest centroid from exported research model."""

    def __init__(self, model: RegimeModel, use_gpu: bool = False):
        self.model = model
        self.regime_counts: Dict[str, int] = {}
        self.last_probabilities: Dict[str, float] = {}
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)
        self._feature_columns = list(model.feature_columns)
        self._mu = np.asarray([model.train_mean.get(c, 0.0) for c in self._feature_columns], dtype=np.float64)
        self._sd = np.asarray([model.train_std.get(c, 1.0) or 1.0 for c in self._feature_columns], dtype=np.float64)
        if self.use_gpu:
            self._centers_gpu = cp.asarray(model.centers, dtype=cp.float64)
        else:
            self._centers_gpu = None

    def predict(self, bar: OHLCVBar, features: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        if not features:
            self.last_probabilities = {}
            return None
        vec = []
        for col in self._feature_columns:
            if col not in features:
                self.last_probabilities = {}
                return None
            vec.append(float(features[col]))
        x = (np.asarray(vec, dtype=np.float64) - self._mu) / self._sd
        if self.use_gpu and self._centers_gpu is not None:
            x_gpu = cp.asarray(x, dtype=cp.float64)
            d = cp.sum((self._centers_gpu - x_gpu[None, :]) ** 2, axis=1)
            d = cp.asnumpy(d)
        else:
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
