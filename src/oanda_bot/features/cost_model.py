from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass
class SpreadTable:
    """Hour/vol-bucket spread lookup with optional per-instrument overrides."""

    table: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.table:
            self.table = {
                "*": {
                    "overlap": {"low": 0.00010, "mid": 0.00012, "high": 0.00016},
                    "london": {"low": 0.00012, "mid": 0.00015, "high": 0.00020},
                    "newyork": {"low": 0.00013, "mid": 0.00016, "high": 0.00022},
                    "tokyo": {"low": 0.00016, "mid": 0.00020, "high": 0.00028},
                    "offhours": {"low": 0.00018, "mid": 0.00024, "high": 0.00034},
                }
            }

    @staticmethod
    def hour_bucket(ts: pd.Timestamp) -> str:
        h = int(ts.hour)
        if 12 <= h < 16:
            return "overlap"
        if 7 <= h < 16:
            return "london"
        if 12 <= h < 21:
            return "newyork"
        if 0 <= h < 9:
            return "tokyo"
        return "offhours"

    def lookup(self, instrument: str, hour_bucket: str, vol_bucket: str) -> float:
        inst_key = instrument if instrument in self.table else "*"
        by_hour = self.table.get(inst_key, self.table.get("*", {}))
        by_vol = by_hour.get(hour_bucket) or by_hour.get("offhours") or {}
        if vol_bucket in by_vol:
            return float(by_vol[vol_bucket])
        if "mid" in by_vol:
            return float(by_vol["mid"])
        return float(next(iter(by_vol.values()))) if by_vol else 0.0


@dataclass
class CostModel:
    spread_table: SpreadTable = field(default_factory=SpreadTable)
    min_slip: float = 0.0
    alpha: float = 0.10
    commission: float = 0.0
    atr_period: int = 14
    vol_window: int = 252

    def add_cost_columns(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        req = {"high", "low", "close"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns for cost model: {sorted(miss)}")

        out = df.copy()
        idx_utc = pd.to_datetime(out.index, utc=True)

        atr = self._atr(out["high"].astype(float), out["low"].astype(float), out["close"].astype(float), self.atr_period)
        out["atr"] = atr

        if "spread_c" in out.columns:
            spread_est = pd.to_numeric(out["spread_c"], errors="coerce")
        else:
            atr_pct = atr / (out["close"].astype(float).abs() + EPS)
            atr_rank = self._rolling_percentile_rank(atr_pct, self.vol_window).fillna(0.5)
            vol_bucket = pd.Series(
                np.where(atr_rank < 1.0 / 3.0, "low", np.where(atr_rank < 2.0 / 3.0, "mid", "high")),
                index=out.index,
            )
            spread_vals = []
            for ts, vb in zip(idx_utc, vol_bucket):
                spread_vals.append(self.spread_table.lookup(instrument, self.spread_table.hour_bucket(ts), str(vb)))
            spread_est = pd.Series(spread_vals, index=out.index, dtype=float)

        slip_raw = self.alpha * atr
        slippage_est = np.maximum(float(self.min_slip), np.nan_to_num(slip_raw.to_numpy(dtype=float), nan=float(self.min_slip)))
        slippage_est = pd.Series(slippage_est, index=out.index, dtype=float)

        out["spread_est"] = pd.to_numeric(spread_est, errors="coerce").fillna(0.0)
        out["slippage_est"] = slippage_est
        out["cost_est"] = out["spread_est"] + out["slippage_est"] + float(self.commission)
        return out

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
        def pct_last(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            v = arr[-1]
            return float(np.sum(arr <= v)) / float(len(arr))

        return series.rolling(window=window, min_periods=20).apply(pct_last, raw=True)
