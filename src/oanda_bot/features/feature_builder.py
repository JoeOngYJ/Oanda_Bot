from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


@dataclass
class FeatureBuilder:
    seq_len: int = 128
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    percentile_window: int = 252

    def build(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        *,
        instrument: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        m15 = self._normalize_ohlcv_index(m15_df)
        h1 = self._normalize_ohlcv_index(h1_df)
        h4 = self._normalize_ohlcv_index(h4_df)

        seq_features = self._build_m15_feature_frame(m15)
        seq_features = seq_features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(seq_features) < self.seq_len:
            raise ValueError(f"Not enough valid M15 rows for sequence length={self.seq_len}.")

        seq_frame = seq_features.iloc[-self.seq_len:]
        as_of = seq_frame.index[-1]

        ctx = self._build_context_vector(m15, h1, h4, as_of).astype(np.float32)
        seq = seq_frame.to_numpy(dtype=np.float32)

        atr_last = float(seq_features.loc[as_of, "atr_14"])
        close_last = float(m15.loc[as_of, "close"])
        meta = {
            "datetime_index": seq_frame.index,
            "instrument": instrument,
            "close": close_last,
            "atr": atr_last,
        }
        return seq, ctx, meta

    def _build_m15_feature_frame(self, m15: pd.DataFrame) -> pd.DataFrame:
        close = m15["close"].astype(float)
        high = m15["high"].astype(float)
        low = m15["low"].astype(float)
        open_ = m15["open"].astype(float)
        volume = m15["volume"].astype(float)

        atr_14 = self._atr(high, low, close, self.atr_period)
        rsi_14 = self._rsi(close, self.rsi_period)
        adx_14 = self._adx(high, low, close, self.adx_period)

        candle_range = (high - low).clip(lower=0.0)
        body = (close - open_).abs()
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)

        minute_of_day = (m15.index.hour * 60 + m15.index.minute).astype(float)
        ang = 2.0 * np.pi * minute_of_day / 1440.0

        bb_mid = close.rolling(self.bb_period, min_periods=self.bb_period).mean()
        bb_std = close.rolling(self.bb_period, min_periods=self.bb_period).std(ddof=0)
        bb_width = (4.0 * bb_std) / (bb_mid.abs() + EPS)

        spread = m15["spread_c"].astype(float) if "spread_c" in m15.columns else pd.Series(0.0, index=m15.index)

        out = pd.DataFrame(
            {
                "ret_1": close.pct_change(1),
                "ret_4": close.pct_change(4),
                "atr_14": atr_14,
                "rsi_14": rsi_14 / 100.0,
                "adx_14": adx_14 / 100.0,
                "body_pct": body / (candle_range + EPS),
                "upper_wick_pct": upper_wick / (candle_range + EPS),
                "lower_wick_pct": lower_wick / (candle_range + EPS),
                "close_pos_in_range": (close - low) / (candle_range + EPS),
                "session_sin": np.sin(ang),
                "session_cos": np.cos(ang),
                "bb_width_pct": self._rolling_percentile(bb_width, self.percentile_window, 30),
                "atr_pct": self._rolling_percentile(atr_14, self.percentile_window, 30),
                "spread_feat": spread / (close.abs() + EPS),
                "vol_pct": self._rolling_percentile(volume, self.percentile_window, 30),
            },
            index=m15.index,
        )
        return out

    def _build_context_vector(
        self,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        h4: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> np.ndarray:
        m15_cut = m15.loc[:as_of]
        h1_cut = h1.loc[:as_of]
        h4_cut = h4.loc[:as_of]
        if h1_cut.empty or h4_cut.empty or m15_cut.empty:
            raise ValueError("Insufficient data at as-of timestamp to build context features.")

        atr_m15 = self._atr(m15_cut["high"], m15_cut["low"], m15_cut["close"], self.atr_period)
        atr_last = float(atr_m15.iloc[-1])
        close_last = float(m15_cut["close"].iloc[-1])

        prev_day = (
            m15_cut[["high", "low"]]
            .resample("1D")
            .agg({"high": "max", "low": "min"})
            .shift(1)
            .reindex(m15_cut.index, method="ffill")
        )
        prev_day_high = float(prev_day["high"].iloc[-1]) if pd.notna(prev_day["high"].iloc[-1]) else np.nan
        prev_day_low = float(prev_day["low"].iloc[-1]) if pd.notna(prev_day["low"].iloc[-1]) else np.nan

        h1_adx = self._adx(h1_cut["high"], h1_cut["low"], h1_cut["close"], self.adx_period)
        h4_adx = self._adx(h4_cut["high"], h4_cut["low"], h4_cut["close"], self.adx_period)

        h1_vol_pct = self._rolling_percentile(h1_cut["volume"].astype(float), self.percentile_window, 20)
        h4_vol_pct = self._rolling_percentile(h4_cut["volume"].astype(float), self.percentile_window, 20)

        hour = int(as_of.hour)
        asia = 1.0 if 0 <= hour < 8 else 0.0
        london = 1.0 if 8 <= hour < 13 else 0.0
        ny = 1.0 if 13 <= hour < 22 else 0.0

        ctx = np.array(
            [
                self._slope(h1_cut["close"], lookback=24),
                self._slope(h4_cut["close"], lookback=12),
                float(h1_adx.iloc[-1]) / 100.0,
                float(h4_adx.iloc[-1]) / 100.0,
                float(h1_vol_pct.iloc[-1]),
                float(h4_vol_pct.iloc[-1]),
                (close_last - prev_day_high) / (atr_last + EPS) if np.isfinite(prev_day_high) else np.nan,
                (close_last - prev_day_low) / (atr_last + EPS) if np.isfinite(prev_day_low) else np.nan,
                asia,
                london,
                ny,
            ],
            dtype=np.float64,
        )
        return np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
        req = {"open", "high", "low", "close", "volume"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing required OHLCV columns: {sorted(miss)}")
        x = df.copy()
        x.index = pd.to_datetime(x.index, utc=True).tz_convert(None)
        x = x[~x.index.duplicated(keep="last")].sort_index()
        return x

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr1 = (high - low).abs()
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = self._true_range(high, low, close)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / (avg_loss + EPS)
        return 100.0 - (100.0 / (1.0 + rs))

    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0.0), up, 0.0), index=high.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0.0), down, 0.0), index=high.index)
        atr = self._atr(high, low, close, period)
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
        return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
        def pct_last(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            v = arr[-1]
            return float(np.sum(arr <= v)) / float(len(arr))

        return series.rolling(window=window, min_periods=min_periods).apply(pct_last, raw=True)

    @staticmethod
    def _slope(close: pd.Series, lookback: int) -> float:
        if len(close) <= lookback:
            return np.nan
        current = float(close.iloc[-1])
        prev = float(close.iloc[-1 - lookback])
        if abs(prev) < EPS:
            return 0.0
        return (current / prev - 1.0) / float(lookback)
