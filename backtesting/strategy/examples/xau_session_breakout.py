"""Institutional-style XAUUSD breakout strategy (M15 execution, HTF bias)."""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.base import StrategyBase
from backtesting.strategy.signal import Signal, SignalDirection


@dataclass
class _TrackedTrade:
    direction: SignalDirection
    entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    opened_day: dt.date


class XAUSessionBreakout(StrategyBase):
    """
    Breakout-only strategy for XAUUSD with regime + session filters.

    Design principles:
    1) Only volatility expansion environments.
    2) No mean-reversion entries.
    3) Asymmetric risk/reward (>= 1:2).
    4) Strict daily risk constraints.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.exec_tf = Timeframe.from_oanda_granularity(str(config.get("exec_tf", "M15")))
        self.bias_tf = Timeframe.from_oanda_granularity(str(config.get("bias_tf", "H1")))
        self.alt_bias_tf = Timeframe.from_oanda_granularity(str(config.get("alt_bias_tf", "H4")))

        self.adx_period = int(config.get("adx_period", 14))
        self.adx_trade_min = float(config.get("adx_trade_min", 25.0))
        self.adx_hard_floor = float(config.get("adx_hard_floor", 20.0))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_rising_lookback = int(config.get("atr_rising_lookback", 5))
        self.bb_period = int(config.get("bb_period", 20))
        self.bb_std = float(config.get("bb_std", 2.0))
        self.atr_compression_quantile = float(config.get("atr_compression_quantile", 0.30))

        self.session_start_hour = int(config.get("session_start_hour", 0))  # GMT
        self.session_end_hour = int(config.get("session_end_hour", 6))  # GMT
        self.trade_start_hour = int(config.get("trade_start_hour", 6))  # London open
        self.trade_end_hour = int(config.get("trade_end_hour", 21))

        self.body_lookback = int(config.get("body_lookback", 5))
        self.volume_lookback = int(config.get("volume_lookback", 5))
        self.require_volume_confirmation = bool(config.get("require_volume_confirmation", True))

        self.risk_per_trade = float(config.get("risk_per_trade", 0.01))
        self.account_equity = float(config.get("account_equity", 10000))
        self.max_trades_per_day = int(config.get("max_trades_per_day", 2))
        self.max_consecutive_losses = int(config.get("max_consecutive_losses", 2))
        self.min_rr = float(config.get("min_rr", 2.0))
        self.stop_atr_mult = float(config.get("stop_atr_mult", 1.2))
        self.min_qty = int(config.get("min_qty", 1))
        self.max_qty = int(config.get("max_qty", 100000))

        self.estimated_spread_pips = float(config.get("estimated_spread_pips", 20.0))
        self.max_spread_pips = float(config.get("max_spread_pips", 35.0))
        self.news_blackout_minutes = int(config.get("news_blackout_minutes", 45))
        self.news_events = self._parse_news_events(config.get("news_events_utc", []))

        # M15 history for indicators (closed bars only).
        hist_len = 600
        self._m15_open: Deque[Decimal] = deque(maxlen=hist_len)
        self._m15_high: Deque[Decimal] = deque(maxlen=hist_len)
        self._m15_low: Deque[Decimal] = deque(maxlen=hist_len)
        self._m15_close: Deque[Decimal] = deque(maxlen=hist_len)
        self._m15_volume: Deque[int] = deque(maxlen=hist_len)
        self._m15_ts: Deque[dt.datetime] = deque(maxlen=hist_len)

        self._bias_close: Dict[Timeframe, Deque[Decimal]] = {
            self.bias_tf: deque(maxlen=300),
            self.alt_bias_tf: deque(maxlen=300),
        }

        self._current_day: Optional[dt.date] = None
        self._asian_high: Optional[Decimal] = None
        self._asian_low: Optional[Decimal] = None
        self._trades_today = 0
        self._consecutive_losses = 0
        self._tracked_trade: Optional[_TrackedTrade] = None

    def get_required_warmup_bars(self) -> Dict[Timeframe, int]:
        warm_exec = max(120, self.bb_period + self.adx_period + 20)
        return {
            self.exec_tf: warm_exec,
            self.bias_tf: 120,
            self.alt_bias_tf: 120,
        }

    def on_market_bar(self, bar: OHLCVBar):
        if bar.timeframe == self.exec_tf:
            self._m15_open.append(bar.open)
            self._m15_high.append(bar.high)
            self._m15_low.append(bar.low)
            self._m15_close.append(bar.close)
            self._m15_volume.append(int(bar.volume))
            self._m15_ts.append(bar.timestamp)
        elif bar.timeframe in self._bias_close:
            self._bias_close[bar.timeframe].append(bar.close)
        return None

    def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        if bar.timeframe != self.exec_tf:
            return None
        if str(bar.instrument) != "XAU_USD":
            return None
        if len(self._m15_close) < self.get_required_warmup_bars()[self.exec_tf]:
            return None

        self._roll_day(bar.timestamp.date())
        self._update_asian_range(bar)
        self._update_tracked_trade(bar)

        if self._trades_today >= self.max_trades_per_day:
            return None
        if self._consecutive_losses >= self.max_consecutive_losses:
            return None
        if self._tracked_trade is not None:
            return None
        if not self._is_trade_session(bar.timestamp):
            return None
        if self._is_news_blackout(bar.timestamp):
            return None
        if self.estimated_spread_pips > self.max_spread_pips:
            return None
        if self._asian_high is None or self._asian_low is None:
            return None

        adx, atr, atr_series, bb_width = self._compute_regime_indicators()
        if adx is None or atr is None or bb_width is None:
            return None
        if adx < self.adx_hard_floor or adx < self.adx_trade_min:
            return None
        if not self._is_atr_expanding(atr_series):
            return None
        if not self._is_bb_expanding():
            return None
        if self._is_atr_compressed(atr_series):
            return None

        bias = self._higher_tf_bias()
        if bias == 0:
            return None

        breakout_long, breakout_short = self._breakout_conditions(bar)
        if breakout_long and bias > 0:
            return self._build_signal(bar, SignalDirection.LONG, atr, adx, bb_width)
        if breakout_short and bias < 0:
            return self._build_signal(bar, SignalDirection.SHORT, atr, adx, bb_width)
        return None

    def _roll_day(self, day: dt.date) -> None:
        if self._current_day is None:
            self._current_day = day
            return
        if day != self._current_day:
            self._current_day = day
            self._asian_high = None
            self._asian_low = None
            self._trades_today = 0
            self._tracked_trade = None

    def _update_asian_range(self, bar: OHLCVBar) -> None:
        hour = bar.timestamp.hour
        if self.session_start_hour <= hour < self.session_end_hour:
            self._asian_high = bar.high if self._asian_high is None else max(self._asian_high, bar.high)
            self._asian_low = bar.low if self._asian_low is None else min(self._asian_low, bar.low)

    def _is_trade_session(self, ts: dt.datetime) -> bool:
        return self.trade_start_hour <= ts.hour < self.trade_end_hour

    def _is_news_blackout(self, ts: dt.datetime) -> bool:
        if not self.news_events:
            return False
        delta = dt.timedelta(minutes=self.news_blackout_minutes)
        return any(abs(ts - ev) <= delta for ev in self.news_events)

    def _compute_regime_indicators(self) -> Tuple[Optional[float], Optional[float], List[float], Optional[float]]:
        high = np.asarray([float(x) for x in self._m15_high], dtype=np.float64)
        low = np.asarray([float(x) for x in self._m15_low], dtype=np.float64)
        close = np.asarray([float(x) for x in self._m15_close], dtype=np.float64)
        if len(close) < max(self.bb_period + 5, self.adx_period * 2 + 5):
            return None, None, [], None

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
        atr = self._rolling_mean(tr, self.atr_period)

        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0.0
        down_move[0] = 0.0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = 100.0 * self._rolling_sum(plus_dm, self.adx_period) / np.where(atr == 0, np.nan, atr)
        minus_di = 100.0 * self._rolling_sum(minus_dm, self.adx_period) / np.where(atr == 0, np.nan, atr)
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
        adx_series = self._rolling_mean(np.nan_to_num(dx, nan=0.0), self.adx_period)

        sma = self._rolling_mean(close, self.bb_period)
        std = self._rolling_std(close, self.bb_period)
        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)
        bb_width = (upper - lower) / np.where(sma == 0, np.nan, sma)

        atr_series = [x for x in atr[-50:] if np.isfinite(x)]
        adx_val = float(adx_series[-1]) if np.isfinite(adx_series[-1]) else None
        atr_val = float(atr[-1]) if np.isfinite(atr[-1]) else None
        bb_val = float(bb_width[-1]) if np.isfinite(bb_width[-1]) else None
        return adx_val, atr_val, [float(x) for x in atr_series], bb_val

    def _is_atr_expanding(self, atr_series: List[float]) -> bool:
        n = self.atr_rising_lookback
        if len(atr_series) < n + 1:
            return False
        tail = atr_series[-(n + 1) :]
        rising_steps = sum(1 for i in range(1, len(tail)) if tail[i] > tail[i - 1])
        return tail[-1] > tail[0] and rising_steps >= max(2, n - 2)

    def _is_bb_expanding(self) -> bool:
        close = np.asarray([float(x) for x in self._m15_close], dtype=np.float64)
        if len(close) < self.bb_period + 10:
            return False
        sma = self._rolling_mean(close, self.bb_period)
        std = self._rolling_std(close, self.bb_period)
        width = (sma + self.bb_std * std - (sma - self.bb_std * std)) / np.where(sma == 0, np.nan, sma)
        tail = [x for x in width[-6:] if np.isfinite(x)]
        if len(tail) < 6:
            return False
        return tail[-1] > tail[0]

    def _is_atr_compressed(self, atr_series: List[float]) -> bool:
        if len(atr_series) < 30:
            return True
        q = float(np.quantile(np.asarray(atr_series, dtype=np.float64), self.atr_compression_quantile))
        return atr_series[-1] <= q

    def _higher_tf_bias(self) -> int:
        votes = []
        for tf in (self.bias_tf, self.alt_bias_tf):
            closes = self._bias_close.get(tf, deque())
            if len(closes) < 60:
                continue
            arr = np.asarray([float(x) for x in closes], dtype=np.float64)
            ema_fast = self._ema(arr, 20)
            ema_slow = self._ema(arr, 50)
            if ema_fast > ema_slow:
                votes.append(1)
            elif ema_fast < ema_slow:
                votes.append(-1)
        if not votes:
            return 0
        s = sum(votes)
        return 1 if s > 0 else -1 if s < 0 else 0

    def _breakout_conditions(self, bar: OHLCVBar) -> Tuple[bool, bool]:
        bodies = [abs(float(c - o)) for o, c in zip(self._m15_open, self._m15_close)]
        if len(bodies) < self.body_lookback + 1:
            return False, False
        body_now = bodies[-1]
        body_avg = float(np.mean(bodies[-(self.body_lookback + 1) : -1]))
        if body_now <= body_avg:
            return False, False

        if self.require_volume_confirmation:
            vols = list(self._m15_volume)
            if len(vols) < self.volume_lookback + 1:
                return False, False
            vol_now = vols[-1]
            vol_avg = float(np.mean(vols[-(self.volume_lookback + 1) : -1]))
            if vol_now <= vol_avg:
                return False, False

        long_break = bar.close > self._asian_high
        short_break = bar.close < self._asian_low
        return long_break, short_break

    def _build_signal(
        self,
        bar: OHLCVBar,
        direction: SignalDirection,
        atr: float,
        adx: float,
        bb_width: float,
    ) -> Signal:
        entry = bar.close
        atr_stop_dist = Decimal(str(self.stop_atr_mult * atr))
        if direction == SignalDirection.LONG:
            candle_stop = bar.low
            stop = min(candle_stop, entry - atr_stop_dist)
            risk_per_unit = entry - stop
            take_profit = entry + (risk_per_unit * Decimal(str(self.min_rr)))
        else:
            candle_stop = bar.high
            stop = max(candle_stop, entry + atr_stop_dist)
            risk_per_unit = stop - entry
            take_profit = entry - (risk_per_unit * Decimal(str(self.min_rr)))

        qty = self._risk_position_size(risk_per_unit)
        self._trades_today += 1
        self._tracked_trade = _TrackedTrade(
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=take_profit,
            opened_day=bar.timestamp.date(),
        )

        return Signal(
            timestamp=bar.timestamp,
            instrument=bar.instrument,
            direction=direction,
            strategy_name=self.name,
            entry_price=entry,
            stop_loss=stop,
            take_profit=take_profit,
            timeframe=self.exec_tf,
            confidence=0.75,
            quantity=qty,
            metadata={
                "regime": "vol_expansion_breakout",
                "adx": adx,
                "bb_width": bb_width,
                "asian_high": float(self._asian_high),
                "asian_low": float(self._asian_low),
                "risk_per_trade": self.risk_per_trade,
                "rr_target": self.min_rr,
                "trail_after_r": 1.5,
            },
        )

    def _risk_position_size(self, risk_per_unit: Decimal) -> int:
        risk_amt = Decimal(str(self.account_equity * self.risk_per_trade))
        if risk_per_unit <= 0:
            return self.min_qty
        qty = int(risk_amt / risk_per_unit)
        qty = max(self.min_qty, min(qty, self.max_qty))
        return qty

    def _update_tracked_trade(self, bar: OHLCVBar) -> None:
        t = self._tracked_trade
        if t is None:
            return
        # Conservative exit ordering: if both touched, stop-loss first.
        if t.direction == SignalDirection.LONG:
            sl_hit = bar.low <= t.stop_loss
            tp_hit = bar.high >= t.take_profit
        else:
            sl_hit = bar.high >= t.stop_loss
            tp_hit = bar.low <= t.take_profit

        if not sl_hit and not tp_hit:
            return

        if sl_hit:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._tracked_trade = None

    @staticmethod
    def _parse_news_events(values: List[str]) -> List[dt.datetime]:
        out: List[dt.datetime] = []
        for v in values or []:
            try:
                ts = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                out.append(ts.astimezone(dt.timezone.utc))
            except Exception:
                continue
        return out

    @staticmethod
    def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < window:
            return out
        csum = np.cumsum(np.insert(arr, 0, 0.0))
        vals = csum[window:] - csum[:-window]
        out[window - 1 :] = vals
        return out

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        s = XAUSessionBreakout._rolling_sum(arr, window)
        return s / float(window)

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < window:
            return out
        for i in range(window - 1, len(arr)):
            out[i] = np.std(arr[i - window + 1 : i + 1])
        return out

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> float:
        alpha = 2.0 / (period + 1.0)
        ema = arr[0]
        for x in arr[1:]:
            ema = alpha * x + (1.0 - alpha) * ema
        return float(ema)
