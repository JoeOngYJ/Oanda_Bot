from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.data.warehouse import DataWarehouse
from oanda_bot.backtesting.labels.forward_return_labeler import make_labels
from oanda_bot.features.feature_builder import FeatureBuilder
from oanda_bot.ml.models.two_stage import DirectionTCNModel, OpportunityTCNModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = object  # type: ignore[assignment]
    TensorDataset = object  # type: ignore[assignment]


BAR_MINUTES = 15
DEFAULT_START = pd.Timestamp("2024-01-01")
DEFAULT_END = pd.Timestamp("2026-03-01")
EPS = 1e-9


@dataclass
class WalkforwardWindow:
    step_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class SampleBundle:
    timestamps: np.ndarray
    seq: np.ndarray
    ctx: np.ndarray
    y_opportunity: np.ndarray
    y_direction: np.ndarray
    gross_ret: np.ndarray
    net_ret: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    cost_est: np.ndarray
    feature_columns: List[str]


@dataclass
class TrainSummary:
    best_val_loss: float
    epochs_ran: int


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _to_utc_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out = out.set_index("datetime")
        else:
            out.index = pd.to_datetime(out.index, utc=True)
    out.index = pd.to_datetime(out.index, utc=True).tz_convert(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = _to_utc_naive_index(df)

    if {"open", "high", "low", "close"}.issubset(x.columns):
        out = x.copy()
    elif {"mid_o", "mid_h", "mid_l", "mid_c"}.issubset(x.columns):
        out = x.copy()
        out["open"] = pd.to_numeric(out["mid_o"], errors="coerce")
        out["high"] = pd.to_numeric(out["mid_h"], errors="coerce")
        out["low"] = pd.to_numeric(out["mid_l"], errors="coerce")
        out["close"] = pd.to_numeric(out["mid_c"], errors="coerce")
    elif {"bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"}.issubset(x.columns):
        out = x.copy()
        out["open"] = (pd.to_numeric(out["bid_o"], errors="coerce") + pd.to_numeric(out["ask_o"], errors="coerce")) / 2.0
        out["high"] = (pd.to_numeric(out["bid_h"], errors="coerce") + pd.to_numeric(out["ask_h"], errors="coerce")) / 2.0
        out["low"] = (pd.to_numeric(out["bid_l"], errors="coerce") + pd.to_numeric(out["ask_l"], errors="coerce")) / 2.0
        out["close"] = (pd.to_numeric(out["bid_c"], errors="coerce") + pd.to_numeric(out["ask_c"], errors="coerce")) / 2.0
    else:
        raise ValueError("Cannot infer OHLC columns. Expected open/high/low/close or mid_* or bid/ask_*")

    if "volume" not in out.columns:
        out["volume"] = 0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    if "spread_c" not in out.columns and {"ask_c", "bid_c"}.issubset(out.columns):
        out["spread_c"] = pd.to_numeric(out["ask_c"], errors="coerce") - pd.to_numeric(out["bid_c"], errors="coerce")

    keep_cols = [c for c in ["open", "high", "low", "close", "volume", "spread_c"] if c in out.columns]
    return out[keep_cols].dropna(subset=["open", "high", "low", "close"]).sort_index()


def _load_tf_file(base: Path, tf: str) -> Optional[pd.DataFrame]:
    p_parq = base / f"{tf}.parquet"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    p_csv = base / f"{tf}.csv"
    if p_csv.exists():
        return pd.read_csv(p_csv)
    return None


def load_or_ensure_triplet(
    instrument: str,
    data_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir) / instrument
    m15_raw = _load_tf_file(base, "M15")
    h1_raw = _load_tf_file(base, "H1")
    h4_raw = _load_tf_file(base, "H4")

    if m15_raw is None or h1_raw is None or h4_raw is None:
        wh = DataWarehouse(Path(data_dir))
        m15_raw = m15_raw if m15_raw is not None else wh.load(instrument, Timeframe.M15)
        h1_raw = h1_raw if h1_raw is not None else wh.load(instrument, Timeframe.H1)
        h4_raw = h4_raw if h4_raw is not None else wh.load(instrument, Timeframe.H4)

    if m15_raw is None or h1_raw is None or h4_raw is None:
        dm = DataManager({"data_dir": str(data_dir), "oanda": {}})
        got = dm.ensure_data(
            instrument=instrument,
            base_timeframe=Timeframe.M15,
            start_date=start.to_pydatetime(),
            end_date=end.to_pydatetime(),
            timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4],
            force_download=True,
            price="BA",
            store_bid_ask=True,
        )
        m15_raw = m15_raw if m15_raw is not None else got.get(Timeframe.M15)
        h1_raw = h1_raw if h1_raw is not None else got.get(Timeframe.H1)
        h4_raw = h4_raw if h4_raw is not None else got.get(Timeframe.H4)

    if m15_raw is None or h1_raw is None or h4_raw is None:
        raise RuntimeError("Missing M15/H1/H4 datasets and automatic ensure_data failed.")

    m15 = _coerce_ohlcv(m15_raw)
    h1 = _coerce_ohlcv(h1_raw)
    h4 = _coerce_ohlcv(h4_raw)

    # Include warmup margin for indicators/sequence construction.
    warmup_start = start - pd.Timedelta(days=40)
    m15 = m15.loc[(m15.index >= warmup_start) & (m15.index < end)]
    h1 = h1.loc[(h1.index >= warmup_start) & (h1.index < end)]
    h4 = h4.loc[(h4.index >= warmup_start) & (h4.index < end)]

    return m15, h1, h4


def add_fallback_costs(df: pd.DataFrame, default_spread: float = 0.20) -> pd.DataFrame:
    out = df.copy()
    h = pd.to_numeric(out["high"], errors="coerce")
    l = pd.to_numeric(out["low"], errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")

    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    if "spread_c" in out.columns:
        out["spread_est"] = pd.to_numeric(out["spread_c"], errors="coerce").fillna(default_spread)
    else:
        out["spread_est"] = float(default_spread)

    out["slippage_est"] = 0.05 * out["atr"].fillna(out["atr"].median())
    out["commission"] = 0.0
    out["cost_est"] = out["spread_est"] + out["slippage_est"] + out["commission"]
    return out


def _expand_seq_features(base: pd.DataFrame) -> pd.DataFrame:
    x = base.copy()
    for col in [
        "ret_1",
        "ret_4",
        "rsi_14",
        "adx_14",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "bb_width_pct",
        "atr_pct",
        "spread_feat",
        "vol_pct",
    ]:
        if col not in x.columns:
            continue
        for lag in (1, 2, 3):
            x[f"{col}_lag{lag}"] = x[col].shift(lag)
    return x


def _wilder_ema_torch(x: "torch.Tensor", period: int) -> "torch.Tensor":
    """Wilder-style EMA on 1D tensor; returns NaN for warmup region."""
    n = int(x.shape[0])
    out = torch.full_like(x, float("nan"))
    if n == 0 or period <= 0 or n < period:
        return out
    alpha = 1.0 / float(period)
    out[period - 1] = torch.nanmean(x[:period])
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def _rolling_mean_std_torch(x: "torch.Tensor", window: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
    n = int(x.shape[0])
    mean = torch.full_like(x, float("nan"))
    std = torch.full_like(x, float("nan"))
    if n < window or window <= 0:
        return mean, std
    w = x.unfold(0, window, 1)  # [n-window+1, window]
    m = w.mean(dim=1)
    v = w.var(dim=1, correction=0)
    mean[window - 1:] = m
    std[window - 1:] = torch.sqrt(torch.clamp(v, min=0.0))
    return mean, std


def _rolling_percentile_torch(x: "torch.Tensor", window: int, min_periods: int) -> "torch.Tensor":
    n = int(x.shape[0])
    out = torch.full_like(x, float("nan"))
    if n == 0:
        return out
    min_p = max(1, int(min_periods))
    win = max(1, int(window))

    # Prefix region where effective window < `window`.
    prefix_end = min(n, win - 1)
    for i in range(min_p - 1, prefix_end):
        arr = x[: i + 1]
        v = arr[-1]
        if not torch.isfinite(v):
            continue
        valid = torch.isfinite(arr)
        denom = int(valid.sum().item())
        if denom <= 0:
            continue
        out[i] = ((arr <= v) & valid).float().sum() / float(denom)

    # Full-window region.
    if n >= win:
        w = x.unfold(0, win, 1)  # [n-win+1, win]
        v = w[:, -1].unsqueeze(1)
        valid_w = torch.isfinite(w)
        valid_v = torch.isfinite(v)
        denom = valid_w.sum(dim=1).clamp_min(1).float()
        num = ((w <= v) & valid_w & valid_v).float().sum(dim=1)
        pct = num / denom
        pct = torch.where(valid_v.squeeze(1), pct, torch.full_like(pct, float("nan")))
        out[win - 1:] = pct

    return out


def _pack_sequences_gpu(
    seq_df: pd.DataFrame,
    *,
    seq_len: int,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise ImportError("PyTorch is required for GPU sequence packing.")
    arr = torch.as_tensor(seq_df.to_numpy(dtype=np.float32), device=device)
    n, _f = arr.shape
    if n < seq_len:
        return np.asarray([], dtype=np.int64), np.empty((0, seq_len, seq_df.shape[1]), dtype=np.float32)

    win = arr.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()  # [N-seq_len+1, seq_len, F]
    valid_row = torch.isfinite(arr).all(dim=1)
    valid_win = valid_row.unfold(0, seq_len, 1).all(dim=1)
    valid_win = valid_win & torch.isfinite(win).all(dim=(1, 2))

    end_idx = torch.arange(seq_len - 1, n, device=device)[valid_win]
    packed = win[valid_win]

    return (
        end_idx.detach().cpu().numpy().astype(np.int64),
        packed.detach().cpu().numpy().astype(np.float32),
    )


def _build_m15_feature_frame_gpu(
    m15: pd.DataFrame,
    *,
    atr_period: int = 14,
    rsi_period: int = 14,
    adx_period: int = 14,
    bb_period: int = 20,
    percentile_window: int = 252,
    device: Optional["torch.device"] = None,
) -> pd.DataFrame:
    if torch is None:
        raise ImportError("PyTorch is required for GPU preprocessing backend.")

    dev = device or torch.device("cuda")
    idx = m15.index

    open_t = torch.as_tensor(m15["open"].to_numpy(dtype=np.float32), device=dev)
    high_t = torch.as_tensor(m15["high"].to_numpy(dtype=np.float32), device=dev)
    low_t = torch.as_tensor(m15["low"].to_numpy(dtype=np.float32), device=dev)
    close_t = torch.as_tensor(m15["close"].to_numpy(dtype=np.float32), device=dev)
    vol_t = torch.as_tensor(m15["volume"].to_numpy(dtype=np.float32), device=dev)
    if "spread_c" in m15.columns:
        spread_t = torch.as_tensor(pd.to_numeric(m15["spread_c"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), device=dev)
    else:
        spread_t = torch.zeros_like(close_t)

    n = int(close_t.shape[0])
    ret_1 = torch.full_like(close_t, float("nan"))
    ret_4 = torch.full_like(close_t, float("nan"))
    if n > 1:
        ret_1[1:] = (close_t[1:] / (close_t[:-1] + EPS)) - 1.0
    if n > 4:
        ret_4[4:] = (close_t[4:] / (close_t[:-4] + EPS)) - 1.0

    prev_close = torch.cat([torch.full((1,), float("nan"), device=dev), close_t[:-1]], dim=0)
    tr1 = (high_t - low_t).abs()
    tr2 = (high_t - prev_close).abs()
    tr3 = (low_t - prev_close).abs()
    tr = torch.maximum(torch.maximum(tr1, tr2), tr3)
    atr = _wilder_ema_torch(tr, atr_period)

    delta = torch.cat([torch.full((1,), float("nan"), device=dev), close_t[1:] - close_t[:-1]], dim=0)
    gain = torch.where(torch.isnan(delta), delta, torch.clamp(delta, min=0.0))
    loss = torch.where(torch.isnan(delta), delta, torch.clamp(-delta, min=0.0))
    avg_gain = _wilder_ema_torch(gain, rsi_period)
    avg_loss = _wilder_ema_torch(loss, rsi_period)
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    up = torch.cat([torch.full((1,), float("nan"), device=dev), high_t[1:] - high_t[:-1]], dim=0)
    down = torch.cat([torch.full((1,), float("nan"), device=dev), low_t[:-1] - low_t[1:]], dim=0)
    plus_dm = torch.where((up > down) & (up > 0.0), up, torch.zeros_like(up))
    minus_dm = torch.where((down > up) & (down > 0.0), down, torch.zeros_like(down))
    plus_di = 100.0 * (_wilder_ema_torch(plus_dm, adx_period) / (atr + EPS))
    minus_di = 100.0 * (_wilder_ema_torch(minus_dm, adx_period) / (atr + EPS))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = _wilder_ema_torch(dx, adx_period)

    candle_range = torch.clamp(high_t - low_t, min=0.0)
    body = (close_t - open_t).abs()
    upper_wick = torch.clamp(high_t - torch.maximum(open_t, close_t), min=0.0)
    lower_wick = torch.clamp(torch.minimum(open_t, close_t) - low_t, min=0.0)

    minute_of_day = torch.as_tensor((idx.hour * 60 + idx.minute).astype(np.float32), device=dev)
    ang = 2.0 * np.pi * minute_of_day / 1440.0
    session_sin = torch.sin(ang)
    session_cos = torch.cos(ang)

    bb_mid, bb_std = _rolling_mean_std_torch(close_t, bb_period)
    bb_width = (4.0 * bb_std) / (bb_mid.abs() + EPS)
    bb_width_pct = _rolling_percentile_torch(bb_width, percentile_window, 30)
    atr_pct = _rolling_percentile_torch(atr, percentile_window, 30)
    vol_pct = _rolling_percentile_torch(vol_t, percentile_window, 30)

    out = pd.DataFrame(
        {
            "ret_1": ret_1.detach().cpu().numpy(),
            "ret_4": ret_4.detach().cpu().numpy(),
            "atr_14": atr.detach().cpu().numpy(),
            "rsi_14": (rsi / 100.0).detach().cpu().numpy(),
            "adx_14": (adx / 100.0).detach().cpu().numpy(),
            "body_pct": (body / (candle_range + EPS)).detach().cpu().numpy(),
            "upper_wick_pct": (upper_wick / (candle_range + EPS)).detach().cpu().numpy(),
            "lower_wick_pct": (lower_wick / (candle_range + EPS)).detach().cpu().numpy(),
            "close_pos_in_range": ((close_t - low_t) / (candle_range + EPS)).detach().cpu().numpy(),
            "session_sin": session_sin.detach().cpu().numpy(),
            "session_cos": session_cos.detach().cpu().numpy(),
            "bb_width_pct": bb_width_pct.detach().cpu().numpy(),
            "atr_pct": atr_pct.detach().cpu().numpy(),
            "spread_feat": (spread_t / (close_t.abs() + EPS)).detach().cpu().numpy(),
            "vol_pct": vol_pct.detach().cpu().numpy(),
        },
        index=idx,
    )
    return out


def build_samples(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    *,
    seq_len: int = 128,
    horizon_bars: int = 8,
    no_trade_band: float = 0.30,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
    preprocess_backend: str = "cpu",
    preprocess_device: Optional["torch.device"] = None,
) -> SampleBundle:
    fb = FeatureBuilder(seq_len=seq_len)
    m15n = fb._normalize_ohlcv_index(m15)
    h1n = fb._normalize_ohlcv_index(h1)
    h4n = fb._normalize_ohlcv_index(h4)

    m15c = add_fallback_costs(m15n)
    labeled = make_labels(m15c, horizon_bars=horizon_bars, no_trade_band=no_trade_band, use_costs=True)
    labeled["y_direction"] = np.where(
        labeled["y_opportunity"] == 1.0,
        np.where(labeled["net_ret"] > 0.0, 1, 0),
        -1,
    )

    backend = str(preprocess_backend).strip().lower()
    if backend == "gpu":
        if torch is None or (preprocess_device is None and not torch.cuda.is_available()):
            warnings.warn("GPU preprocessing requested but CUDA is unavailable; falling back to CPU backend.")
            backend = "cpu"
        else:
            dev = preprocess_device or torch.device("cuda")
            seq_base = _build_m15_feature_frame_gpu(
                m15c,
                atr_period=fb.atr_period,
                rsi_period=fb.rsi_period,
                adx_period=fb.adx_period,
                bb_period=fb.bb_period,
                percentile_window=fb.percentile_window,
                device=dev,
            )
            seq_df = _expand_seq_features(seq_base).replace([np.inf, -np.inf], np.nan)
    if backend == "cpu":
        seq_df = _expand_seq_features(fb._build_m15_feature_frame(m15c)).replace([np.inf, -np.inf], np.nan)
    feat_cols = list(seq_df.columns)

    times: List[pd.Timestamp] = []
    seq_list: List[np.ndarray] = []
    ctx_list: List[np.ndarray] = []
    y_opp_list: List[float] = []
    y_dir_list: List[int] = []
    gross_list: List[float] = []
    net_list: List[float] = []
    close_list: List[float] = []
    atr_list: List[float] = []
    cost_list: List[float] = []

    if backend == "gpu":
        dev = preprocess_device or torch.device("cuda")
        end_idx, seq_packed = _pack_sequences_gpu(seq_df[feat_cols], seq_len=seq_len, device=dev)
        idx_values = seq_df.index.to_numpy()
        y_opp_s = labeled["y_opportunity"]
        y_dir_s = labeled["y_direction"]
        for j, i_end in enumerate(end_idx.tolist()):
            ts = pd.Timestamp(idx_values[i_end])
            if ts < start or ts >= end:
                continue
            y_opp = y_opp_s.get(ts, np.nan)
            y_dir = y_dir_s.get(ts, np.nan)
            if pd.isna(y_opp) or pd.isna(y_dir):
                continue
            try:
                ctx = fb._build_context_vector(m15c, h1n, h4n, ts).astype(np.float32)
            except Exception:
                continue
            times.append(ts)
            seq_list.append(seq_packed[j])
            ctx_list.append(ctx)
            y_opp_list.append(float(y_opp))
            y_dir_list.append(int(y_dir))
            gross_list.append(float(labeled.at[ts, "gross_ret"]) if pd.notna(labeled.at[ts, "gross_ret"]) else np.nan)
            net_list.append(float(labeled.at[ts, "net_ret"]) if pd.notna(labeled.at[ts, "net_ret"]) else np.nan)
            close_list.append(float(m15c.at[ts, "close"]))
            atr_list.append(float(m15c.at[ts, "atr"]) if pd.notna(m15c.at[ts, "atr"]) else np.nan)
            cost_list.append(float(m15c.at[ts, "cost_est"]) if pd.notna(m15c.at[ts, "cost_est"]) else np.nan)
    else:
        for ts in labeled.index:
            if ts < start or ts >= end:
                continue
            if pd.isna(labeled.at[ts, "y_opportunity"]) or pd.isna(labeled.at[ts, "y_direction"]):
                continue

            w = seq_df.loc[:ts].tail(seq_len)
            if len(w) != seq_len:
                continue
            if w[feat_cols].isna().any(axis=None):
                continue

            try:
                ctx = fb._build_context_vector(m15c, h1n, h4n, ts).astype(np.float32)
            except Exception:
                continue

            times.append(ts)
            seq_list.append(w[feat_cols].to_numpy(dtype=np.float32))
            ctx_list.append(ctx)
            y_opp_list.append(float(labeled.at[ts, "y_opportunity"]))
            y_dir_list.append(int(labeled.at[ts, "y_direction"]))
            gross_list.append(float(labeled.at[ts, "gross_ret"]) if pd.notna(labeled.at[ts, "gross_ret"]) else np.nan)
            net_list.append(float(labeled.at[ts, "net_ret"]) if pd.notna(labeled.at[ts, "net_ret"]) else np.nan)
            close_list.append(float(m15c.at[ts, "close"]))
            atr_list.append(float(m15c.at[ts, "atr"]) if pd.notna(m15c.at[ts, "atr"]) else np.nan)
            cost_list.append(float(m15c.at[ts, "cost_est"]) if pd.notna(m15c.at[ts, "cost_est"]) else np.nan)

    if not times:
        raise RuntimeError("No samples created. Check data columns/range and feature warmup.")

    return SampleBundle(
        timestamps=np.asarray([t.isoformat() for t in times]),
        seq=np.asarray(seq_list, dtype=np.float32),
        ctx=np.asarray(ctx_list, dtype=np.float32),
        y_opportunity=np.asarray(y_opp_list, dtype=np.float32),
        y_direction=np.asarray(y_dir_list, dtype=np.int64),
        gross_ret=np.asarray(gross_list, dtype=np.float32),
        net_ret=np.asarray(net_list, dtype=np.float32),
        close=np.asarray(close_list, dtype=np.float32),
        atr=np.asarray(atr_list, dtype=np.float32),
        cost_est=np.asarray(cost_list, dtype=np.float32),
        feature_columns=feat_cols,
    )


def save_sample_cache(
    bundle: SampleBundle,
    path: Path,
    *,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seq_len: int,
    horizon_bars: int,
    no_trade_band: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=bundle.timestamps,
        seq=bundle.seq.astype(np.float32),
        ctx=bundle.ctx.astype(np.float32),
        y_opportunity=bundle.y_opportunity.astype(np.float32),
        y_direction=bundle.y_direction.astype(np.int64),
        gross_ret=bundle.gross_ret.astype(np.float32),
        net_ret=bundle.net_ret.astype(np.float32),
        close=bundle.close.astype(np.float32),
        atr=bundle.atr.astype(np.float32),
        cost_est=bundle.cost_est.astype(np.float32),
        feature_columns=np.asarray(bundle.feature_columns, dtype=object),
        instrument=np.asarray([instrument], dtype=object),
        start=np.asarray([str(start)]),
        end=np.asarray([str(end)]),
        seq_len=np.asarray([int(seq_len)]),
        horizon_bars=np.asarray([int(horizon_bars)]),
        no_trade_band=np.asarray([float(no_trade_band)]),
    )


def load_sample_cache(path: Path) -> SampleBundle:
    z = np.load(path, allow_pickle=True)
    n = int(z["seq"].shape[0])

    def _opt(name: str) -> np.ndarray:
        if name in z.files:
            return z[name].astype(np.float32)
        return np.full(n, np.nan, dtype=np.float32)

    return SampleBundle(
        timestamps=z["timestamps"],
        seq=z["seq"].astype(np.float32),
        ctx=z["ctx"].astype(np.float32),
        y_opportunity=z["y_opportunity"].astype(np.float32),
        y_direction=z["y_direction"].astype(np.int64),
        gross_ret=_opt("gross_ret"),
        net_ret=_opt("net_ret"),
        close=_opt("close"),
        atr=_opt("atr"),
        cost_est=_opt("cost_est"),
        feature_columns=[str(x) for x in z["feature_columns"].tolist()],
    )


def calibrate_gate_from_validation(
    p_trade_val: np.ndarray,
    *,
    target_trade_rate: float = 0.20,
) -> float:
    x = np.asarray(p_trade_val, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.60
    t = float(np.clip(target_trade_rate, 0.01, 0.99))
    q = float(np.clip(1.0 - t, 0.0, 1.0))
    return float(np.quantile(x, q))


def generate_windows(
    *,
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
    train_months: int = 18,
    val_months: int = 2,
    test_months: int = 2,
    step_months: int = 1,
) -> List[WalkforwardWindow]:
    windows: List[WalkforwardWindow] = []
    cur = pd.Timestamp(start)
    i = 0
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(months=train_months)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)
        test_start = val_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        step_id = f"step_{i:03d}_{train_start.strftime('%Y%m%d')}"
        windows.append(
            WalkforwardWindow(
                step_id=step_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        i += 1
        cur = cur + pd.DateOffset(months=step_months)
    return windows


def split_masks_for_window(timestamps_iso: np.ndarray, window: WalkforwardWindow, horizon_bars: int = 8) -> Dict[str, np.ndarray]:
    ts = pd.to_datetime(pd.Series(timestamps_iso), utc=True).dt.tz_convert(None)
    horizon_td = pd.Timedelta(minutes=BAR_MINUTES * horizon_bars)

    train = (ts >= window.train_start) & (ts < (window.val_start - horizon_td))
    val = (ts >= window.val_start) & (ts < (window.test_start - horizon_td))
    test = (ts >= window.test_start) & (ts < (window.test_end - horizon_td))

    return {
        "train": train.to_numpy(),
        "val": val.to_numpy(),
        "test": test.to_numpy(),
    }


def _make_opp_loader(seq: np.ndarray, ctx: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(seq, dtype=torch.float32),
        torch.as_tensor(ctx, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _dir_to_class(y_dir: np.ndarray) -> np.ndarray:
    out = np.full_like(y_dir, -1, dtype=np.int64)
    out[y_dir == 1] = 0  # long class
    out[y_dir == 0] = 1  # short class
    return out


def _make_dir_loader(seq: np.ndarray, ctx: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(seq, dtype=torch.float32),
        torch.as_tensor(ctx, dtype=torch.float32),
        torch.as_tensor(_dir_to_class(y), dtype=torch.int64),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_opportunity_model(
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    ctx_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[OpportunityTCNModel, TrainSummary]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for train_opportunity_model")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpportunityTCNModel(seq_features=seq_train.shape[2], ctx_dim=ctx_train.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_loader = _make_opp_loader(seq_train, ctx_train, y_train, batch_size=batch_size, shuffle=True)
    va_loader = _make_opp_loader(seq_val, ctx_val, y_val, batch_size=batch_size, shuffle=False)

    best = float("inf")
    best_state = None
    wait = 0
    epochs_ran = 0

    for _epoch in range(1, epochs + 1):
        epochs_ran = _epoch
        model.train()
        for seq_b, ctx_b, y_b in tr_loader:
            seq_b = seq_b.to(dev)
            ctx_b = ctx_b.to(dev)
            y_b = y_b.to(dev)
            opt.zero_grad()
            p = model(seq_b, ctx_b)
            loss = nn.functional.binary_cross_entropy(p, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for seq_b, ctx_b, y_b in va_loader:
                seq_b = seq_b.to(dev)
                ctx_b = ctx_b.to(dev)
                y_b = y_b.to(dev)
                p = model(seq_b, ctx_b)
                l = nn.functional.binary_cross_entropy(p, y_b)
                val_losses.append(float(l.detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainSummary(best_val_loss=float(best), epochs_ran=int(epochs_ran))


def train_direction_model(
    seq_train: np.ndarray,
    ctx_train: np.ndarray,
    y_train: np.ndarray,
    seq_val: np.ndarray,
    ctx_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[DirectionTCNModel, TrainSummary]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for train_direction_model")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectionTCNModel(seq_features=seq_train.shape[2], ctx_dim=ctx_train.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_loader = _make_dir_loader(seq_train, ctx_train, y_train, batch_size=batch_size, shuffle=True)
    va_loader = _make_dir_loader(seq_val, ctx_val, y_val, batch_size=batch_size, shuffle=False)

    best = float("inf")
    best_state = None
    wait = 0
    epochs_ran = 0

    for _epoch in range(1, epochs + 1):
        epochs_ran = _epoch
        model.train()
        for seq_b, ctx_b, y_b in tr_loader:
            seq_b = seq_b.to(dev)
            ctx_b = ctx_b.to(dev)
            y_b = y_b.to(dev)
            mask = y_b != -1
            if not torch.any(mask):
                continue
            opt.zero_grad()
            logits = model.forward_logits(seq_b, ctx_b)
            loss = nn.functional.cross_entropy(logits[mask], y_b[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for seq_b, ctx_b, y_b in va_loader:
                seq_b = seq_b.to(dev)
                ctx_b = ctx_b.to(dev)
                y_b = y_b.to(dev)
                mask = y_b != -1
                if not torch.any(mask):
                    continue
                logits = model.forward_logits(seq_b, ctx_b)
                l = nn.functional.cross_entropy(logits[mask], y_b[mask])
                val_losses.append(float(l.detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainSummary(best_val_loss=float(best), epochs_ran=int(epochs_ran))


def predict_probabilities(
    opp_model: OpportunityTCNModel,
    dir_model: DirectionTCNModel,
    seq: np.ndarray,
    ctx: np.ndarray,
    *,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:
        raise ImportError("PyTorch is required for predict_probabilities")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opp_model = opp_model.to(dev)
    dir_model = dir_model.to(dev)
    opp_model.eval()
    dir_model.eval()

    ds = TensorDataset(torch.as_tensor(seq, dtype=torch.float32), torch.as_tensor(ctx, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    p_trade_list: List[np.ndarray] = []
    p_long_list: List[np.ndarray] = []
    p_short_list: List[np.ndarray] = []

    with torch.no_grad():
        for seq_b, ctx_b in loader:
            seq_b = seq_b.to(dev)
            ctx_b = ctx_b.to(dev)
            p_trade = opp_model(seq_b, ctx_b).detach().cpu().numpy()
            p_dir = dir_model(seq_b, ctx_b).detach().cpu().numpy()
            p_trade_list.append(p_trade)
            p_long_list.append(p_dir[:, 0])
            p_short_list.append(p_dir[:, 1])

    return (
        np.concatenate(p_trade_list, axis=0),
        np.concatenate(p_long_list, axis=0),
        np.concatenate(p_short_list, axis=0),
    )


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        return float("nan")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    sum_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def evaluate_test_slice(
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    y_opportunity: np.ndarray,
    y_direction: np.ndarray,
    *,
    gate: float = 0.60,
) -> Dict[str, float]:
    active = p_trade > float(gate)
    dir_pred = np.where(p_long > p_short, 1, 0)

    trade_rate = float(np.mean(active)) if active.size else 0.0
    direction_balance = float(np.mean(dir_pred[active])) if np.any(active) else float("nan")

    valid = (y_direction != -1) & active
    direction_acc = float(np.mean(dir_pred[valid] == y_direction[valid])) if np.any(valid) else float("nan")

    auc = _binary_auc(y_opportunity.astype(int), p_trade)

    return {
        "test_mean_p_trade": float(np.mean(p_trade)) if p_trade.size else 0.0,
        "test_trade_rate": trade_rate,
        "test_direction_balance": direction_balance,
        "test_dir_acc": direction_acc,
        "test_opp_auc": auc,
    }


def make_signals_frame(
    timestamps_iso: np.ndarray,
    p_trade: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
) -> pd.DataFrame:
    idx = pd.to_datetime(pd.Series(timestamps_iso), utc=True).dt.tz_convert(None)
    return pd.DataFrame(
        {
            "datetime": idx,
            "p_trade": p_trade.astype(float),
            "p_long": p_long.astype(float),
            "p_short": p_short.astype(float),
        }
    )


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_step_models(step_dir: Path, seq_features: int, ctx_dim: int, device: Optional[torch.device] = None) -> Tuple[OpportunityTCNModel, DirectionTCNModel]:
    if torch is None:
        raise ImportError("PyTorch is required for load_step_models")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opp = OpportunityTCNModel(seq_features=seq_features, ctx_dim=ctx_dim)
    direction = DirectionTCNModel(seq_features=seq_features, ctx_dim=ctx_dim)

    opp.load_state_dict(torch.load(step_dir / "opportunity.pt", map_location=dev))
    direction.load_state_dict(torch.load(step_dir / "direction.pt", map_location=dev))
    opp.to(dev).eval()
    direction.to(dev).eval()
    return opp, direction
