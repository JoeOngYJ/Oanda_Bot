from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.data.warehouse import DataWarehouse
from oanda_bot.backtesting.labels.forward_return_labeler import make_labels
from oanda_bot.features.feature_builder import FeatureBuilder


DEFAULT_DATA_DIR = Path("data/backtesting")
INSTRUMENT = "XAU_USD"
HORIZON_BARS = 8


def _parse_utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC").tz_convert(None)


TRAIN_START = _parse_utc("2024-01-01")
VAL_START = _parse_utc("2025-10-01")
TEST_START = _parse_utc("2026-01-01")
TEST_END = _parse_utc("2026-03-01")


def _read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        return df.sort_index()
    return None


def load_or_ensure_ohlcv(
    instrument: str = INSTRUMENT,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir) / instrument
    m15 = _read_parquet_if_exists(base / "M15.parquet")
    h1 = _read_parquet_if_exists(base / "H1.parquet")
    h4 = _read_parquet_if_exists(base / "H4.parquet")

    # Try warehouse paths if direct paths are incomplete.
    if m15 is None or h1 is None or h4 is None:
        wh = DataWarehouse(Path(data_dir))
        m15 = m15 if m15 is not None else wh.load(instrument, Timeframe.M15)
        h1 = h1 if h1 is not None else wh.load(instrument, Timeframe.H1)
        h4 = h4 if h4 is not None else wh.load(instrument, Timeframe.H4)
        for df in (m15, h1, h4):
            if df is not None:
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

    # Final fallback: try ensure_data (requires OANDA token/config).
    if m15 is None or h1 is None or h4 is None:
        try:
            dm = DataManager({"data_dir": str(data_dir), "oanda": {}})
            start = TRAIN_START.to_pydatetime()
            end = TEST_END.to_pydatetime()
            data = dm.ensure_data(
                instrument=instrument,
                base_timeframe=Timeframe.M15,
                start_date=start,
                end_date=end,
                timeframes=[Timeframe.M15, Timeframe.H1, Timeframe.H4],
                price="M",
                store_bid_ask=False,
            )
            m15 = m15 if m15 is not None else data.get(Timeframe.M15)
            h1 = h1 if h1 is not None else data.get(Timeframe.H1)
            h4 = h4 if h4 is not None else data.get(Timeframe.H4)
        except Exception as exc:
            raise RuntimeError(
                "Missing required XAU_USD data for M15/H1/H4 and automatic download failed. "
                "Provide parquet files at data/backtesting/XAU_USD/{M15,H1,H4}.parquet "
                "or configure OANDA_API_TOKEN for DataManager download."
            ) from exc

    if m15 is None or h1 is None or h4 is None:
        raise RuntimeError(
            "Missing required XAU_USD data for M15/H1/H4. "
            "Expected parquet under data/backtesting/XAU_USD or available OANDA download config."
        )
    return m15.sort_index(), h1.sort_index(), h4.sort_index()


def add_ohlc_fallback_costs(df: pd.DataFrame, default_spread: float = 0.20) -> pd.DataFrame:
    out = df.copy()
    h = pd.to_numeric(out["high"], errors="coerce").astype(float)
    l = pd.to_numeric(out["low"], errors="coerce").astype(float)
    c = pd.to_numeric(out["close"], errors="coerce").astype(float)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    out["atr"] = atr
    if "spread_c" in out.columns:
        out["spread_est"] = pd.to_numeric(out["spread_c"], errors="coerce").fillna(default_spread).astype(float)
    else:
        out["spread_est"] = float(default_spread)
    out["slippage_est"] = 0.05 * out["atr"].fillna(out["atr"].median())
    out["commission"] = 0.0
    out["cost_est"] = out["spread_est"] + out["slippage_est"] + out["commission"]
    return out


def _expand_seq_features(base: pd.DataFrame) -> pd.DataFrame:
    x = base.copy()
    lag_cols = [
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
    ]
    for c in lag_cols:
        if c not in x.columns:
            continue
        for k in (1, 2, 3):
            x[f"{c}_lag{k}"] = x[c].shift(k)
    return x


def build_feature_label_table(
    m15: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    *,
    instrument: str = INSTRUMENT,
    horizon_bars: int = HORIZON_BARS,
    no_trade_band: float = 0.30,
    seq_len: int = 128,
) -> Dict[str, np.ndarray]:
    fb = FeatureBuilder(seq_len=seq_len)
    m15n = fb._normalize_ohlcv_index(m15)
    h1n = fb._normalize_ohlcv_index(h1)
    h4n = fb._normalize_ohlcv_index(h4)

    m15c = add_ohlc_fallback_costs(m15n)
    labeled = make_labels(m15c, horizon_bars=horizon_bars, no_trade_band=no_trade_band, use_costs=True)
    # For stage-2 direction training use -1 for invalid, else binary long(1)/short(0).
    labeled["y_direction"] = np.where(
        labeled["y_opportunity"] == 1.0,
        np.where(labeled["net_ret"] > 0.0, 1, 0),
        -1,
    )

    seq_base = _expand_seq_features(fb._build_m15_feature_frame(m15c))
    seq_base = seq_base.replace([np.inf, -np.inf], np.nan)

    times = []
    seq_list = []
    ctx_list = []
    y_opp = []
    y_dir = []
    close_l = []
    atr_l = []
    cost_l = []
    cols = list(seq_base.columns)

    for ts in labeled.index:
        if ts < TRAIN_START or ts >= TEST_END:
            continue
        if pd.isna(labeled.at[ts, "y_opportunity"]) or pd.isna(labeled.at[ts, "y_direction"]):
            continue
        w = seq_base.loc[:ts].tail(seq_len)
        if len(w) != seq_len or w[cols].isna().any(axis=None):
            continue
        try:
            ctx = fb._build_context_vector(m15c, h1n, h4n, ts).astype(np.float32)
        except Exception:
            continue
        times.append(ts)
        seq_list.append(w[cols].to_numpy(dtype=np.float32))
        ctx_list.append(ctx)
        y_opp.append(float(labeled.at[ts, "y_opportunity"]))
        y_dir.append(int(labeled.at[ts, "y_direction"]))
        close_l.append(float(m15c.at[ts, "close"]))
        atr_l.append(float(m15c.at[ts, "atr"]) if pd.notna(m15c.at[ts, "atr"]) else float("nan"))
        cost_l.append(float(m15c.at[ts, "cost_est"]) if pd.notna(m15c.at[ts, "cost_est"]) else float("nan"))

    if not times:
        raise RuntimeError("No valid sequence samples were created.")

    return {
        "timestamps": np.asarray([t.isoformat() for t in times]),
        "seq": np.asarray(seq_list, dtype=np.float32),
        "ctx": np.asarray(ctx_list, dtype=np.float32),
        "y_opportunity": np.asarray(y_opp, dtype=np.float32),
        "y_direction": np.asarray(y_dir, dtype=np.int64),
        "close": np.asarray(close_l, dtype=np.float32),
        "atr": np.asarray(atr_l, dtype=np.float32),
        "cost_est": np.asarray(cost_l, dtype=np.float32),
        "feature_columns": np.asarray(cols),
        "instrument": np.asarray([instrument]),
    }


def split_masks_with_embargo(timestamps_iso: np.ndarray, horizon_bars: int = HORIZON_BARS) -> Dict[str, np.ndarray]:
    t = pd.to_datetime(pd.Series(timestamps_iso), utc=True).dt.tz_convert(None)
    train = (t >= TRAIN_START) & (t < VAL_START)
    val = (t >= VAL_START) & (t < TEST_START)
    test = (t >= TEST_START) & (t < TEST_END)

    # Purge by dropping last H bars from train and val splits to prevent overlap leakage.
    def drop_last_h(mask: pd.Series, h: int) -> pd.Series:
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) <= h:
            return pd.Series(False, index=mask.index)
        keep = np.zeros(len(mask), dtype=bool)
        keep[idx[:-h]] = True
        return pd.Series(keep, index=mask.index)

    train = drop_last_h(train, horizon_bars)
    val = drop_last_h(val, horizon_bars)

    return {
        "train": train.to_numpy(),
        "val": val.to_numpy(),
        "test": test.to_numpy(),
    }
