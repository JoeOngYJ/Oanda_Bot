#!/usr/bin/env python3
"""Run multi-pair, multi-timeframe strategy research with walk-forward stability."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oanda_bot.backtesting.analysis.metrics import expectancy_per_trade, profit_factor
from oanda_bot.backtesting.core.engine import BacktestEngine
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.data.manager import DataManager
from oanda_bot.backtesting.strategy.examples.atr_breakout import ATRBreakout
from oanda_bot.backtesting.strategy.examples.breakout import Breakout
from oanda_bot.backtesting.strategy.examples.ema_pullback import EMATrendPullback
from oanda_bot.backtesting.strategy.examples.ensemble_vote import EnsembleVoteStrategy
from oanda_bot.backtesting.strategy.examples.intermarket_mtf_confluence import IntermarketMTFConfluence
from oanda_bot.backtesting.strategy.examples.mean_reversion import MeanReversion
from oanda_bot.backtesting.strategy.examples.multi_tf_trend import MultiTimeframeTrendStrategy
from oanda_bot.backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from oanda_bot.backtesting.strategy.examples.volatility_compression_breakout import VolatilityCompressionBreakout


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research strategies across instrument universe with correlation and walk-forward consistency."
    )
    parser.add_argument("--instruments", default="EUR_USD,GBP_USD,USD_JPY,XAU_USD")
    parser.add_argument("--base-tf", default="M15")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--wf-windows", type=int, default=4, help="Walk-forward windows (>=1).")
    parser.add_argument("--min-stability", type=float, default=0.25, help="Minimum stability score for shortlist.")
    parser.add_argument("--min-trades", type=float, default=1.0, help="Minimum mean test trades for shortlist.")
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.75,
        help="Maximum absolute return correlation allowed between shortlisted instruments.",
    )
    parser.add_argument("--initial-capital", type=float, default=10000)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--pricing-model", default="oanda_core", choices=["spread_only", "oanda_core"])
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument("--demo-bars", type=int, default=0)
    parser.add_argument(
        "--intermarket-sweep",
        action="store_true",
        help="Enable extended IntermarketMTFConfluence parameter sweep.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers for instrument-level parallelism (1 = sequential).",
    )
    parser.add_argument(
        "--vol-targeting",
        action="store_true",
        help="Enable volatility-targeted position sizing in execution simulator.",
    )
    parser.add_argument(
        "--target-annual-vol",
        type=float,
        default=0.15,
        help="Target annualized volatility used when --vol-targeting is enabled.",
    )
    parser.add_argument(
        "--vol-lookback-bars",
        type=int,
        default=96,
        help="Lookback bars for realized volatility estimate when --vol-targeting is enabled.",
    )
    parser.add_argument(
        "--max-exposure-pct",
        type=float,
        default=None,
        help="Optional max gross notional exposure as fraction of initial capital (e.g., 1.0 = 100%).",
    )
    parser.add_argument(
        "--max-quantity",
        type=int,
        default=None,
        help="Optional hard upper bound for order quantity after risk sizing.",
    )
    parser.add_argument(
        "--candidate-shortlist",
        default="",
        help="Optional CSV shortlist to filter strategy candidates by name/config.",
    )
    return parser.parse_args()


def _parse_dt(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _make_demo_df(start: dt.datetime, tf: Timeframe, bars: int, seed: int):
    rng = np.random.default_rng(seed=seed)
    step = dt.timedelta(seconds=tf.seconds)
    idx = [start + i * step for i in range(bars)]
    price = 1.1000 + rng.normal(0, 0.01)

    opens, highs, lows, closes, vols = [], [], [], [], []
    for _ in range(bars):
        drift = rng.normal(0.0, 0.0008)
        close = max(0.0001, price + drift)
        span = abs(rng.normal(0.0010, 0.0003))
        high = max(price, close) + span
        low = min(price, close) - span
        opens.append(price)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        vols.append(int(rng.integers(50, 800)))
        price = close

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=pd.to_datetime(idx, utc=True),
    )


def _resample(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    return df.resample(tf.to_pandas_freq()).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def _strategy_candidates(base_tf: Timeframe, intermarket_sweep: bool = False) -> List[Tuple[str, Dict]]:
    candidates: List[Tuple[str, Dict]] = []

    for lookback in (20, 40):
        candidates.append(
            (
                "Breakout",
                {
                    "name": f"Breakout_L{lookback}",
                    "class": Breakout,
                    "timeframes": [base_tf],
                    "lookback": lookback,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.008,
                    "min_breakout_pct": 0.0002,
                    "quantity": 10000,
                },
            )
        )

    for sma_period in (20, 30):
        for dev in (0.0020, 0.0030):
            candidates.append(
                (
                    "MeanReversion",
                    {
                        "name": f"MeanRev_SMA{sma_period}_D{dev}",
                        "class": MeanReversion,
                        "timeframes": [base_tf],
                        "sma_period": sma_period,
                        "deviation_pct": dev,
                        "stop_loss_pct": 0.004,
                        "take_profit_pct": 0.003,
                        "quantity": 10000,
                    },
                )
            )

    for fast, slow in ((20, 100), (50, 200)):
        candidates.append(
            (
                "EMATrendPullback",
                {
                    "name": f"EMAPullback_{fast}_{slow}",
                    "class": EMATrendPullback,
                    "timeframes": [base_tf],
                    "fast_period": fast,
                    "slow_period": slow,
                    "pullback_pct": 0.0008,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.009,
                    "quantity": 10000,
                },
            )
        )

    for lookback in (20, 30):
        candidates.append(
            (
                "ATRBreakout",
                {
                    "name": f"ATRBreakout_L{lookback}",
                    "class": ATRBreakout,
                    "timeframes": [base_tf],
                    "lookback": lookback,
                    "atr_period": 14,
                    "atr_mult": 1.2,
                    "stop_loss_pct": 0.0045,
                    "take_profit_pct": 0.009,
                    "quantity": 10000,
                },
            )
        )

    for std in (1.8, 2.2):
        candidates.append(
            (
                "RSIBollingerReversion",
                {
                    "name": f"RSIBB_STD{std}",
                    "class": RSIBollingerReversion,
                    "timeframes": [base_tf],
                    "window": 20,
                    "std_mult": std,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.003,
                    "quantity": 10000,
                },
            )
        )

    candidates.append(
        (
            "VolatilityCompressionBreakout",
            {
                "name": "VolCompression_0.75",
                "class": VolatilityCompressionBreakout,
                "timeframes": [base_tf],
                "range_lookback": 20,
                "atr_period": 14,
                "compression_window": 40,
                "compression_ratio": 0.75,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.01,
                "quantity": 10000,
            },
        )
    )

    candidates.append(
        (
            "EnsembleVote",
            {
                "name": "Ensemble_TrendBreakout",
                "class": EnsembleVoteStrategy,
                "timeframes": [base_tf],
                "min_votes": 2,
                "cooldown_bars": 8,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.008,
                "quantity": 10000,
                "components": [
                    {
                        "class": EMATrendPullback,
                        "timeframes": [base_tf],
                        "fast_period": 20,
                        "slow_period": 100,
                        "pullback_pct": 0.0008,
                    },
                    {
                        "class": ATRBreakout,
                        "timeframes": [base_tf],
                        "lookback": 20,
                        "atr_period": 14,
                        "atr_mult": 1.2,
                    },
                    {
                        "class": Breakout,
                        "timeframes": [base_tf],
                        "lookback": 40,
                    },
                ],
            },
        )
    )

    if intermarket_sweep:
        for ema_fast, ema_slow in ((20, 100), (50, 200)):
            for align in (0.5, 0.67):
                for rs_min in (0.0001, 0.0002):
                    for cooldown in (6, 10):
                        candidates.append(
                            (
                                "IntermarketMTFConfluence",
                                {
                                    "name": f"InterMTF_F{ema_fast}_S{ema_slow}_A{align}_RS{rs_min}_CD{cooldown}",
                                    "class": IntermarketMTFConfluence,
                                    "timeframes": [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15],
                                    "ema_fast": ema_fast,
                                    "ema_slow": ema_slow,
                                    "min_ref_alignment": align,
                                    "relative_strength_lookback": 12,
                                    "relative_strength_min": rs_min,
                                    "stop_loss_pct": 0.004,
                                    "take_profit_pct": 0.008,
                                    "quantity": 10000,
                                    "cooldown_bars": cooldown,
                                    "uses_market_context": True,
                                },
                            )
                        )
    else:
        candidates.append(
            (
                "IntermarketMTFConfluence",
                {
                    "name": "Intermarket_MTF_Confluence",
                    "class": IntermarketMTFConfluence,
                    "timeframes": [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15],
                    "ema_fast": 50,
                    "ema_slow": 200,
                    "min_ref_alignment": 0.5,
                    "relative_strength_lookback": 12,
                    "relative_strength_min": 0.0002,
                    "stop_loss_pct": 0.004,
                    "take_profit_pct": 0.008,
                    "quantity": 10000,
                    "cooldown_bars": 8,
                    "uses_market_context": True,
                },
            )
        )

    candidates.append(
        (
            "EnsembleVote",
            {
                "name": "Ensemble_MeanRev",
                "class": EnsembleVoteStrategy,
                "timeframes": [base_tf],
                "min_votes": 2,
                "cooldown_bars": 6,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.0035,
                "quantity": 10000,
                "components": [
                    {
                        "class": MeanReversion,
                        "timeframes": [base_tf],
                        "sma_period": 30,
                        "deviation_pct": 0.002,
                    },
                    {
                        "class": RSIBollingerReversion,
                        "timeframes": [base_tf],
                        "window": 20,
                        "std_mult": 2.0,
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                    },
                ],
            },
        )
    )

    candidates.append(
        (
            "MultiTFTrend",
            {
                "name": "MultiTF_Trend_50_200",
                "class": MultiTimeframeTrendStrategy,
                "timeframes": [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15],
                "ema_fast": 50,
                "ema_slow": 200,
            },
        )
    )

    return candidates


def _run_engine(
    instrument: str,
    strategy_cfg: Dict,
    execution_cfg: Dict,
    data_dict: Dict[Timeframe, pd.DataFrame],
    market_data_dict: Dict[str, Dict[Timeframe, pd.DataFrame]] | None = None,
) -> Dict:
    base_tf = min(data_dict.keys(), key=lambda tf: tf.seconds)
    cfg = {
        "data": {
            "instrument": instrument,
            "base_timeframe": base_tf,
            "start_date": data_dict[base_tf].index.min().to_pydatetime(),
            "end_date": data_dict[base_tf].index.max().to_pydatetime(),
        },
        "strategy": strategy_cfg,
        "execution": execution_cfg,
        "data_dict": data_dict,
    }
    if market_data_dict:
        cfg["market_data_dict"] = market_data_dict
    result = BacktestEngine(cfg).run()
    net_pnl = result.final_equity - execution_cfg["initial_capital"]
    return {
        "total_trades": result.total_trades,
        "net_pnl": float(net_pnl),
        "win_rate": float(result.win_rate),
        "expectancy": float(expectancy_per_trade(result.trades)),
        "profit_factor": float(profit_factor(result.trades)),
        "max_drawdown": float(result.max_drawdown),
        "fees": float(result.total_fees_paid),
        "sharpe": float(result.sharpe_ratio),
    }


def _slice_dict(data: Dict[Timeframe, pd.DataFrame], start_idx, end_idx) -> Dict[Timeframe, pd.DataFrame]:
    out = {}
    for tf, df in data.items():
        out[tf] = df.loc[(df.index >= start_idx) & (df.index <= end_idx)]
    return out


def _slice_market_dict(
    market_data: Dict[str, Dict[Timeframe, pd.DataFrame]],
    start_idx,
    end_idx,
) -> Dict[str, Dict[Timeframe, pd.DataFrame]]:
    out: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
    for instrument, tf_map in market_data.items():
        out[instrument] = {}
        for tf, df in tf_map.items():
            out[instrument][tf] = df.loc[(df.index >= start_idx) & (df.index <= end_idx)]
    return out


def _compute_correlation(base_tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    close_df = pd.DataFrame({k: v["close"] for k, v in base_tf_data.items()}).dropna(how="all")
    returns = close_df.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def _walk_forward_windows(n_rows: int, train_ratio: float, windows: int) -> List[Tuple[int, int, int, int]]:
    if windows < 1:
        windows = 1
    base_train = int(n_rows * train_ratio)
    base_train = max(base_train, 200)
    remaining = n_rows - base_train
    if remaining <= 0:
        return []
    test_len = max(50, remaining // windows)

    ranges: List[Tuple[int, int, int, int]] = []
    train_end = base_train - 1
    for _ in range(windows):
        test_start = train_end + 1
        test_end = min(n_rows - 1, test_start + test_len - 1)
        if test_start >= n_rows or test_end <= test_start:
            break
        ranges.append((0, train_end, test_start, test_end))
        train_end = test_end
        if train_end >= n_rows - 2:
            break
    return ranges


def _stability_score(window_net_pnls: List[float], window_pfs: List[float]) -> float:
    if not window_net_pnls:
        return 0.0
    pos_ratio = sum(1 for v in window_net_pnls if v > 0) / len(window_net_pnls)
    mean_net = float(np.mean(window_net_pnls))
    std_net = float(np.std(window_net_pnls))
    consistency = 0.0
    if mean_net != 0:
        consistency = max(0.0, min(1.0, 1.0 - (std_net / (abs(mean_net) + 1e-9))))
    finite_pfs = [pf for pf in window_pfs if np.isfinite(pf)]
    mean_pf = float(np.mean(finite_pfs)) if finite_pfs else 0.0
    pf_component = max(0.0, min(1.0, mean_pf / 2.0))
    return round((0.5 * pos_ratio) + (0.25 * consistency) + (0.25 * pf_component), 6)


def _filter_candidates_by_shortlist(
    candidates: List[Tuple[str, Dict]],
    shortlist_path: str,
) -> List[Tuple[str, Dict]]:
    if not shortlist_path:
        return candidates
    p = Path(shortlist_path)
    if not p.exists():
        print(f"Shortlist file not found: {shortlist_path}. Using full candidate set.", flush=True)
        return candidates

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        print(f"Failed to read shortlist CSV ({exc}). Using full candidate set.", flush=True)
        return candidates

    if "strategy_name" not in df.columns:
        print("Shortlist CSV missing 'strategy_name' column. Using full candidate set.", flush=True)
        return candidates

    if "config_name" in df.columns:
        keys = set(zip(df["strategy_name"].astype(str), df["config_name"].astype(str)))
        filtered = [
            (name, cfg)
            for (name, cfg) in candidates
            if (str(name), str(cfg.get("name", ""))) in keys
        ]
        if not filtered:
            names = set(df["strategy_name"].astype(str))
            filtered = [(name, cfg) for (name, cfg) in candidates if str(name) in names]
    else:
        names = set(df["strategy_name"].astype(str))
        filtered = [(name, cfg) for (name, cfg) in candidates if str(name) in names]

    if not filtered:
        print("No candidate matched shortlist CSV. Using full candidate set.", flush=True)
        return candidates
    print(
        f"Candidate filter applied: {len(candidates)} -> {len(filtered)} using {shortlist_path}",
        flush=True,
    )
    return filtered


def _evaluate_instrument(
    instrument: str,
    universe_data: Dict[str, Dict[Timeframe, pd.DataFrame]],
    base_tf: Timeframe,
    candidates: List[Tuple[str, Dict]],
    execution_cfg: Dict,
    train_ratio: float,
    wf_windows: int,
) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    window_rows: List[Dict] = []

    data_dict = universe_data[instrument]
    base_df = data_dict[base_tf].sort_index()
    wf_ranges = _walk_forward_windows(len(base_df), train_ratio, wf_windows)
    if not wf_ranges:
        return rows, window_rows

    for strategy_name, strategy_cfg in candidates:
        required_set = set(strategy_cfg.get("timeframes", [base_tf]))
        uses_market_context = bool(strategy_cfg.get("uses_market_context", False))

        train_net_list: List[float] = []
        test_net_list: List[float] = []
        test_pf_list: List[float] = []
        test_exp_list: List[float] = []
        test_dd_list: List[float] = []
        test_wr_list: List[float] = []
        test_trades_list: List[int] = []
        test_fees_list: List[float] = []
        test_sharpe_list: List[float] = []
        windows_run = 0

        for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(wf_ranges, start=1):
            train_start_ts = base_df.index[tr_s]
            train_end_ts = base_df.index[tr_e]
            test_start_ts = base_df.index[te_s]
            test_end_ts = base_df.index[te_e]

            train_dict = _slice_dict(data_dict, train_start_ts, train_end_ts)
            test_dict = _slice_dict(data_dict, test_start_ts, test_end_ts)
            train_input = {tf: train_dict[tf] for tf in required_set if tf in train_dict and not train_dict[tf].empty}
            test_input = {tf: test_dict[tf] for tf in required_set if tf in test_dict and not test_dict[tf].empty}
            if len(train_input) != len(required_set) or len(test_input) != len(required_set):
                continue

            strategy_for_run = dict(strategy_cfg)
            train_market_input = None
            test_market_input = None
            if uses_market_context:
                refs = [inst for inst in universe_data.keys() if inst != instrument]
                if not refs:
                    continue
                strategy_for_run["primary_instrument"] = instrument
                strategy_for_run["reference_instruments"] = refs
                train_market_input = _slice_market_dict(universe_data, train_start_ts, train_end_ts)
                test_market_input = _slice_market_dict(universe_data, test_start_ts, test_end_ts)
                ok_train_refs = all(
                    ref in train_market_input
                    and all(tf in train_market_input[ref] and not train_market_input[ref][tf].empty for tf in required_set)
                    for ref in refs
                )
                ok_test_refs = all(
                    ref in test_market_input
                    and all(tf in test_market_input[ref] and not test_market_input[ref][tf].empty for tf in required_set)
                    for ref in refs
                )
                if not ok_train_refs or not ok_test_refs:
                    continue

            train_stats = _run_engine(
                instrument, strategy_for_run, execution_cfg, train_input, market_data_dict=train_market_input
            )
            test_stats = _run_engine(
                instrument, strategy_for_run, execution_cfg, test_input, market_data_dict=test_market_input
            )
            windows_run += 1

            train_net_list.append(train_stats["net_pnl"])
            test_net_list.append(test_stats["net_pnl"])
            test_pf_list.append(test_stats["profit_factor"])
            test_exp_list.append(test_stats["expectancy"])
            test_dd_list.append(test_stats["max_drawdown"])
            test_wr_list.append(test_stats["win_rate"])
            test_trades_list.append(int(test_stats["total_trades"]))
            test_fees_list.append(test_stats["fees"])
            test_sharpe_list.append(test_stats["sharpe"])

            window_rows.append(
                {
                    "instrument": instrument,
                    "strategy_name": strategy_name,
                    "window": w_idx,
                    "train_start": str(train_start_ts),
                    "train_end": str(train_end_ts),
                    "test_start": str(test_start_ts),
                    "test_end": str(test_end_ts),
                    "train_net_pnl": round(train_stats["net_pnl"], 6),
                    "test_net_pnl": round(test_stats["net_pnl"], 6),
                    "test_trades": int(test_stats["total_trades"]),
                    "test_expectancy": round(test_stats["expectancy"], 6),
                    "test_win_rate": round(test_stats["win_rate"], 6),
                    "test_profit_factor": round(test_stats["profit_factor"], 6),
                    "test_max_drawdown": round(test_stats["max_drawdown"], 6),
                }
            )

        if windows_run == 0:
            continue

        stable = _stability_score(test_net_list, test_pf_list)
        finite_pfs = [v for v in test_pf_list if np.isfinite(v)]
        rows.append(
            {
                "instrument": instrument,
                "strategy_name": strategy_name,
                "params": json.dumps(
                    {k: v for k, v in strategy_cfg.items() if k not in {"class", "timeframes", "uses_market_context"}},
                    default=str,
                ),
                "windows_run": windows_run,
                "positive_window_ratio": round(sum(1 for v in test_net_list if v > 0) / windows_run, 6),
                "stability_score": stable,
                "train_net_pnl_mean": round(float(np.mean(train_net_list)), 6),
                "test_net_pnl_mean": round(float(np.mean(test_net_list)), 6),
                "test_net_pnl_std": round(float(np.std(test_net_list)), 6),
                "test_trades_mean": round(float(np.mean(test_trades_list)), 6),
                "test_expectancy_mean": round(float(np.mean(test_exp_list)), 6),
                "test_win_rate_mean": round(float(np.mean(test_wr_list)), 6),
                "test_profit_factor_mean": round(float(np.mean(finite_pfs)) if finite_pfs else 0.0, 6),
                "test_max_drawdown_mean": round(float(np.mean(test_dd_list)), 6),
                "test_fees_mean": round(float(np.mean(test_fees_list)), 6),
                "test_sharpe_mean": round(float(np.mean(test_sharpe_list)), 6),
            }
        )

    return rows, window_rows


# Fork-based worker globals for low-overhead parallelism.
_MP_UNIVERSE_DATA: Dict[str, Dict[Timeframe, pd.DataFrame]] | None = None
_MP_BASE_TF: Timeframe | None = None
_MP_CANDIDATES: List[Tuple[str, Dict]] | None = None
_MP_EXECUTION_CFG: Dict | None = None
_MP_TRAIN_RATIO: float = 0.7
_MP_WF_WINDOWS: int = 4


def _evaluate_instrument_mp(instrument: str) -> Tuple[List[Dict], List[Dict]]:
    assert _MP_UNIVERSE_DATA is not None
    assert _MP_BASE_TF is not None
    assert _MP_CANDIDATES is not None
    assert _MP_EXECUTION_CFG is not None
    return _evaluate_instrument(
        instrument=instrument,
        universe_data=_MP_UNIVERSE_DATA,
        base_tf=_MP_BASE_TF,
        candidates=_MP_CANDIDATES,
        execution_cfg=_MP_EXECUTION_CFG,
        train_ratio=_MP_TRAIN_RATIO,
        wf_windows=_MP_WF_WINDOWS,
    )


def main() -> int:
    args = parse_args()
    base_tf = Timeframe.from_oanda_granularity(args.base_tf)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)

    if not (0.5 <= args.train_ratio <= 0.9):
        raise SystemExit("--train-ratio must be between 0.5 and 0.9")
    if args.wf_windows < 1:
        raise SystemExit("--wf-windows must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    instruments = [x.strip() for x in args.instruments.split(",") if x.strip()]
    if not instruments:
        raise SystemExit("No instruments provided.")

    execution_cfg = {
        "initial_capital": float(args.initial_capital),
        "slippage_pips": float(args.slippage_pips),
        "pricing_model": args.pricing_model,
        "spreads_pips": {
            "EUR_USD": 1.4,
            "GBP_USD": 2.0,
            "USD_JPY": 1.4,
            "USD_CAD": 2.2,
            "AUD_USD": 1.4,
            "XAU_USD": 20.0,
        },
        "core_commission_per_10k_units": 1.0,
        "volatility_targeting_enabled": bool(args.vol_targeting),
        "target_annual_volatility": float(args.target_annual_vol),
        "volatility_lookback_bars": int(args.vol_lookback_bars),
    }
    if args.max_exposure_pct is not None:
        execution_cfg["max_concurrent_exposure_pct"] = float(args.max_exposure_pct)
    if args.max_quantity is not None:
        execution_cfg["max_quantity"] = int(args.max_quantity)

    required_tfs = {base_tf, Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.M15}
    candidates = _strategy_candidates(base_tf, intermarket_sweep=args.intermarket_sweep)
    candidates = _filter_candidates_by_shortlist(candidates, args.candidate_shortlist)

    universe_data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
    base_tf_data: Dict[str, pd.DataFrame] = {}
    dm = DataManager({})

    for idx, instrument in enumerate(instruments):
        if args.demo_bars > 0:
            base_df = _make_demo_df(start, base_tf, args.demo_bars, seed=100 + idx)
            data_dict = {tf: _resample(base_df, tf) for tf in required_tfs}
        else:
            data_dict = dm.ensure_data(
                instrument=instrument,
                base_timeframe=base_tf,
                start_date=start,
                end_date=end,
                timeframes=list(required_tfs),
            )

        base_df = data_dict[base_tf].sort_index()
        if len(base_df) < 300:
            print(f"Skipping {instrument}: not enough bars ({len(base_df)})")
            continue

        universe_data[instrument] = data_dict
        base_tf_data[instrument] = base_df

    if not universe_data:
        raise SystemExit("No usable instrument data found.")

    corr = _compute_correlation(base_tf_data)
    rows: List[Dict] = []
    window_rows: List[Dict] = []

    inst_list = list(universe_data.keys())
    if args.workers == 1:
        for instrument in inst_list:
            inst_rows, inst_window_rows = _evaluate_instrument(
                instrument=instrument,
                universe_data=universe_data,
                base_tf=base_tf,
                candidates=candidates,
                execution_cfg=execution_cfg,
                train_ratio=args.train_ratio,
                wf_windows=args.wf_windows,
            )
            rows.extend(inst_rows)
            window_rows.extend(inst_window_rows)
            print(f"Completed {instrument}: rows={len(inst_rows)} windows={len(inst_window_rows)}", flush=True)
    else:
        global _MP_UNIVERSE_DATA, _MP_BASE_TF, _MP_CANDIDATES, _MP_EXECUTION_CFG, _MP_TRAIN_RATIO, _MP_WF_WINDOWS
        _MP_UNIVERSE_DATA = universe_data
        _MP_BASE_TF = base_tf
        _MP_CANDIDATES = candidates
        _MP_EXECUTION_CFG = execution_cfg
        _MP_TRAIN_RATIO = args.train_ratio
        _MP_WF_WINDOWS = args.wf_windows

        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
                futures = {executor.submit(_evaluate_instrument_mp, instrument): instrument for instrument in inst_list}
                for future in concurrent.futures.as_completed(futures):
                    instrument = futures[future]
                    try:
                        inst_rows, inst_window_rows = future.result()
                        rows.extend(inst_rows)
                        window_rows.extend(inst_window_rows)
                        print(
                            f"Completed {instrument}: rows={len(inst_rows)} windows={len(inst_window_rows)}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"Worker failed for {instrument}: {exc}", flush=True)
        except (PermissionError, OSError) as exc:
            print(
                f"Multiprocessing unavailable in this environment ({exc}); falling back to sequential mode.",
                flush=True,
            )
            for instrument in inst_list:
                inst_rows, inst_window_rows = _evaluate_instrument(
                    instrument=instrument,
                    universe_data=universe_data,
                    base_tf=base_tf,
                    candidates=candidates,
                    execution_cfg=execution_cfg,
                    train_ratio=args.train_ratio,
                    wf_windows=args.wf_windows,
                )
                rows.extend(inst_rows)
                window_rows.extend(inst_window_rows)
                print(f"Completed {instrument}: rows={len(inst_rows)} windows={len(inst_window_rows)}", flush=True)

    rows.sort(
        key=lambda r: (
            float(r["stability_score"]),
            float(r["test_net_pnl_mean"]),
            float(r["test_profit_factor_mean"]),
            -float(r["test_max_drawdown_mean"]),
        ),
        reverse=True,
    )

    shortlist_rows: List[Dict] = []
    for row in rows:
        if float(row["stability_score"]) < args.min_stability:
            continue
        if float(row["test_trades_mean"]) < args.min_trades:
            continue

        allow = True
        if not corr.empty:
            this_inst = row["instrument"]
            if this_inst not in corr.columns:
                continue
            for picked in shortlist_rows:
                picked_inst = picked["instrument"]
                if picked_inst not in corr.columns:
                    continue
                pair_corr = float(corr.loc[this_inst, picked_inst])
                if abs(pair_corr) > float(args.max_corr):
                    allow = False
                    break
        if allow:
            shortlist_rows.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    base = f"universe_research_{stamp}"
    csv_path = out_dir / f"{base}.csv"
    windows_csv_path = out_dir / f"{base}_windows.csv"
    shortlist_csv_path = out_dir / f"{base}_shortlist.csv"
    md_path = out_dir / f"{base}.md"
    corr_path = out_dir / f"{base}_correlation.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "instrument",
            "strategy_name",
            "params",
            "windows_run",
            "positive_window_ratio",
            "stability_score",
            "train_net_pnl_mean",
            "test_net_pnl_mean",
            "test_net_pnl_std",
            "test_trades_mean",
            "test_expectancy_mean",
            "test_win_rate_mean",
            "test_profit_factor_mean",
            "test_max_drawdown_mean",
            "test_fees_mean",
            "test_sharpe_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if window_rows:
        with windows_csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "instrument",
                "strategy_name",
                "window",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "train_net_pnl",
                "test_net_pnl",
                "test_trades",
                "test_expectancy",
                "test_win_rate",
                "test_profit_factor",
                "test_max_drawdown",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in window_rows:
                writer.writerow(row)

    if shortlist_rows:
        with shortlist_csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "instrument",
                "strategy_name",
                "params",
                "windows_run",
                "positive_window_ratio",
                "stability_score",
                "test_net_pnl_mean",
                "test_net_pnl_std",
                "test_trades_mean",
                "test_expectancy_mean",
                "test_win_rate_mean",
                "test_profit_factor_mean",
                "test_max_drawdown_mean",
                "test_fees_mean",
                "test_sharpe_mean",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in shortlist_rows:
                writer.writerow({k: row.get(k) for k in fieldnames})

    if not corr.empty:
        corr.to_csv(corr_path)

    md_lines = ["# Universe Strategy Research", ""]
    md_lines.append(f"Generated: {_utc_now().isoformat()}")
    md_lines.append(f"Instruments: {', '.join(universe_data.keys())}")
    md_lines.append(f"Walk-forward windows requested: {args.wf_windows}")
    md_lines.append("")
    md_lines.append("## Top 15 Candidates (Walk-Forward Stability Ranking)")
    md_lines.append("")
    md_lines.append(
        "| Rank | Instrument | Strategy | Windows | Stability | Mean Net PnL | Net Std | Mean Expectancy | Mean Win | Mean PF | Mean Max DD |"
    )
    md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows[:15], 1):
        md_lines.append(
            f"| {i} | {r['instrument']} | {r['strategy_name']} | {r['windows_run']} | {r['stability_score']} | "
            f"{r['test_net_pnl_mean']} | {r['test_net_pnl_std']} | {r['test_expectancy_mean']} | "
            f"{float(r['test_win_rate_mean']):.2%} | {r['test_profit_factor_mean']} | {float(r['test_max_drawdown_mean']):.2%} |"
        )
    if corr.empty:
        md_lines.append("\nCorrelation matrix unavailable (insufficient overlapping data).")
    else:
        md_lines.append("\nCorrelation CSV: `{}`".format(corr_path))
    if window_rows:
        md_lines.append("Window-level CSV: `{}`".format(windows_csv_path))
    md_lines.append("")
    md_lines.append("## Correlation-Aware Shortlist")
    md_lines.append("")
    md_lines.append(
        f"Filters: stability >= {args.min_stability}, mean trades >= {args.min_trades}, |corr| <= {args.max_corr}"
    )
    if not shortlist_rows:
        md_lines.append("No candidates passed shortlist filters.")
    else:
        md_lines.append("")
        md_lines.append("| Rank | Instrument | Strategy | Stability | Mean Net PnL | Mean Trades | Mean PF | Mean Max DD |")
        md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for i, r in enumerate(shortlist_rows, 1):
            md_lines.append(
                f"| {i} | {r['instrument']} | {r['strategy_name']} | {r['stability_score']} | "
                f"{r['test_net_pnl_mean']} | {r['test_trades_mean']} | {r['test_profit_factor_mean']} | "
                f"{float(r['test_max_drawdown_mean']):.2%} |"
            )
        md_lines.append("")
        md_lines.append(f"Shortlist CSV: `{shortlist_csv_path}`")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    if window_rows:
        print(f"Windows CSV: {windows_csv_path}")
    if shortlist_rows:
        print(f"Shortlist CSV: {shortlist_csv_path}")
    print(f"Report: {md_path}")
    if not corr.empty:
        print(f"Correlation: {corr_path}")
    if rows:
        top = rows[0]
        print(
            "Top candidate: "
            f"{top['instrument']} / {top['strategy_name']} "
            f"stability={top['stability_score']} mean_net={top['test_net_pnl_mean']} pf={top['test_profit_factor_mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
