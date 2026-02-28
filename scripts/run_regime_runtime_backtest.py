#!/usr/bin/env python3
"""Run regime-classifier driven strategy selection in Backtester runtime."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.core.backtester import Backtester
from backtesting.core.regime_runtime import (
    KMeansRegimePredictor,
    MultiTimeframeRegimeFeatureEngineer,
    RegimeFeatureEngineer,
    RegimeModel,
)
from backtesting.core.timeframe import Timeframe
from backtesting.data.manager import DataManager
from backtesting.strategy.examples.atr_breakout import ATRBreakout
from backtesting.strategy.examples.breakout import Breakout
from backtesting.strategy.examples.ema_pullback import EMATrendPullback
from backtesting.strategy.examples.mean_reversion import MeanReversion
from backtesting.strategy.examples.regime_ensemble_decision import RegimeEnsembleDecisionStrategy
from backtesting.strategy.examples.regime_switch_router import RegimeSwitchRouter
from backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from backtesting.strategy.examples.volatility_compression_breakout import VolatilityCompressionBreakout


def parse_args():
    p = argparse.ArgumentParser(description="Run regime runtime backtest with strategy auto-selection.")
    p.add_argument("--model-json", required=True, help="Path from run_regime_gpu_research *_runtime_model.json")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--tf", default="M15")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fill-mode", default="next_open", choices=["touch", "next_open"])
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--decision-mode", default="ensemble", choices=["ensemble", "router"])
    p.add_argument("--risk-per-trade-pct", type=float, default=0.01)
    p.add_argument("--max-notional-exposure-pct", type=float, default=1.0)
    p.add_argument("--min-quantity", type=int, default=1)
    p.add_argument("--max-quantity", type=int, default=100000)
    p.add_argument("--max-drawdown-stop-pct", type=float, default=0.20)
    p.add_argument("--daily-loss-limit-pct", type=float, default=0.05)
    p.add_argument("--financing", choices=["on", "off"], default="on")
    p.add_argument("--default-financing-long-rate", type=float, default=0.03)
    p.add_argument("--default-financing-short-rate", type=float, default=0.03)
    p.add_argument("--rollover-hour-utc", type=int, default=22)
    p.add_argument(
        "--strategy-params-csv",
        default="",
        help="Optional universe shortlist CSV to override runtime strategy params for the selected instrument.",
    )
    return p.parse_args()


def _strategy_library(tf: Timeframe) -> Dict[str, Dict]:
    return {
        "Breakout": {
            "name": "Breakout_runtime",
            "class": Breakout,
            "timeframes": [tf],
            "lookback": 40,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.008,
            "min_breakout_pct": 0.0002,
            "quantity": 10000,
        },
        "MeanReversion": {
            "name": "MeanRev_runtime",
            "class": MeanReversion,
            "timeframes": [tf],
            "sma_period": 20,
            "deviation_pct": 0.003,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": 10000,
        },
        "EMATrendPullback": {
            "name": "EMAPullback_runtime",
            "class": EMATrendPullback,
            "timeframes": [tf],
            "fast_period": 20,
            "slow_period": 100,
            "pullback_pct": 0.0008,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.009,
            "quantity": 10000,
        },
        "ATRBreakout": {
            "name": "ATRBreakout_runtime",
            "class": ATRBreakout,
            "timeframes": [tf],
            "lookback": 20,
            "atr_period": 14,
            "atr_mult": 1.2,
            "stop_loss_pct": 0.0045,
            "take_profit_pct": 0.009,
            "quantity": 10000,
        },
        "RSIBollingerReversion": {
            "name": "RSIBB_runtime",
            "class": RSIBollingerReversion,
            "timeframes": [tf],
            "window": 20,
            "std_mult": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": 10000,
        },
        "VolatilityCompressionBreakout": {
            "name": "VolCompression_runtime",
            "class": VolatilityCompressionBreakout,
            "timeframes": [tf],
            "range_lookback": 20,
            "atr_period": 14,
            "compression_window": 40,
            "compression_ratio": 0.75,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.01,
            "quantity": 10000,
        },
    }

def _regime_style_weights(regime_to_strategy: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for regime, strategy_name in regime_to_strategy.items():
        weights = {"trend": 0.7, "mean_reversion": 0.7, "breakout": 0.7, "stat_arb": 0.4}
        n = str(strategy_name)
        if "MeanReversion" in n or "RSI" in n:
            weights["mean_reversion"] = 1.8
        if "Breakout" in n or "Compression" in n:
            weights["breakout"] = 1.8
        if "EMA" in n or "Trend" in n:
            weights["trend"] = 1.8
        out[str(regime)] = weights
    return out


def _allowed_override_keys() -> Dict[str, set]:
    return {
        "Breakout": {"lookback", "stop_loss_pct", "take_profit_pct", "min_breakout_pct", "quantity"},
        "MeanReversion": {"sma_period", "deviation_pct", "stop_loss_pct", "take_profit_pct", "quantity"},
        "EMATrendPullback": {"fast_period", "slow_period", "pullback_pct", "stop_loss_pct", "take_profit_pct", "quantity"},
        "ATRBreakout": {"lookback", "atr_period", "atr_mult", "stop_loss_pct", "take_profit_pct", "quantity"},
        "RSIBollingerReversion": {
            "window",
            "std_mult",
            "rsi_period",
            "rsi_oversold",
            "rsi_overbought",
            "stop_loss_pct",
            "take_profit_pct",
            "quantity",
        },
        "VolatilityCompressionBreakout": {
            "range_lookback",
            "atr_period",
            "compression_window",
            "compression_ratio",
            "stop_loss_pct",
            "take_profit_pct",
            "quantity",
        },
    }


def _load_param_overrides_from_csv(csv_path: str, instrument: str) -> Dict[str, Dict]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"strategy params CSV not found: {csv_path}")

    best_rows: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("instrument") != instrument:
                continue
            strategy = str(row.get("strategy_name", "")).strip()
            if not strategy:
                continue
            try:
                stability = float(row.get("stability_score", "nan"))
            except ValueError:
                stability = float("nan")
            existing = best_rows.get(strategy)
            if existing is None or stability > existing["stability_score"]:
                best_rows[strategy] = {"stability_score": stability, "row": row}

    out: Dict[str, Dict] = {}
    for strategy, payload in best_rows.items():
        raw_params = payload["row"].get("params", "{}")
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError:
            continue
        if isinstance(params, dict):
            out[strategy] = params
    return out


def _apply_overrides(config: Dict, strategy_name: str, overrides: Dict[str, Dict]) -> Dict:
    params = overrides.get(strategy_name)
    if not params:
        return {}
    allowed = _allowed_override_keys().get(strategy_name, set())
    applied = {}
    for k, v in params.items():
        if k in allowed:
            config[k] = v
            applied[k] = v
    return applied


def _default_reference_price(instrument: str) -> float:
    defaults = {
        "EUR_USD": 1.10,
        "GBP_USD": 1.28,
        "USD_JPY": 150.0,
        "XAU_USD": 2000.0,
    }
    return float(defaults.get(instrument, 1.0))


def _reference_price_from_data(
    instrument: str,
    tf: Timeframe,
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[float, str]:
    try:
        dm = DataManager({})
        data = dm.ensure_data(
            instrument=instrument,
            base_timeframe=tf,
            start_date=start,
            end_date=end,
            timeframes=[tf],
            force_download=False,
        )
        df = data.get(tf)
        if df is not None and not df.empty and "close" in df.columns:
            series = df["close"].dropna()
            if not series.empty:
                median_close = float(series.median())
                if median_close > 0:
                    return median_close, "data_median_close"
    except Exception:
        pass
    return _default_reference_price(instrument), "static_fallback"


def _usd_notional_per_unit(instrument: str, ref_price: float) -> float:
    if instrument.startswith("USD_"):
        return 1.0
    if instrument.endswith("_USD"):
        return max(ref_price, 1e-9)
    return 1.0


def _risk_size_quantity(
    *,
    instrument: str,
    initial_capital: float,
    risk_per_trade_pct: float,
    max_notional_exposure_pct: float,
    stop_loss_pct: float,
    min_quantity: int,
    max_quantity: int,
    reference_price: float,
) -> int:
    risk_budget = max(initial_capital, 0.0) * max(risk_per_trade_pct, 0.0)
    stop_loss_pct = max(float(stop_loss_pct), 1e-9)
    reference_price = max(float(reference_price), 1e-9)

    risk_per_unit = reference_price * stop_loss_pct
    qty_from_risk = int(risk_budget / risk_per_unit) if risk_per_unit > 0 else 0

    usd_per_unit = _usd_notional_per_unit(instrument, reference_price)
    max_notional_usd = max(initial_capital, 0.0) * max(max_notional_exposure_pct, 0.0)
    qty_from_notional = int(max_notional_usd / max(usd_per_unit, 1e-9)) if max_notional_usd > 0 else 0

    qty = min(qty_from_risk, qty_from_notional)
    if max_quantity > 0:
        qty = min(qty, max_quantity)
    if qty < max(min_quantity, 1):
        return 0
    return int(qty)


def _strategy_stop_loss_pct(strategy_name: str, config: Dict) -> float:
    if "stop_loss_pct" in config:
        try:
            return float(config["stop_loss_pct"])
        except (TypeError, ValueError):
            return 0.004
    # Conservative fallback for unknown strategy configs.
    if strategy_name == "ATRBreakout":
        return 0.0045
    return 0.004


def _apply_runtime_risk_sizing(
    *,
    instrument: str,
    tf: Timeframe,
    start: dt.datetime,
    end: dt.datetime,
    initial_capital: float,
    risk_per_trade_pct: float,
    max_notional_exposure_pct: float,
    min_quantity: int,
    max_quantity: int,
    strategy_cfgs: Dict[str, Dict],
) -> tuple[Dict[str, int], float, str]:
    ref_price, ref_source = _reference_price_from_data(
        instrument=instrument,
        tf=tf,
        start=start,
        end=end,
    )
    assigned: Dict[str, int] = {}
    for strategy_name, cfg in strategy_cfgs.items():
        stop_loss_pct = _strategy_stop_loss_pct(strategy_name, cfg)
        qty = _risk_size_quantity(
            instrument=instrument,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            max_notional_exposure_pct=max_notional_exposure_pct,
            stop_loss_pct=stop_loss_pct,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            reference_price=ref_price,
        )
        cfg["quantity"] = int(qty)
        assigned[strategy_name] = int(qty)
    return assigned, ref_price, ref_source


class RuntimeGuardrailRiskManager:
    def __init__(
        self,
        initial_capital: float,
        max_drawdown_stop_pct: Optional[float],
        daily_loss_limit_pct: Optional[float],
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.max_drawdown_stop_pct = (
            float(max_drawdown_stop_pct) if max_drawdown_stop_pct is not None else None
        )
        self.daily_loss_limit_pct = (
            float(daily_loss_limit_pct) if daily_loss_limit_pct is not None else None
        )
        self.peak_equity = float(initial_capital)
        self.current_day = None
        self.day_start_equity = float(initial_capital)
        self.rejections = {"drawdown_stop": 0, "daily_loss_stop": 0}

    def assess(self, signal, bar, portfolio, state):
        system_state = state.get("system_state")
        equity = float(getattr(system_state, "total_equity", self.initial_capital))
        if equity > self.peak_equity:
            self.peak_equity = equity

        day = getattr(bar.timestamp, "date", lambda: None)()
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity

        if self.max_drawdown_stop_pct is not None and self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd >= self.max_drawdown_stop_pct:
                self.rejections["drawdown_stop"] += 1
                return None

        if self.daily_loss_limit_pct is not None and self.day_start_equity > 0:
            daily_loss = (self.day_start_equity - equity) / self.day_start_equity
            if daily_loss >= self.daily_loss_limit_pct:
                self.rejections["daily_loss_stop"] += 1
                return None

        return signal


def main() -> int:
    args = parse_args()
    tf = Timeframe.from_oanda_granularity(args.tf)
    model = RegimeModel.load(args.model_json)
    if any(col.startswith(("m15_", "h1_", "h4_")) for col in model.feature_columns):
        feature_engineer = MultiTimeframeRegimeFeatureEngineer()
    else:
        feature_engineer = RegimeFeatureEngineer()
    regime_predictor = KMeansRegimePredictor(model)

    library = _strategy_library(tf)
    start_dt = dt.datetime.fromisoformat(args.start)
    end_dt = dt.datetime.fromisoformat(args.end)
    overrides = _load_param_overrides_from_csv(args.strategy_params_csv, args.instrument)
    applied_overrides: Dict[str, Dict] = {}
    for strategy_name, cfg in library.items():
        applied = _apply_overrides(cfg, strategy_name, overrides)
        if applied:
            applied_overrides[strategy_name] = applied
    assigned_quantities, reference_price_used, reference_price_source = _apply_runtime_risk_sizing(
        instrument=args.instrument,
        tf=tf,
        start=start_dt,
        end=end_dt,
        initial_capital=float(args.initial_capital),
        risk_per_trade_pct=float(args.risk_per_trade_pct),
        max_notional_exposure_pct=float(args.max_notional_exposure_pct),
        min_quantity=int(args.min_quantity),
        max_quantity=int(args.max_quantity),
        strategy_cfgs=library,
    )

    strategies = {}
    for strategy_name in set(model.regime_to_strategy.values()):
        if strategy_name in library:
            strategies[strategy_name] = library[strategy_name]
    if not strategies:
        raise SystemExit("No runtime strategies matched model regime_to_strategy mapping.")

    default_strategy = next(iter(strategies.keys()))
    router_cfg = {
        "name": "RegimeSwitchRouter",
        "class": RegimeSwitchRouter,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "regime_to_strategy": model.regime_to_strategy,
        "default_strategy": default_strategy,
        "strategies": strategies,
    }
    ensemble_cfg = {
        "name": "RegimeEnsemble",
        "class": RegimeEnsembleDecisionStrategy,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "decision_threshold": 0.25,
        "min_votes": 1,
        "regime_style_weights": _regime_style_weights(model.regime_to_strategy),
        "modules": {
            "trend_ema_pullback": {
                "name": "trend_ema_pullback",
                "class": EMATrendPullback,
                "timeframes": [tf],
                "style": "trend",
                "weight": 1.0,
                "fast_period": 20,
                "slow_period": 100,
                "pullback_pct": 0.0008,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
            "trend_breakout": {
                "name": "trend_breakout",
                "class": Breakout,
                "timeframes": [tf],
                "style": "trend",
                "weight": 0.9,
                "lookback": 40,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.008,
                "min_breakout_pct": 0.0002,
                "quantity": 10000,
            },
            "range_mean_reversion": {
                "name": "range_mean_reversion",
                "class": MeanReversion,
                "timeframes": [tf],
                "style": "mean_reversion",
                "weight": 1.0,
                "sma_period": 20,
                "deviation_pct": 0.003,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.003,
                "quantity": 10000,
            },
            "range_rsi_reversion": {
                "name": "range_rsi_reversion",
                "class": RSIBollingerReversion,
                "timeframes": [tf],
                "style": "mean_reversion",
                "weight": 0.8,
                "window": 20,
                "std_mult": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.003,
                "quantity": 10000,
            },
            "vol_breakout": {
                "name": "vol_breakout",
                "class": ATRBreakout,
                "timeframes": [tf],
                "style": "breakout",
                "weight": 1.0,
                "lookback": 20,
                "atr_period": 14,
                "atr_mult": 1.2,
                "stop_loss_pct": 0.0045,
                "take_profit_pct": 0.009,
                "quantity": 10000,
            },
            "vol_compression": {
                "name": "vol_compression",
                "class": VolatilityCompressionBreakout,
                "timeframes": [tf],
                "style": "breakout",
                "weight": 0.9,
                "range_lookback": 20,
                "atr_period": 14,
                "compression_window": 40,
                "compression_ratio": 0.75,
                "stop_loss_pct": 0.004,
                "take_profit_pct": 0.01,
                "quantity": 10000,
            },
        },
    }
    module_strategy_map = {
        "trend_ema_pullback": "EMATrendPullback",
        "trend_breakout": "Breakout",
        "range_mean_reversion": "MeanReversion",
        "range_rsi_reversion": "RSIBollingerReversion",
        "vol_breakout": "ATRBreakout",
        "vol_compression": "VolatilityCompressionBreakout",
    }
    for module_name, strategy_name in module_strategy_map.items():
        if module_name in ensemble_cfg["modules"]:
            _apply_overrides(ensemble_cfg["modules"][module_name], strategy_name, overrides)
            if strategy_name in assigned_quantities:
                ensemble_cfg["modules"][module_name]["quantity"] = int(assigned_quantities[strategy_name])

    selected_strategy_cfg = ensemble_cfg if args.decision_mode == "ensemble" else router_cfg

    ctx = {
        "data": {
            "instrument": args.instrument,
            "base_timeframe": tf,
            "start_date": start_dt,
            "end_date": end_dt,
        },
        "strategy": selected_strategy_cfg,
        "execution": {
            "initial_capital": float(args.initial_capital),
            "fill_mode": args.fill_mode,
            "slippage_pips": 0.2,
            "pricing_model": "oanda_core",
            "spreads_pips": {
                "EUR_USD": 1.4,
                "GBP_USD": 2.0,
                "USD_JPY": 1.4,
                "XAU_USD": 20.0,
            },
            "core_commission_per_10k_units": 1.0,
            "min_quantity": int(args.min_quantity),
            "max_quantity": int(args.max_quantity),
            "financing_enabled": args.financing == "on",
            "default_financing_long_rate": float(args.default_financing_long_rate),
            "default_financing_short_rate": float(args.default_financing_short_rate),
            "rollover_hour_utc": int(args.rollover_hour_utc),
            "wednesday_triple_rollover": True,
            # Annualized financing assumptions; positive values are costs.
            "financing_long_rate_by_instrument": {
                "EUR_USD": 0.025,
                "GBP_USD": 0.03,
                "USD_JPY": 0.015,
                "XAU_USD": 0.10,
            },
            "financing_short_rate_by_instrument": {
                "EUR_USD": 0.02,
                "GBP_USD": 0.025,
                "USD_JPY": 0.015,
                "XAU_USD": 0.08,
            },
        },
    }

    risk_manager = RuntimeGuardrailRiskManager(
        initial_capital=float(args.initial_capital),
        max_drawdown_stop_pct=float(args.max_drawdown_stop_pct),
        daily_loss_limit_pct=float(args.daily_loss_limit_pct),
    )

    result = Backtester(
        context=ctx,
        feature_engineer=feature_engineer,
        regime_predictor=regime_predictor,
        risk_manager=risk_manager,
    ).run()

    print(f"Trades: {result.total_trades}")
    print(f"Final equity: {result.final_equity:.2f}")
    print(f"Net PnL: {result.final_equity - args.initial_capital:.2f}")
    print(f"Win rate: {result.win_rate:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Fees: {result.total_fees_paid:.2f}")
    print(f"Financing: {result.total_financing_paid:.2f}")
    print(f"Regime counts: {regime_predictor.regime_counts}")
    print(f"Regime->strategy: {model.regime_to_strategy}")
    print(f"Decision mode: {args.decision_mode}")
    print(f"Strategy params CSV: {args.strategy_params_csv or 'none'}")
    print(f"Applied param overrides: {applied_overrides if applied_overrides else 'none'}")
    print(
        "Risk controls: "
        f"risk_per_trade_pct={args.risk_per_trade_pct}, "
        f"max_notional_exposure_pct={args.max_notional_exposure_pct}, "
        f"min_quantity={args.min_quantity}, max_quantity={args.max_quantity}, "
        f"max_drawdown_stop_pct={args.max_drawdown_stop_pct}, daily_loss_limit_pct={args.daily_loss_limit_pct}"
    )
    print(
        "Financing config: "
        f"enabled={args.financing}, rollover_hour_utc={args.rollover_hour_utc}, "
        f"default_long_rate={args.default_financing_long_rate}, default_short_rate={args.default_financing_short_rate}"
    )
    print(f"Assigned quantities: {assigned_quantities}")
    print(f"Reference price used: {reference_price_used:.6f} ({reference_price_source})")
    print(f"Risk manager rejections: {risk_manager.rejections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
