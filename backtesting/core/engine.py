"""Main backtest orchestrator."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol
from backtesting.data.models import OHLCVBar
from backtesting.execution.commission import (
    FixedPerTradeCommissionModel,
    OandaCoreCommissionModel,
    OandaSpreadOnlyCommissionModel,
)
from backtesting.execution.simulator import ExecutionSimulator
from backtesting.execution.slippage import SlippageModel
from backtesting.strategy.signal import Signal, SignalDirection


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    final_equity: float
    total_fees_paid: float
    total_financing_paid: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    filled_orders: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """Event-loop style backtest runner using strategy + execution simulator."""

    def __init__(self, context: Optional[dict] = None):
        self.context = context or {}

    def run(self) -> BacktestResult:
        config = self.context or {}
        loaded = self._load_data(config)
        strategy = self._build_strategy(config)
        simulator = self._build_execution_simulator(config)

        if self._is_market_data_dict(loaded):
            market_data_dict = loaded
            primary_instrument = str(config["data"]["instrument"])
            base_tf = config["data"]["base_timeframe"]
            events = self._build_market_events(market_data_dict)

            for timestamp, instrument, tf, row in events:
                bar = self._row_to_bar(timestamp, tf, row, instrument)
                strategy.on_market_bar(bar)
                if instrument == primary_instrument:
                    signal = strategy.on_bar(bar)
                    if signal is not None:
                        simulator.process_signal(signal)
                if tf == base_tf:
                    simulator.process_bar(bar)
        else:
            data_dict = loaded
            base_tf = config["data"]["base_timeframe"]
            base_df = data_dict[base_tf]
            tf_indices = {tf: set(df.index) for tf, df in data_dict.items()}

            # Main backtest loop.
            for timestamp in base_df.index:
                # Feed only closed bars per timeframe to avoid synthetic duplicates.
                for tf, df in data_dict.items():
                    if timestamp not in tf_indices[tf]:
                        continue
                    bar = self._row_to_bar(timestamp, tf, df.loc[timestamp], config["data"]["instrument"])
                    strategy.on_market_bar(bar)
                    signal = strategy.on_bar(bar)
                    if signal is not None:
                        simulator.process_signal(signal)

                # Use base timeframe bar for execution checks.
                base_bar = self._row_to_bar(
                    timestamp,
                    base_tf,
                    base_df.loc[timestamp],
                    config["data"]["instrument"],
                )
                simulator.process_bar(base_bar)

        trades, equity_curve, max_dd = self._build_trade_stats(
            simulator.filled_orders,
            Decimal(str(config["execution"]["initial_capital"])),
        )
        total_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = (wins / total_trades) if total_trades else 0.0
        sharpe = self._trade_sharpe(trades, Decimal(str(config["execution"]["initial_capital"])))

        final_equity = equity_curve[-1] if equity_curve else float(config["execution"]["initial_capital"])

        return BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            final_equity=final_equity,
            total_fees_paid=float(simulator.portfolio.total_fees_paid),
            total_financing_paid=float(simulator.portfolio.total_financing),
            trades=trades,
            equity_curve=equity_curve,
            filled_orders=simulator.filled_orders,
        )

    def _load_data(self, config: Dict):
        if "market_data_dict" in config and config["market_data_dict"]:
            return config["market_data_dict"]
        if "data_dict" in config and config["data_dict"]:
            return config["data_dict"]

        data_cfg = config.get("data", {})
        instrument = data_cfg["instrument"]
        base_tf = data_cfg["base_timeframe"]
        start = data_cfg.get("start_date")
        end = data_cfg.get("end_date")

        from backtesting.data.manager import DataManager

        manager = DataManager(config)

        strategy_cfg = config.get("strategy", {})
        timeframes = strategy_cfg.get("timeframes", [base_tf])
        return manager.ensure_data(
            instrument=instrument,
            base_timeframe=base_tf,
            start_date=start,
            end_date=end,
            timeframes=timeframes,
            force_download=False,
        )

    @staticmethod
    def _is_market_data_dict(data) -> bool:
        if not isinstance(data, dict) or not data:
            return False
        first_val = next(iter(data.values()))
        return isinstance(first_val, dict)

    @staticmethod
    def _build_market_events(
        market_data_dict: Dict[str, Dict[Timeframe, pd.DataFrame]]
    ) -> List[Tuple[dt.datetime, str, Timeframe, pd.Series]]:
        events: List[Tuple[dt.datetime, str, Timeframe, pd.Series]] = []
        for instrument, tf_map in market_data_dict.items():
            for tf, df in tf_map.items():
                for ts, row in df.sort_index().iterrows():
                    events.append((ts, instrument, tf, row))
        events.sort(key=lambda x: (x[0], x[1], x[2].seconds))
        return events

    def _build_strategy(self, config: Dict):
        strategy_cfg = config["strategy"]
        strategy_cls = strategy_cfg["class"]
        return strategy_cls(strategy_cfg)

    def _build_execution_simulator(self, config: Dict) -> ExecutionSimulator:
        exec_cfg = config.get("execution", {})
        initial_capital = Decimal(str(exec_cfg.get("initial_capital", 10000)))

        spread_cfg = exec_cfg.get("spreads_pips", {})
        default_spread = Decimal(str(exec_cfg.get("default_spread_pips", "1.5")))
        slippage = Decimal(str(exec_cfg.get("slippage_pips", "0")))
        slippage_model = SlippageModel(
            spread_pips_by_instrument=spread_cfg or None,
            default_spread_pips=default_spread,
            slippage_pips=slippage,
        )

        # Backward compatibility: `commission_per_trade` from old scripts.
        if "commission_per_trade" in exec_cfg:
            commission_model = FixedPerTradeCommissionModel(
                Decimal(str(exec_cfg.get("commission_per_trade", 0)))
            )
        else:
            pricing_model = str(exec_cfg.get("pricing_model", "spread_only")).lower()
            if pricing_model == "oanda_core":
                commission_model = OandaCoreCommissionModel(
                    Decimal(str(exec_cfg.get("core_commission_per_10k_units", "1.00")))
                )
            else:
                commission_model = OandaSpreadOnlyCommissionModel()

        return ExecutionSimulator(
            initial_capital=initial_capital,
            slippage_model=slippage_model,
            commission_model=commission_model,
            fill_mode=str(exec_cfg.get("fill_mode", "touch")),
            volatility_targeting_enabled=bool(exec_cfg.get("volatility_targeting_enabled", False)),
            target_annual_volatility=Decimal(str(exec_cfg.get("target_annual_volatility", "0.15"))),
            volatility_lookback_bars=int(exec_cfg.get("volatility_lookback_bars", 96)),
            max_concurrent_exposure_pct=(
                Decimal(str(exec_cfg["max_concurrent_exposure_pct"]))
                if "max_concurrent_exposure_pct" in exec_cfg
                else None
            ),
            min_quantity=int(exec_cfg.get("min_quantity", 1)),
            max_quantity=(
                int(exec_cfg["max_quantity"])
                if "max_quantity" in exec_cfg and exec_cfg.get("max_quantity") is not None
                else None
            ),
            base_timeframe_seconds=int(config["data"]["base_timeframe"].seconds),
            financing_enabled=bool(exec_cfg.get("financing_enabled", False)),
            financing_long_rate_by_instrument=exec_cfg.get("financing_long_rate_by_instrument", {}),
            financing_short_rate_by_instrument=exec_cfg.get("financing_short_rate_by_instrument", {}),
            default_financing_long_rate=Decimal(str(exec_cfg.get("default_financing_long_rate", "0.03"))),
            default_financing_short_rate=Decimal(str(exec_cfg.get("default_financing_short_rate", "0.03"))),
            rollover_hour_utc=int(exec_cfg.get("rollover_hour_utc", 22)),
            wednesday_triple_rollover=bool(exec_cfg.get("wednesday_triple_rollover", True)),
        )

    @staticmethod
    def _row_to_bar(
        timestamp: dt.datetime,
        timeframe: Timeframe,
        row: pd.Series,
        instrument: str,
    ) -> OHLCVBar:
        return OHLCVBar(
            timestamp=timestamp,
            timeframe=timeframe,
            instrument=InstrumentSymbol(instrument),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
        )

    def _build_trade_stats(
        self,
        fills: List[Dict],
        initial_capital: Decimal,
    ) -> tuple[List[Dict], List[float], float]:
        # FIFO matching per instrument to derive realized trades.
        open_lots: Dict[str, Deque[Dict]] = defaultdict(deque)
        trades: List[Dict] = []
        equity = initial_capital
        equity_curve: List[float] = [float(equity)]
        peak = equity
        max_dd = Decimal("0")

        for fill in sorted(fills, key=lambda f: f["timestamp"]):
            instrument = fill["instrument"]
            side = fill["direction"]
            qty_remaining = int(fill["quantity"])
            fill_price = Decimal(str(fill["fill_price"]))
            commission = Decimal(str(fill["commission"]))
            exit_comm_per_unit = commission / Decimal(max(qty_remaining, 1))

            while qty_remaining > 0 and open_lots[instrument]:
                lot = open_lots[instrument][0]
                if lot["direction"] == side:
                    break

                match_qty = min(qty_remaining, lot["quantity"])
                entry_price = lot["entry_price"]
                entry_comm_per_unit = lot["commission_per_unit"]

                if lot["direction"] == SignalDirection.LONG and side == SignalDirection.SHORT:
                    gross = (fill_price - entry_price) * Decimal(match_qty)
                else:
                    gross = (entry_price - fill_price) * Decimal(match_qty)

                trade_commission = (entry_comm_per_unit + exit_comm_per_unit) * Decimal(match_qty)
                pnl = gross - trade_commission

                trades.append(
                    {
                        "timestamp": fill["timestamp"],
                        "instrument": instrument,
                        "entry_side": lot["direction"].value,
                        "exit_side": side.value,
                        "quantity": match_qty,
                        "entry_price": float(entry_price),
                        "exit_price": float(fill_price),
                        "commission": float(trade_commission),
                        "pnl": float(pnl),
                    }
                )

                equity += pnl
                equity_curve.append(float(equity))
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
                if drawdown > max_dd:
                    max_dd = drawdown

                lot["quantity"] -= match_qty
                qty_remaining -= match_qty
                if lot["quantity"] == 0:
                    open_lots[instrument].popleft()

            if qty_remaining > 0:
                open_lots[instrument].append(
                    {
                        "direction": side,
                        "quantity": qty_remaining,
                        "entry_price": fill_price,
                        "commission_per_unit": exit_comm_per_unit,
                    }
                )

        return trades, equity_curve, float(max_dd)

    @staticmethod
    def _trade_sharpe(trades: List[Dict], initial_capital: Decimal) -> float:
        if len(trades) < 2:
            return 0.0
        returns = [Decimal(str(t["pnl"])) / initial_capital for t in trades]
        mean = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean) ** 2 for r in returns) / Decimal(len(returns) - 1)
        if variance <= 0:
            return 0.0
        std = variance.sqrt()
        if std == 0:
            return 0.0
        return float((mean / std) * Decimal(len(returns)).sqrt())
