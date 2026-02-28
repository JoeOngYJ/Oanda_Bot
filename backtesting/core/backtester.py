"""Real-time style backtester with explicit pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

import pandas as pd

from backtesting.core.engine import BacktestEngine, BacktestResult
from backtesting.core.state import SystemState
from backtesting.data.models import OHLCVBar
from backtesting.strategy.signal import Signal


class FeatureEngineer:
    """Feature hook for per-bar feature computation."""

    def on_market_bar(self, bar: OHLCVBar, state: Dict[str, Any]) -> None:
        """Optional hook to ingest every market bar before base-tf compute."""
        return None

    def compute(self, bar: OHLCVBar, state: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class RegimePredictor:
    """Regime hook for per-bar market regime tagging."""

    def predict(self, bar: OHLCVBar, features: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
        return None


class RiskManager:
    """Risk hook for per-signal filtering or transformation."""

    def assess(
        self,
        signal: Signal,
        bar: OHLCVBar,
        portfolio,
        state: Dict[str, Any],
    ) -> Optional[Signal]:
        return signal


@dataclass
class Backtester:
    """
    Backtester that simulates real-time stage ordering on each base-timeframe bar:
    1) data ingestion
    2) feature compute
    3) regime predict
    4) strategy signal
    5) risk assess
    6) execution + state update
    """

    context: Dict[str, Any]
    feature_engineer: FeatureEngineer = FeatureEngineer()
    regime_predictor: RegimePredictor = RegimePredictor()
    risk_manager: RiskManager = RiskManager()
    latest_state: Optional[SystemState] = None

    def run(self) -> BacktestResult:
        engine = BacktestEngine(self.context)
        config = self.context or {}
        loaded = engine._load_data(config)
        strategy = engine._build_strategy(config)
        simulator = engine._build_execution_simulator(config)

        system_state = SystemState.create(config["execution"]["initial_capital"])
        self.latest_state = system_state
        state: Dict[str, Any] = {
            "system_state": system_state,
            "bar_count": 0,
            "last_regime": None,
            "last_features": {},
            "snapshot_path": config.get("state", {}).get("snapshot_path"),
            "snapshot_every_bars": int(config.get("state", {}).get("snapshot_every_bars", 1)),
        }

        if engine._is_market_data_dict(loaded):
            market_data_dict = loaded
            primary_instrument = str(config["data"]["instrument"])
            base_tf = config["data"]["base_timeframe"]
            events = engine._build_market_events(market_data_dict)
            for timestamp, instrument, tf, row in events:
                bar = engine._row_to_bar(timestamp, tf, row, instrument)
                self.feature_engineer.on_market_bar(bar, state)
                strategy.on_market_bar(bar)
                if instrument == primary_instrument and tf == base_tf:
                    self._step(bar, strategy, simulator, state)
                if tf == base_tf:
                    prev = len(simulator.filled_orders)
                    simulator.process_bar(bar)
                    self._post_execution_update(simulator, state, new_fills=simulator.filled_orders[prev:])
        else:
            data_dict = loaded
            base_tf = config["data"]["base_timeframe"]
            base_df = data_dict[base_tf]
            tf_indices = {tf: set(df.index) for tf, df in data_dict.items()}

            for timestamp in base_df.index:
                for tf, df in data_dict.items():
                    if timestamp not in tf_indices[tf]:
                        continue
                    bar = engine._row_to_bar(timestamp, tf, df.loc[timestamp], config["data"]["instrument"])
                    self.feature_engineer.on_market_bar(bar, state)
                    strategy.on_market_bar(bar)

                base_bar = engine._row_to_bar(
                    timestamp,
                    base_tf,
                    base_df.loc[timestamp],
                    config["data"]["instrument"],
                )
                self._step(base_bar, strategy, simulator, state)
                prev = len(simulator.filled_orders)
                simulator.process_bar(base_bar)
                self._post_execution_update(simulator, state, new_fills=simulator.filled_orders[prev:])

        trades, equity_curve, max_dd = engine._build_trade_stats(
            simulator.filled_orders,
            Decimal(str(config["execution"]["initial_capital"])),
        )
        total_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = (wins / total_trades) if total_trades else 0.0
        sharpe = engine._trade_sharpe(trades, Decimal(str(config["execution"]["initial_capital"])))
        final_equity = equity_curve[-1] if equity_curve else float(config["execution"]["initial_capital"])

        return BacktestResult(
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            final_equity=final_equity,
            total_fees_paid=float(simulator.portfolio.total_fees_paid),
            trades=trades,
            equity_curve=equity_curve,
            filled_orders=simulator.filled_orders,
        )

    def _step(self, bar: OHLCVBar, strategy, simulator, state: Dict[str, Any]) -> None:
        system_state: SystemState = state["system_state"]
        system_state.ingest_bar(bar)
        features = self.feature_engineer.compute(bar, state)
        system_state.append_features(bar, features)
        regime = self.regime_predictor.predict(bar, features, state)
        regime_probabilities = dict(getattr(self.regime_predictor, "last_probabilities", {}) or {})
        if hasattr(strategy, "set_regime"):
            strategy.set_regime(regime)
        if hasattr(strategy, "set_regime_probabilities"):
            strategy.set_regime_probabilities(regime_probabilities)
        signal = strategy.on_bar(bar)
        state["last_features"] = features
        state["last_regime"] = regime
        state["last_regime_probabilities"] = regime_probabilities
        system_state.last_detected_regime = regime

        if signal is None:
            return

        metadata = dict(signal.metadata or {})
        metadata["features"] = features
        metadata["regime"] = regime
        metadata["regime_probabilities"] = regime_probabilities
        signal.metadata = metadata

        approved = self.risk_manager.assess(signal, bar, simulator.portfolio, state)
        if approved is not None:
            simulator.process_signal(approved)

    def _post_execution_update(self, simulator, state: Dict[str, Any], new_fills: list) -> None:
        system_state: SystemState = state["system_state"]
        system_state.bar_count += 1
        state["bar_count"] = system_state.bar_count
        if new_fills:
            system_state.append_fills(new_fills)
        system_state.sync_from_portfolio(simulator.portfolio)
        snapshot_path = state.get("snapshot_path")
        snapshot_every = max(int(state.get("snapshot_every_bars", 1)), 1)
        if snapshot_path and (system_state.bar_count % snapshot_every == 0):
            system_state.persist_snapshot(snapshot_path)
