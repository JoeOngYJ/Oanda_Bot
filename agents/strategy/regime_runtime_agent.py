"""Regime runtime strategy agent for live/paper tick streams.

Builds OHLCV bars from market ticks, runs multi-timeframe regime prediction,
and publishes TradeSignal events into the shared message bus.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from agents.market_data.oanda_client import OandaStreamClient
from backtesting.core.regime_runtime import (
    KMeansRegimePredictor,
    MultiTimeframeRegimeFeatureEngineer,
    RegimeFeatureEngineer,
    RegimeModel,
)
from backtesting.core.timeframe import Timeframe
from backtesting.data.models import OHLCVBar
from backtesting.strategy.examples.atr_breakout import ATRBreakout
from backtesting.strategy.examples.breakout import Breakout
from backtesting.strategy.examples.ema_pullback import EMATrendPullback
from backtesting.strategy.examples.mean_reversion import MeanReversion
from backtesting.strategy.examples.regime_ensemble_decision import RegimeEnsembleDecisionStrategy
from backtesting.strategy.examples.regime_switch_router import RegimeSwitchRouter
from backtesting.strategy.examples.rsi_bollinger_reversion import RSIBollingerReversion
from backtesting.strategy.examples.volatility_compression_breakout import VolatilityCompressionBreakout
from backtesting.strategy.signal import SignalDirection
from shared.config import Config
from shared.message_bus import MessageBus
from shared.models import Instrument, MarketTick, Side, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class _CandleState:
    start: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


def _resolve_warmup_counts(
    base_bars: int,
    m15_bars: Optional[int],
    h1_bars: Optional[int],
    h4_bars: Optional[int],
    d1_bars: Optional[int],
) -> Dict[Timeframe, int]:
    base = max(int(base_bars), 50)
    defaults = {
        Timeframe.M15: max(base, 1500),
        Timeframe.H1: max(base // 4, 400),
        Timeframe.H4: max(base // 16, 250),
        Timeframe.D1: max(base // 96, 200),
    }
    explicit = {
        Timeframe.M15: m15_bars,
        Timeframe.H1: h1_bars,
        Timeframe.H4: h4_bars,
        Timeframe.D1: d1_bars,
    }
    out: Dict[Timeframe, int] = {}
    for tf, default_count in defaults.items():
        v = explicit[tf]
        out[tf] = max(int(v), 50) if v is not None else int(default_count)
    return out


class _BarBuilder:
    def __init__(self, timeframe: Timeframe, instrument: str):
        self.timeframe = timeframe
        self.instrument = instrument
        self.current: Optional[_CandleState] = None

    def _floor_ts(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        epoch = int(ts.timestamp())
        bucket = epoch - (epoch % self.timeframe.seconds)
        return datetime.fromtimestamp(bucket, tz=timezone.utc)

    def update(self, ts: datetime, price: Decimal, volume: int) -> Optional[OHLCVBar]:
        bucket = self._floor_ts(ts)
        if self.current is None:
            self.current = _CandleState(bucket, price, price, price, price, int(volume))
            return None

        if bucket == self.current.start:
            self.current.high = max(self.current.high, price)
            self.current.low = min(self.current.low, price)
            self.current.close = price
            self.current.volume += int(volume)
            return None

        prev = self.current
        self.current = _CandleState(bucket, price, price, price, price, int(volume))
        return OHLCVBar(
            timestamp=prev.start,
            timeframe=self.timeframe,
            instrument=self.instrument,
            open=prev.open,
            high=prev.high,
            low=prev.low,
            close=prev.close,
            volume=int(prev.volume),
        )


def _resolve_gpu_mode(gpu_mode: str) -> bool:
    if gpu_mode == "off":
        return False
    try:
        import cupy as cp  # type: ignore

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        if gpu_mode == "on":
            raise SystemExit("GPU requested (--gpu on) but CuPy/CUDA is unavailable.")
        return False


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


def _build_strategy(tf: Timeframe, regime_to_strategy: Dict[str, str], decision_mode: str, quantity: int):
    library = {
        "Breakout": {
            "name": "Breakout_live",
            "class": Breakout,
            "timeframes": [tf],
            "lookback": 40,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.008,
            "min_breakout_pct": 0.0002,
            "quantity": quantity,
        },
        "MeanReversion": {
            "name": "MeanRev_live",
            "class": MeanReversion,
            "timeframes": [tf],
            "sma_period": 20,
            "deviation_pct": 0.003,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": quantity,
        },
        "EMATrendPullback": {
            "name": "EMAPullback_live",
            "class": EMATrendPullback,
            "timeframes": [tf],
            "fast_period": 20,
            "slow_period": 100,
            "pullback_pct": 0.0008,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.009,
            "quantity": quantity,
        },
        "ATRBreakout": {
            "name": "ATRBreakout_live",
            "class": ATRBreakout,
            "timeframes": [tf],
            "lookback": 20,
            "atr_period": 14,
            "atr_mult": 1.2,
            "stop_loss_pct": 0.0045,
            "take_profit_pct": 0.009,
            "quantity": quantity,
        },
        "RSIBollingerReversion": {
            "name": "RSIBB_live",
            "class": RSIBollingerReversion,
            "timeframes": [tf],
            "window": 20,
            "std_mult": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.003,
            "quantity": quantity,
        },
        "VolatilityCompressionBreakout": {
            "name": "VolCompression_live",
            "class": VolatilityCompressionBreakout,
            "timeframes": [tf],
            "range_lookback": 20,
            "atr_period": 14,
            "compression_window": 40,
            "compression_ratio": 0.75,
            "stop_loss_pct": 0.004,
            "take_profit_pct": 0.01,
            "quantity": quantity,
        },
    }

    strategies = {}
    for strategy_name in set(regime_to_strategy.values()):
        if strategy_name in library:
            strategies[strategy_name] = library[strategy_name]
    if not strategies:
        raise RuntimeError("No runtime strategies matched regime_to_strategy mapping")

    if decision_mode == "router":
        cfg = {
            "name": "RegimeSwitchRouterLive",
            "class": RegimeSwitchRouter,
            "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
            "regime_to_strategy": regime_to_strategy,
            "default_strategy": next(iter(strategies.keys())),
            "strategies": strategies,
        }
        return RegimeSwitchRouter(cfg)

    ensemble_cfg = {
        "name": "RegimeEnsembleLive",
        "class": RegimeEnsembleDecisionStrategy,
        "timeframes": [tf, Timeframe.H1, Timeframe.H4, Timeframe.D1],
        "decision_threshold": 0.25,
        "min_votes": 1,
        "regime_style_weights": _regime_style_weights(regime_to_strategy),
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
                "quantity": quantity,
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
                "quantity": quantity,
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
                "quantity": quantity,
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
                "quantity": quantity,
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
                "quantity": quantity,
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
                "quantity": quantity,
            },
        },
    }
    return RegimeEnsembleDecisionStrategy(ensemble_cfg)


class RegimeRuntimeStrategyAgent:
    def __init__(
        self,
        config: Config,
        message_bus: MessageBus,
        model_json: str,
        instrument: str,
        decision_mode: str,
        quantity: int,
        min_confidence: float,
        gpu_mode: str,
        warmup_enabled: bool,
        warmup_base_bars: int,
        warmup_m15_bars: Optional[int],
        warmup_h1_bars: Optional[int],
        warmup_h4_bars: Optional[int],
        warmup_d1_bars: Optional[int],
    ):
        self.config = config
        self.message_bus = message_bus
        self.model = RegimeModel.load(model_json)
        self.instrument = Instrument(instrument)
        self.instrument_symbol = str(self.instrument.value)
        self.base_tf = Timeframe.M15
        self.running = False
        self.min_confidence = float(min_confidence)
        self.warmup_enabled = bool(warmup_enabled)
        self.warmup_counts = _resolve_warmup_counts(
            base_bars=warmup_base_bars,
            m15_bars=warmup_m15_bars,
            h1_bars=warmup_h1_bars,
            h4_bars=warmup_h4_bars,
            d1_bars=warmup_d1_bars,
        )
        self.oanda_client = OandaStreamClient(config)

        use_gpu = _resolve_gpu_mode(gpu_mode)
        if any(c.startswith(("m15_", "h1_", "h4_", "d1_")) for c in self.model.feature_columns):
            self.feature_engineer = MultiTimeframeRegimeFeatureEngineer(use_gpu=use_gpu)
        else:
            self.feature_engineer = RegimeFeatureEngineer(use_gpu=use_gpu)
        self.regime_predictor = KMeansRegimePredictor(self.model, use_gpu=use_gpu)
        self.strategy = _build_strategy(
            tf=self.base_tf,
            regime_to_strategy=self.model.regime_to_strategy,
            decision_mode=decision_mode,
            quantity=int(quantity),
        )
        self.state: Dict[str, object] = {}

        self.builders: Dict[Timeframe, _BarBuilder] = {
            Timeframe.M15: _BarBuilder(Timeframe.M15, self.instrument_symbol),
            Timeframe.H1: _BarBuilder(Timeframe.H1, self.instrument_symbol),
            Timeframe.H4: _BarBuilder(Timeframe.H4, self.instrument_symbol),
            Timeframe.D1: _BarBuilder(Timeframe.D1, self.instrument_symbol),
        }

    async def start(self) -> None:
        logger.info(
            "Starting RegimeRuntimeStrategyAgent",
            extra={
                "instrument": self.instrument_symbol,
                "base_tf": self.base_tf.name,
                "feature_cols": len(self.model.feature_columns),
            },
        )
        self.running = True
        if self.warmup_enabled:
            await self._warmup_state()

        async for message in self.message_bus.subscribe("market_data"):
            if not self.running:
                break
            try:
                tick = MarketTick(**message)
                if tick.instrument != self.instrument:
                    continue
                await self._on_tick(tick)
            except Exception as e:
                logger.error("Regime strategy tick processing error: %s", e, exc_info=True)

    async def stop(self) -> None:
        self.running = False
        await self.oanda_client.close()

    async def _warmup_state(self) -> None:
        """
        Preload historical bars so regime/strategy state starts with context.
        """
        logger.info(
            "Warmup start",
            extra={
                "instrument": self.instrument_symbol,
                "m15_bars": self.warmup_counts[Timeframe.M15],
                "h1_bars": self.warmup_counts[Timeframe.H1],
                "h4_bars": self.warmup_counts[Timeframe.H4],
                "d1_bars": self.warmup_counts[Timeframe.D1],
            },
        )
        tfs = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        bars_by_tf: Dict[Timeframe, List[OHLCVBar]] = {}
        for tf in tfs:
            count = int(self.warmup_counts[tf])
            raw = await asyncio.to_thread(
                self.oanda_client.get_recent_candles,
                self.instrument,
                tf.to_oanda_granularity(),
                count,
            )
            parsed: List[OHLCVBar] = []
            for row in raw:
                try:
                    parsed.append(
                        OHLCVBar(
                            timestamp=row["time"],
                            timeframe=tf,
                            instrument=self.instrument_symbol,
                            open=Decimal(str(row["open"])),
                            high=Decimal(str(row["high"])),
                            low=Decimal(str(row["low"])),
                            close=Decimal(str(row["close"])),
                            volume=int(row["volume"]),
                        )
                    )
                except Exception:
                    continue
            bars_by_tf[tf] = parsed
            logger.info("Warmup fetched", extra={"tf": tf.name, "bars": len(parsed)})

        merged: List[OHLCVBar] = []
        for seq in bars_by_tf.values():
            merged.extend(seq)
        merged.sort(key=lambda b: (b.timestamp, -b.timeframe.seconds))

        for bar in merged:
            self.feature_engineer.on_market_bar(bar, self.state)
            self.strategy.on_market_bar(bar)
            self.builders[bar.timeframe].current = _CandleState(
                start=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=int(bar.volume),
            )

        logger.info("Warmup complete", extra={"bars_total": len(merged)})

    async def _on_tick(self, tick: MarketTick) -> None:
        ts = tick.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        mid = (tick.bid + tick.ask) / Decimal("2")
        vol = int((tick.bid_volume or 0) + (tick.ask_volume or 0))

        closed: List[OHLCVBar] = []
        for tf, builder in self.builders.items():
            bar = builder.update(ts=ts, price=mid, volume=vol)
            if bar is not None:
                closed.append(bar)

        if not closed:
            return

        # Process higher timeframes first so M15 uses freshest HTF state on boundary bars.
        closed.sort(key=lambda b: b.timeframe.seconds, reverse=True)

        for bar in closed:
            self.feature_engineer.on_market_bar(bar, self.state)
            self.strategy.on_market_bar(bar)
            if bar.timeframe == self.base_tf:
                await self._on_base_bar(bar)

    async def _on_base_bar(self, bar: OHLCVBar) -> None:
        features = self.feature_engineer.compute(bar, self.state)
        regime = self.regime_predictor.predict(bar, features, self.state)
        regime_probs = dict(getattr(self.regime_predictor, "last_probabilities", {}) or {})

        if hasattr(self.strategy, "set_regime"):
            self.strategy.set_regime(regime)
        if hasattr(self.strategy, "set_regime_probabilities"):
            self.strategy.set_regime_probabilities(regime_probs)

        signal = self.strategy.on_bar(bar)
        if signal is None:
            return
        if float(signal.confidence) < self.min_confidence:
            return

        side = Side.BUY if signal.direction == SignalDirection.LONG else Side.SELL
        trade_signal = TradeSignal(
            signal_id=str(uuid.uuid4()),
            instrument=self.instrument,
            side=side,
            quantity=int(signal.quantity),
            confidence=float(signal.confidence),
            rationale=f"Regime={regime} ensemble_score={signal.metadata.get('ensemble_score', 0):.4f}",
            strategy_name=str(signal.strategy_name),
            strategy_version="regime-runtime-live-v1",
            entry_price=Decimal(str(signal.entry_price)),
            stop_loss=Decimal(str(signal.stop_loss)) if signal.stop_loss is not None else None,
            take_profit=Decimal(str(signal.take_profit)) if signal.take_profit is not None else None,
            timestamp=bar.timestamp,
            metadata={
                "regime": regime,
                "regime_probabilities": regime_probs,
                "features": features,
            },
        )
        await self.message_bus.publish("signals", trade_signal.model_dump(mode="json"))
        logger.info(
            "Published regime signal",
            extra={
                "instrument": self.instrument_symbol,
                "side": trade_signal.side.value,
                "quantity": int(trade_signal.quantity),
                "confidence": float(trade_signal.confidence),
                "regime": regime,
            },
        )


def parse_args():
    p = argparse.ArgumentParser(description="Regime runtime strategy agent for live/paper streams")
    p.add_argument("--model-json", required=True)
    p.add_argument("--instrument", default="XAU_USD")
    p.add_argument("--decision-mode", default="ensemble", choices=["ensemble", "router"])
    p.add_argument("--quantity", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.25)
    p.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--warmup", choices=["on", "off"], default="on")
    p.add_argument("--warmup-base-bars", type=int, default=1500)
    p.add_argument("--warmup-m15-bars", type=int, default=None)
    p.add_argument("--warmup-h1-bars", type=int, default=None)
    p.add_argument("--warmup-h4-bars", type=int, default=None)
    p.add_argument("--warmup-d1-bars", type=int, default=None)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    config = Config.load()
    bus = MessageBus(config)
    await bus.connect()

    agent = RegimeRuntimeStrategyAgent(
        config=config,
        message_bus=bus,
        model_json=args.model_json,
        instrument=args.instrument,
        decision_mode=args.decision_mode,
        quantity=args.quantity,
        min_confidence=args.min_confidence,
        gpu_mode=args.gpu,
        warmup_enabled=(args.warmup == "on"),
        warmup_base_bars=args.warmup_base_bars,
        warmup_m15_bars=args.warmup_m15_bars,
        warmup_h1_bars=args.warmup_h1_bars,
        warmup_h4_bars=args.warmup_h4_bars,
        warmup_d1_bars=args.warmup_d1_bars,
    )

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()
        await bus.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
