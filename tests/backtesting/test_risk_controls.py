from datetime import datetime, timedelta, timezone
from decimal import Decimal

from backtesting.core.timeframe import Timeframe
from backtesting.core.types import InstrumentSymbol
from backtesting.data.models import OHLCVBar
from backtesting.execution.commission import OandaSpreadOnlyCommissionModel
from backtesting.execution.simulator import ExecutionSimulator
from backtesting.execution.slippage import SlippageModel
from backtesting.strategy.signal import Signal, SignalDirection


def _bar(ts: datetime, close: Decimal) -> OHLCVBar:
    return OHLCVBar(
        timestamp=ts,
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=close,
        high=close + Decimal("0.0020"),
        low=close - Decimal("0.0020"),
        close=close,
        volume=100,
    )


def test_volatility_targeting_reduces_quantity_in_high_vol_regime():
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=SlippageModel(
            spread_pips_by_instrument={"EUR_USD": Decimal("0")},
            default_spread_pips=Decimal("0"),
            slippage_pips=Decimal("0"),
        ),
        commission_model=OandaSpreadOnlyCommissionModel(),
        volatility_targeting_enabled=True,
        target_annual_volatility=Decimal("0.05"),
        volatility_lookback_bars=10,
        base_timeframe_seconds=3600,
    )

    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = Decimal("1.1000")
    # Alternate large swings to create elevated realized volatility.
    for i in range(12):
        price = price * (Decimal("1.02") if i % 2 == 0 else Decimal("0.98"))
        sim.process_bar(_bar(ts + timedelta(hours=i), price))

    signal = Signal(
        timestamp=ts + timedelta(hours=12),
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.LONG,
        strategy_name="risk_control_test",
        entry_price=price,
        stop_loss=None,
        take_profit=None,
        timeframe=Timeframe.H1,
        confidence=1.0,
        quantity=10000,
    )
    sim.process_signal(signal)
    sim.process_bar(_bar(ts + timedelta(hours=12), price))

    assert len(sim.filled_orders) == 1
    filled_qty = sim.filled_orders[0]["quantity"]
    assert 0 < filled_qty < 10000


def test_max_concurrent_exposure_caps_entry_quantity():
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=SlippageModel(
            spread_pips_by_instrument={"EUR_USD": Decimal("0")},
            default_spread_pips=Decimal("0"),
            slippage_pips=Decimal("0"),
        ),
        commission_model=OandaSpreadOnlyCommissionModel(),
        max_concurrent_exposure_pct=Decimal("0.20"),
    )

    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = Decimal("1.0")
    signal = Signal(
        timestamp=ts,
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.LONG,
        strategy_name="risk_control_test",
        entry_price=price,
        stop_loss=None,
        take_profit=None,
        timeframe=Timeframe.H1,
        confidence=1.0,
        quantity=5000,
    )
    sim.process_signal(signal)
    sim.process_bar(_bar(ts, price))

    assert len(sim.filled_orders) == 1
    # Cap notional = 20% * 10,000 = 2,000 -> at price 1.0 max qty is 2,000.
    assert sim.filled_orders[0]["quantity"] == 2000
