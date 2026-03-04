from decimal import Decimal
from datetime import datetime, timezone

from oanda_bot.backtesting.data.models import OHLCVBar
from oanda_bot.backtesting.execution.simulator import ExecutionSimulator
from oanda_bot.backtesting.execution.slippage import SlippageModel
from oanda_bot.backtesting.execution.commission import (
    OandaCoreCommissionModel,
    OandaSpreadOnlyCommissionModel,
)
from oanda_bot.backtesting.core.timeframe import Timeframe
from oanda_bot.backtesting.core.types import InstrumentSymbol
from oanda_bot.backtesting.strategy.signal import Signal
from oanda_bot.backtesting.strategy.signal import SignalDirection


def test_spread_applied_to_long_price():
    model = SlippageModel(
        spread_pips_by_instrument={"EUR_USD": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    filled = model.apply(Decimal("1.10000"), SignalDirection.LONG, "EUR_USD")
    # Half spread = 0.7 pips = 0.00007
    assert filled == Decimal("1.10007")


def test_spread_applied_to_short_price():
    model = SlippageModel(
        spread_pips_by_instrument={"EUR_USD": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    filled = model.apply(Decimal("1.10000"), SignalDirection.SHORT, "EUR_USD")
    # Half spread = 0.7 pips = 0.00007
    assert filled == Decimal("1.09993")


def test_jpy_pip_size_handling():
    model = SlippageModel(
        spread_pips_by_instrument={"USD_JPY": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    filled = model.apply(Decimal("150.000"), SignalDirection.LONG, "USD_JPY")
    # JPY pip size 0.01 -> half spread = 0.007
    assert filled == Decimal("150.007")


def test_oanda_core_commission_per_10k_units():
    model = OandaCoreCommissionModel(per_10k_units=Decimal("1.00"))
    commission = model.calculate(Decimal("1.1000"), quantity=10000)
    assert commission == Decimal("1.00")


def test_oanda_core_commission_scales_with_size():
    model = OandaCoreCommissionModel(per_10k_units=Decimal("1.00"))
    commission = model.calculate(Decimal("1.1000"), quantity=25000)
    assert commission == Decimal("2.50")


def test_spread_only_commission_is_zero():
    model = OandaSpreadOnlyCommissionModel()
    commission = model.calculate(Decimal("1.1000"), quantity=100000)
    assert commission == Decimal("0")


def test_execution_simulator_applies_spread_and_commission():
    slippage = SlippageModel(
        spread_pips_by_instrument={"EUR_USD": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    commission = OandaCoreCommissionModel(per_10k_units=Decimal("1.00"))
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=slippage,
        commission_model=commission,
    )
    signal = Signal(
        timestamp=datetime.now(timezone.utc),
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.LONG,
        strategy_name="test",
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0900"),
        take_profit=Decimal("1.1200"),
        timeframe=Timeframe.H1,
        confidence=0.9,
        quantity=10000,
    )
    bar = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1000"),
        high=Decimal("1.1010"),
        low=Decimal("1.0990"),
        close=Decimal("1.1005"),
        volume=100,
    )

    sim.process_signal(signal)
    sim.process_bar(bar)

    assert len(sim.filled_orders) == 1
    fill = sim.filled_orders[0]
    assert fill["fill_price"] == Decimal("1.10007")
    assert fill["commission"] == Decimal("1.00")


def test_simulator_closes_long_on_stop_loss():
    slippage = SlippageModel(
        spread_pips_by_instrument={"EUR_USD": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    commission = OandaCoreCommissionModel(per_10k_units=Decimal("1.00"))
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=slippage,
        commission_model=commission,
    )
    signal = Signal(
        timestamp=datetime.now(timezone.utc),
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.LONG,
        strategy_name="test",
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.0990"),
        take_profit=Decimal("1.1200"),
        timeframe=Timeframe.H1,
        confidence=0.9,
        quantity=10000,
    )
    entry_bar = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1000"),
        high=Decimal("1.1010"),
        low=Decimal("1.0995"),
        close=Decimal("1.1005"),
        volume=100,
    )
    stop_bar = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1002"),
        high=Decimal("1.1007"),
        low=Decimal("1.0985"),
        close=Decimal("1.0992"),
        volume=100,
    )

    sim.process_signal(signal)
    sim.process_bar(entry_bar)
    sim.process_bar(stop_bar)

    assert len(sim.filled_orders) == 2
    assert sim.filled_orders[1]["fill_reason"] == "stop_loss"
    # Exit side for long close is SHORT -> bid fill = stop - half spread
    assert sim.filled_orders[1]["fill_price"] == Decimal("1.09893")
    assert sim.filled_orders[1]["commission"] == Decimal("1.00")


def test_simulator_closes_short_on_take_profit():
    slippage = SlippageModel(
        spread_pips_by_instrument={"EUR_USD": Decimal("1.4")},
        slippage_pips=Decimal("0"),
    )
    commission = OandaCoreCommissionModel(per_10k_units=Decimal("1.00"))
    sim = ExecutionSimulator(
        initial_capital=Decimal("10000"),
        slippage_model=slippage,
        commission_model=commission,
    )
    signal = Signal(
        timestamp=datetime.now(timezone.utc),
        instrument=InstrumentSymbol("EUR_USD"),
        direction=SignalDirection.SHORT,
        strategy_name="test",
        entry_price=Decimal("1.1000"),
        stop_loss=Decimal("1.1020"),
        take_profit=Decimal("1.0980"),
        timeframe=Timeframe.H1,
        confidence=0.9,
        quantity=10000,
    )
    entry_bar = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.1000"),
        high=Decimal("1.1005"),
        low=Decimal("1.0990"),
        close=Decimal("1.0997"),
        volume=100,
    )
    tp_bar = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        timeframe=Timeframe.H1,
        instrument=InstrumentSymbol("EUR_USD"),
        open=Decimal("1.0992"),
        high=Decimal("1.0998"),
        low=Decimal("1.0978"),
        close=Decimal("1.0983"),
        volume=100,
    )

    sim.process_signal(signal)
    sim.process_bar(entry_bar)
    sim.process_bar(tp_bar)

    assert len(sim.filled_orders) == 2
    assert sim.filled_orders[1]["fill_reason"] == "take_profit"
    # Exit side for short close is LONG -> ask fill = tp + half spread
    assert sim.filled_orders[1]["fill_price"] == Decimal("1.09807")
    assert sim.filled_orders[1]["commission"] == Decimal("1.00")
