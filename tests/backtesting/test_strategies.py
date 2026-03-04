def test_strategy_base_import():
    from oanda_bot.backtesting.strategy.base import StrategyBase

    assert hasattr(StrategyBase, "on_bar")
