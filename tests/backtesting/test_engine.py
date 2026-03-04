def test_engine_import():
    from oanda_bot.backtesting.core.engine import BacktestEngine

    eng = BacktestEngine()
    assert hasattr(eng, "run")
