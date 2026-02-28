def test_engine_import():
    from backtesting.core.engine import BacktestEngine

    eng = BacktestEngine()
    assert hasattr(eng, "run")
