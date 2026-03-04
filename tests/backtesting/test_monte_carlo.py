def test_monte_carlo_import():
    from oanda_bot.backtesting.analysis.monte_carlo import run_monte_carlo

    assert callable(run_monte_carlo)
