"""Basic tests for data manager (placeholder)."""
def test_data_manager_import():
    from oanda_bot.backtesting.data.manager import DataManager

    dm = DataManager()
    assert dm is not None
