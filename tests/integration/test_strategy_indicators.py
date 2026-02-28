from agents.strategy.indicators import sma, rsi


def test_sma():
    prices = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = sma(prices, 3)
    assert result == 4.0, f"SMA expected 4.0, got {result}"
    print("SMA test passed")


def test_rsi():
    prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
    result = rsi(prices, 14)
    assert result is not None, "RSI returned None"
    # Expected around 70.46; allow tolerance
    assert 65.0 <= result <= 75.0, f"RSI out of expected range: {result}"
    print(f"RSI test passed: {result:.2f}")


if __name__ == "__main__":
    test_sma()
    test_rsi()
