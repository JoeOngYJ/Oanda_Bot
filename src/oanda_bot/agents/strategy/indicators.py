"""
Technical indicators for trading strategies.
All functions are pure, deterministic, and handle edge cases.
"""

import numpy as np
from typing import List, Optional


def sma(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average.

    Args:
        prices: List of prices
        period: Number of periods for average

    Returns:
        SMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    return float(np.mean(prices[-period:]))


def ema(prices: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: List of prices
        period: Number of periods for average

    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None

    # Calculate smoothing factor
    multiplier = 2.0 / (period + 1)

    # Start with SMA for first value
    ema_value = float(np.mean(prices[:period]))

    # Calculate EMA for remaining values
    for price in prices[period:]:
        ema_value = (price * multiplier) + (ema_value * (1 - multiplier))

    return ema_value


def rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (0-100).

    Args:
        prices: List of prices
        period: RSI period (default 14)

    Returns:
        RSI value or None if insufficient data
    """
    if len(prices) < period + 1:
        return None

    # Calculate price changes
    deltas = np.diff(prices[-period-1:])

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gain and loss
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    # Handle edge case: no losses
    if avg_loss == 0:
        return 100.0

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_value = 100.0 - (100.0 / (1.0 + rs))

    return float(rsi_value)


def bollinger_bands(
    prices: List[float],
    period: int = 20,
    std_dev: float = 2.0
) -> Optional[tuple[float, float, float]]:
    """
    Calculate Bollinger Bands.

    Args:
        prices: List of prices
        period: Period for moving average
        std_dev: Number of standard deviations

    Returns:
        Tuple of (upper_band, middle_band, lower_band) or None
    """
    if len(prices) < period:
        return None

    recent_prices = prices[-period:]
    middle_band = float(np.mean(recent_prices))
    std = float(np.std(recent_prices))

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    return (upper_band, middle_band, lower_band)


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period

    Returns:
        ATR value or None if insufficient data
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])

        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)

    if len(true_ranges) < period:
        return None

    return float(np.mean(true_ranges[-period:]))
