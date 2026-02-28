# backtesting/core/timeframe.py

from enum import Enum
from typing import List
import re


class Timeframe(Enum):
    """
    Timeframe enumeration with pandas resample string, seconds, and OANDA granularity.
    Each member stores: (pandas_freq, seconds, oanda_granularity)
    """
    M15 = ("15T", 900, "M15")
    M30 = ("30T", 1800, "M30")
    H1 = ("1H", 3600, "H1")
    H4 = ("4H", 14400, "H4")
    D1 = ("1D", 86400, "D1")

    def __init__(self, pandas_freq: str, seconds: int, oanda_gran: str):
        self._pandas_freq = pandas_freq
        self._seconds = seconds
        self._oanda_gran = oanda_gran

    @property
    def seconds(self) -> int:
        """Return timeframe duration in seconds."""
        return self._seconds

    def to_pandas_freq(self) -> str:
        """Return pandas resample frequency string."""
        return self._pandas_freq

    def to_oanda_granularity(self) -> str:
        """Return OANDA granularity string."""
        return self._oanda_gran

    @classmethod
    def _normalize_pandas_freq(cls, freq: str) -> str:
        """
        Normalize pandas frequency string to canonical form.
        Examples: "60T" -> "1H", "60min" -> "1H", "1440T" -> "1D"
        """
        freq = freq.strip().upper()

        # Handle minute variants
        if freq.endswith("MIN"):
            freq = freq[:-3] + "T"

        # Extract number and unit
        match = re.match(r'^(\d+)([A-Z]+)$', freq)
        if not match:
            return freq

        num, unit = int(match.group(1)), match.group(2)

        # Convert minutes to hours if divisible by 60
        if unit == "T" and num >= 60 and num % 60 == 0:
            num = num // 60
            unit = "H"

        # Convert hours to days if divisible by 24
        if unit == "H" and num >= 24 and num % 24 == 0:
            num = num // 24
            unit = "D"

        return f"{num}{unit}"

    @classmethod
    def from_pandas_freq(cls, freq: str) -> "Timeframe":
        """
        Create Timeframe from pandas frequency string.
        Tolerant of variants like "60T" -> "1H".

        Args:
            freq: Pandas frequency string (e.g., "15T", "1H", "60min")

        Returns:
            Timeframe enum member

        Raises:
            ValueError: If frequency string doesn't match any timeframe
        """
        normalized = cls._normalize_pandas_freq(freq)

        for tf in cls:
            if tf.to_pandas_freq() == normalized:
                return tf

        raise ValueError(f"Unknown pandas frequency: {freq} (normalized: {normalized})")

    @classmethod
    def from_oanda_granularity(cls, gran: str) -> "Timeframe":
        """
        Create Timeframe from OANDA granularity string.
        Case-insensitive (e.g., "H1" or "h1").

        Args:
            gran: OANDA granularity string (e.g., "M15", "H1")

        Returns:
            Timeframe enum member

        Raises:
            ValueError: If granularity doesn't match any timeframe
        """
        gran_upper = gran.strip().upper()

        for tf in cls:
            if tf.to_oanda_granularity().upper() == gran_upper:
                return tf

        raise ValueError(f"Unknown OANDA granularity: {gran}")

    @classmethod
    def get_hierarchy(cls) -> List["Timeframe"]:
        """
        Return timeframes ordered by duration (lowest first).

        Returns:
            List of Timeframe members sorted by seconds
        """
        return sorted(cls, key=lambda tf: tf.seconds)

    @classmethod
    def validate_bidirectional(cls) -> None:
        """
        Validate that pandas<->oanda mappings are bijective and round-trip correctly.
        Raises AssertionError if validation fails.
        Should be called at module import time.
        """
        # Check uniqueness of pandas frequencies
        pandas_freqs = [tf.to_pandas_freq() for tf in cls]
        assert len(pandas_freqs) == len(set(pandas_freqs)), \
            "Duplicate pandas frequencies detected"

        # Check uniqueness of OANDA granularities
        oanda_grans = [tf.to_oanda_granularity() for tf in cls]
        assert len(oanda_grans) == len(set(oanda_grans)), \
            "Duplicate OANDA granularities detected"

        # Check uniqueness of seconds
        seconds_list = [tf.seconds for tf in cls]
        assert len(seconds_list) == len(set(seconds_list)), \
            "Duplicate seconds values detected"

        # Test round-trip: pandas freq -> Timeframe -> pandas freq
        for tf in cls:
            pandas_freq = tf.to_pandas_freq()
            reconstructed = cls.from_pandas_freq(pandas_freq)
            assert reconstructed == tf, \
                f"Pandas round-trip failed for {tf.name}: {pandas_freq} -> {reconstructed.name}"

        # Test round-trip: OANDA gran -> Timeframe -> OANDA gran
        for tf in cls:
            oanda_gran = tf.to_oanda_granularity()
            reconstructed = cls.from_oanda_granularity(oanda_gran)
            assert reconstructed == tf, \
                f"OANDA round-trip failed for {tf.name}: {oanda_gran} -> {reconstructed.name}"

        # Test hierarchy ordering
        hierarchy = cls.get_hierarchy()
        for i in range(len(hierarchy) - 1):
            assert hierarchy[i].seconds < hierarchy[i + 1].seconds, \
                f"Hierarchy ordering failed: {hierarchy[i].name} >= {hierarchy[i + 1].name}"


# Validate at module import time
Timeframe.validate_bidirectional()
