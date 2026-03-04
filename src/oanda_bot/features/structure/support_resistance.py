# backtesting/features/structure/support_resistance.py

import pandas as pd
from typing import List, Tuple
from scipy.signal import argrelextrema

class SupportResistanceDetector:
    """
    Detect support and resistance levels using swing points.
    
    Algorithm:
    1. Find local minima/maxima (swing points)
    2. Cluster nearby levels
    3. Weight by number of touches
    """
    
    @staticmethod
    def detect_levels(
        df: pd.DataFrame,
        lookback: int = 100,
        proximity_threshold: float = 0.001  # 0.1%
    ) -> Tuple[List[float], List[float]]:
        """
        Returns:
            (support_levels, resistance_levels)
        """
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        
        # Find local extrema
        resistance_idx = argrelextrema(highs, np.greater, order=5)[0]
        support_idx = argrelextrema(lows, np.less, order=5)[0]
        
        resistance_prices = highs[resistance_idx]
        support_prices = lows[support_idx]
        
        # Cluster levels within proximity threshold
        resistance_levels = SupportResistanceDetector._cluster_levels(
            resistance_prices, proximity_threshold
        )
        support_levels = SupportResistanceDetector._cluster_levels(
            support_prices, proximity_threshold
        )
        
        return support_levels, resistance_levels
    
    @staticmethod
    def _cluster_levels(prices: np.ndarray, threshold: float) -> List[float]:
        """Cluster nearby price levels"""
        if len(prices) == 0:
            return []
        
        clusters = []
        sorted_prices = np.sort(prices)
        
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            if (price - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(price)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [price]
        
        clusters.append(np.mean(current_cluster))
        return clusters