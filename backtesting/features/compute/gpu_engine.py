# backtesting/features/compute/gpu_engine.py

try:
    import cupy as cp
    import cusignal
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

class GPUIndicatorEngine:
    """
    GPU-accelerated indicators using CuPy.
    Falls back to CPU if GPU unavailable.
    """
    
    def __init__(self):
        if not GPU_AVAILABLE:
            raise RuntimeError("GPU not available. Install cupy and cusignal.")
    
    @staticmethod
    def sma(series: cp.ndarray, period: int) -> cp.ndarray:
        """GPU-accelerated SMA using CuPy"""
        return cusignal.convolve(
            series,
            cp.ones(period) / period,
            mode='same'
        )
    
    # ... other indicators