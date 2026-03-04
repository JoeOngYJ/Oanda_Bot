# backtesting/utils/parallel.py

from multiprocessing import Pool, cpu_count
from typing import List, Callable, Any
import logging

logger = logging.getLogger(__name__)

class ParallelBacktestRunner:
    """
    Run multiple backtests in parallel.
    
    Use Cases:
    - Parameter optimization (grid search)
    - Multi-strategy testing
    - Monte Carlo simulations
    """
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
    
    def run_parallel(
        self,
        backtest_func: Callable,
        param_sets: List[Dict]
    ) -> List[Any]:
        """
        Run backtest function with different parameters in parallel.
        
        Args:
            backtest_func: Function that takes config dict and returns result
            param_sets: List of parameter dictionaries
        
        Returns:
            List of results (same order as param_sets)
        """
        logger.info(f"Running {len(param_sets)} backtests on {self.n_workers} workers")
        
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(backtest_func, param_sets)
        
        return results