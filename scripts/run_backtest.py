# scripts/run_backtest.py

from datetime import datetime
from backtesting.core.engine import BacktestEngine
from backtesting.core.timeframe import Timeframe
from backtesting.strategy.examples.multi_tf_trend import MultiTimeframeTrendStrategy
from backtesting.analysis.monte_carlo import MonteCarloSimulator

def main():
    # Configuration
    config = {
        'data': {
            'instrument': 'EUR_USD',
            'base_timeframe': Timeframe.M15,
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2024, 1, 1),
        },
        'strategy': {
            'name': 'MultiTF_Trend',
            'class': MultiTimeframeTrendStrategy,
            'timeframes': [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15],
            'ema_fast': 50,
            'ema_slow': 200
        },
        'execution': {
            'initial_capital': 10000,
            'slippage_pips': 0.2,
            'pricing_model': 'oanda_core',  # spread_only | oanda_core
            'spreads_pips': {
                'EUR_USD': 1.4,
                'GBP_USD': 2.0,
                'USD_JPY': 1.4,
            },
            'core_commission_per_10k_units': 1.0,
        }
    }
    
    # Run backtest
    engine = BacktestEngine(config)
    result = engine.run()
    
    # Print summary
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    
    # Run Monte Carlo
    mc = MonteCarloSimulator(result.trades, config['execution']['initial_capital'])
    mc_result = mc.run_permutation_simulation(n_simulations=10000)
    
    print(f"\nMonte Carlo Results:")
    print(f"Risk of Ruin: {mc_result.risk_of_ruin:.2%}")
    print(f"95% CI: {mc_result.confidence_intervals[95]}")
    
    # Export to InfluxDB for Grafana
    from backtesting.visualization.influxdb_exporter import export_to_influxdb
    export_to_influxdb(result, config.get('influxdb'))

if __name__ == '__main__':
    main()
