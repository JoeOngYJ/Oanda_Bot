from backtesting.analysis.metrics import (
    expectancy_per_trade,
    max_drawdown_from_equity,
    profit_factor,
    sharpe,
    win_rate,
)
from backtesting.analysis.reports import build_report


def test_metrics_basic_values():
    trades = [{"pnl": 10}, {"pnl": -5}, {"pnl": 15}]
    assert round(win_rate(trades), 6) == round(2 / 3, 6)
    assert expectancy_per_trade(trades) == 20 / 3
    assert profit_factor(trades) == 5.0
    assert sharpe([0.01, 0.02, -0.01]) != 0.0
    assert round(max_drawdown_from_equity([100, 110, 105, 95]), 6) == round((110 - 95) / 110, 6)


def test_build_report_outputs_markdown_table():
    rows = [
        {
            "strategy_name": "Breakout",
            "params": {"lookback": 20},
            "total_trades": 10,
            "net_pnl": 42.0,
            "expectancy": 4.2,
            "win_rate": 0.6,
            "profit_factor": 1.4,
            "max_drawdown": 0.08,
        }
    ]
    report = build_report(rows)
    assert "# Strategy Research Report" in report
    assert "| Rank | Strategy |" in report
    assert "Breakout" in report
