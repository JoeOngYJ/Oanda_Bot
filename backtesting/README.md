# Backtesting Module

Core backtesting components:
1. `core/`: orchestration engine and base types.
2. `data/`: downloader, warehouse, and data manager.
3. `strategy/`: base strategy interfaces and strategy examples.
4. `execution/`: fill simulation, spread/slippage, and commission models.
5. `analysis/`: post-run metrics and reports (partially stubbed).
6. `visualization/`: chart/export helpers.

Execution modeling includes:
1. Spread-aware fills.
2. Slippage.
3. Commission models.
4. Stop-loss/take-profit lifecycle exits.

Research workflow:
1. Run `python scripts/run_strategy_research.py` to evaluate breakout and mean-reversion parameter grids.
2. Review leaderboard outputs in `data/research/` (CSV + Markdown).
