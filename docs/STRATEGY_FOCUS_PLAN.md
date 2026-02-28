# Strategy Focus Plan

This defines where to focus strategy research first.

## 1. Initial Universe
Start with:
1. `EUR_USD`
2. `GBP_USD`
3. `USD_JPY`
4. `XAU_USD`

Rationale:
1. High liquidity majors plus one metals instrument with different behavior profile.
2. Good balance of trend, mean-reversion, and volatility regime diversity.

## 2. Timeframe Stack (Top-Down)
1. `D1`: regime/trend context
2. `H4`: intermediate structure confirmation
3. `H1`: setup context
4. `M15`: execution timing

Use top-down alignment for trend-following models and lower-timeframe trigger precision.

## 3. Strategy Families to Prioritize
1. Breakout:
   1. Donchian/range break with volatility filter.
2. Mean Reversion:
   1. Deviation from moving average with strict stop logic.
3. Multi-Timeframe Trend:
   1. Higher-timeframe trend confirmation + lower-timeframe trigger.
4. Trend Pullback:
   1. EMA alignment + pullback continuation entries.
5. Volatility Expansion:
   1. ATR breakout and volatility compression-to-expansion models.
6. Ensemble/Voting:
   1. Combine multiple models and trade only when votes align.
7. Intermarket MTF Confluence:
   1. Primary pair trend + reference-pair alignment + relative-strength filter.

## 4. Correlation Control
1. Evaluate rolling return correlations across selected instruments.
2. Avoid scaling multiple highly-correlated strategies at the same time.
3. Prefer diversification across low-correlation edges, not only across symbols.

## 5. Research Workflow
1. Download/prepare data:
   1. `python scripts/download_data.py --instrument EUR_USD --tf M15 --start 2022-01-01 --end 2025-12-31`
   2. Repeat for `GBP_USD`, `USD_JPY`, `XAU_USD`.
2. Run per-strategy research:
   1. `make strategy-research INSTRUMENT=EUR_USD TF=M15 START=2022-01-01 END=2025-12-31`
3. Run universe-level research with walk-forward and correlation filters:
   1. `make universe-research INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 START=2022-01-01 END=2025-12-31 WF_WINDOWS=8 MIN_STABILITY=0.30 MIN_TRADES=2 MAX_CORR=0.70`
4. Use GPU pre-screening to narrow parameter sets before full walk-forward:
   1. `make gpu-prescreener INSTRUMENT=EUR_USD TF=M15 START=2022-01-01 END=2025-12-31 GPU=auto TOP_N=20`
   2. Run universe walk-forward on shortlisted configs:
   3. `make universe-research INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 START=2022-01-01 END=2025-12-31 WF_WINDOWS=8 MIN_STABILITY=0.30 MIN_TRADES=2 MAX_CORR=0.70 CANDIDATE_SHORTLIST=data/research/<gpu_shortlist.csv>`
5. Review:
   1. Net test PnL and test PnL std
   2. Expectancy and profit factor
   3. Drawdown and fees
   4. Correlation matrix
   5. Walk-forward stability score
   6. Correlation-aware shortlist CSV
6. Run period/regime identification:
   1. `make regime-gpu-research INSTRUMENT=EUR_USD TF=M15 START=2022-01-01 END=2025-12-31 REGIMES=4 GPU=auto`
   2. Use `*_best_by_regime.csv` to map regimes to best strategies.
7. Run intermarket MTF candidate in universe research:
   1. `make universe-research INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD BASE_TF=M15 START=2022-01-01 END=2025-12-31 WF_WINDOWS=8 MIN_STABILITY=0.30 MIN_TRADES=2 MAX_CORR=0.70`
   2. Check rows with `strategy_name=IntermarketMTFConfluence`.

## 6. Promotion Rule
Promote only candidates that pass:
1. Positive out-of-sample expectancy.
2. Acceptable drawdown relative to return.
3. Stable results across multiple windows.
4. Not strongly redundant with already-selected portfolio strategies.
5. Minimum activity (`test_trades_mean >= 2`) to avoid overfitting to sparse trades.
