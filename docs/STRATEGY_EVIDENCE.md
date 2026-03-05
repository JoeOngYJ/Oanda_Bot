# Strategy Evidence Map

This maps implemented strategy families to empirical research and practical usage.

## 1. Time-Series Trend Following
Implemented families:
1. `EMATrendPullback`
2. `Breakout`
3. `ATRBreakout`
4. `MultiTimeframeTrendStrategy`

Evidence:
1. Moskowitz, Ooi, Pedersen (2012), "Time series momentum" (empirical evidence across asset classes):
   1. https://doi.org/10.1016/j.jfineco.2011.11.003
2. Hurst, Ooi, Pedersen (AQR), "A Century of Evidence on Trend-Following Investing":
   1. https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing

Use case:
1. Better in directional/trending regimes.
2. Avoid oversized deployment during noisy low-vol sideways periods.

## 2. Mean Reversion
Implemented families:
1. `MeanReversion`
2. `RSIBollingerReversion`

Evidence:
1. Poterba & Summers (1988), long-horizon mean reversion in stock returns:
   1. https://doi.org/10.1016/0304-405X(88)90021-9
2. Lo, Mamaysky, Wang (2000), foundations for technical pattern/indicator statistical treatment:
   1. https://doi.org/10.1111/0022-1082.00265

Use case:
1. Best when market is range-bound and volatility shocks are limited.
2. Require strict stop-loss and spread-aware execution because mean-reversion edges are often small.

## 3. Volatility Regime / Compression-Expansion
Implemented families:
1. `VolatilityCompressionBreakout`
2. `ATRBreakout`

Evidence:
1. Bollerslev (1986), volatility clustering basis:
   1. https://doi.org/10.1016/0304-4076(86)90063-1
2. Engle (1982), ARCH dynamics and volatility persistence:
   1. https://doi.org/10.2307/1912773

Use case:
1. Compression-to-expansion setups target regime transitions.
2. Position sizing should scale with ATR/realized volatility.

## 4. Ensemble / Model Crossing
Implemented family:
1. `EnsembleVoteStrategy`

Evidence:
1. Ensemble methods can improve robustness when base models capture different structure:
   1. https://link.springer.com/article/10.1023/A:1018054314350

Use case:
1. Use voting to reduce single-model noise.
2. Still enforce turnover and cost constraints.

## 5. Validation Standard (Required)
1. Walk-forward evaluation, not single in-sample fit.
2. Cost-aware simulation (`spread + slippage + broker commission`).
3. Per-regime model selection using the current XAUUSD two-stage training pipeline.
4. Correlation-aware shortlist via `scripts/run_universe_research.py`.

No strategy is "always safe" or guaranteed profitable. Promotion must be based on out-of-sample stability and risk limits.
