# XAUUSD Breakout-Only Strategy (Institutional Spec)

This strategy is implemented in:
`backtesting/strategy/examples/xau_session_breakout.py`

## Objective

1. Trade only volatility expansion.
2. Avoid mean-reversion/ranging conditions.
3. Keep asymmetric R:R (`>= 1:2`).
4. Limit drawdown via strict daily risk rules.

## Timeframes

1. Execution: `M15`
2. Higher-timeframe bias: `H1` and `H4`

## Regime Filters (No-Trade unless all pass)

1. `ADX(14) >= adx_trade_min` (default `25`)
2. `ATR(14)` expansion over recent candles (default lookback `5`)
3. Bollinger width expansion (`BB(20, 2.0)`)
4. Reject ATR-compression state (`atr_compression_quantile`, default `30th percentile`)
5. Hard floor: `ADX >= 20`

## Entry Logic

1. Build Asian range from `00:00–06:00 GMT`
2. Trade only from London open onward (default `06:00–21:00 GMT`)
3. Long if full candle close above Asian high and HTF bias is bullish
4. Short if full candle close below Asian low and HTF bias is bearish
5. Breakout candle quality:
   1. body > average body of prior `N` candles (default `5`)
   2. volume > average volume of prior `N` candles (default `5`, if enabled)

## Stops / Targets

1. Stop uses stricter breakout protection:
   1. Long: `min(candle_low, entry - 1.2*ATR)`
   2. Short: `max(candle_high, entry + 1.2*ATR)`
2. TP is fixed at `min_rr * risk` (default `2R`)
3. Trailing/partial intent is recorded in metadata for extension.

## Risk Management

1. Position sizing by risk:
   1. risk amount = `account_equity * risk_per_trade` (default `1%`)
   2. quantity = `risk amount / stop distance`
2. Max `2` trades/day
3. Stop trading after `2` consecutive losses (internal conservative tracker)
4. Spread guard:
   1. skip trades if estimated spread > threshold
5. News blackout:
   1. skip windows around configured events (`news_events_utc`)

## No-Lookahead Policy

1. Uses only closed bars and prior-window statistics.
2. Breakout compares close against prebuilt session range.
3. Indicator calculations use historical arrays only.
4. Fill behavior configured by engine (`next_open` recommended for realism).

## Run Commands

Single run:
`make xau-breakout-backtest START=2025-01-01 END=2025-12-31`

Optimization:
`make xau-breakout-opt TRAIN_START=2023-01-01 TRAIN_END=2024-12-31 TEST_START=2025-01-01 TEST_END=2025-12-31`

## Optimization Ranges (Recommended)

1. `adx_trade_min`: `22, 25, 28, 32`
2. `stop_atr_mult`: `1.0, 1.2, 1.5, 1.8`
3. `min_rr`: `2.0, 2.5, 3.0`
4. `atr_compression_quantile`: `0.20, 0.30, 0.40`
5. `body_lookback`: `4, 5, 8`
6. `volume_lookback`: `4, 5, 8`
7. `max_spread_pips`: `20, 25, 30, 35`
8. `trade_start_hour`: `6, 7, 8`
9. `trade_end_hour`: `17, 20, 21`

## Multi-Timeframe Regime Training

Train regime classifier with M15/H1/H4 features:
`make train-mtf-regime INSTRUMENTS=XAU_USD,EUR_USD,GBP_USD START=2022-01-01 END=2024-12-31 GPU=auto`

This produces:
1. model JSON with feature normalization + centroids + regime strategy mapping
2. per-row labeled CSV for analysis
