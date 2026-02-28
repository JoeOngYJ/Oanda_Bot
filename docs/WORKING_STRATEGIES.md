# Working Strategies (Regime Runtime Sweep)

This document summarizes the latest cross-instrument, cross-timeframe regime-runtime validation.

## Scope

1. Decision mode: `ensemble`
2. Fill mode: `next_open`
3. Model build window: `2024-01-01` to `2024-12-31`
4. Untouched evaluation window: `2025-01-01` to `2025-12-31`
5. Instruments: `EUR_USD`, `GBP_USD`, `USD_JPY`, `XAU_USD`
6. Timeframes: `M15`, `H1`

Raw sweep file:
`data/research/regime_runtime_sweep_20260228_133046.csv`

## Promotion Gate

A setup is marked `PASS` only if all conditions hold:

1. `net_pnl > 0`
2. `max_drawdown <= 20%`
3. `trades >= 50`
4. `sharpe > 0`

## Results

| Instrument | TF | Trades | Net PnL | Sharpe | Max DD | Status |
|---|---|---:|---:|---:|---:|---|
| EUR_USD | H1 | 251 | -725.59 | -0.9338 | 13.22% | REJECT |
| EUR_USD | M15 | 789 | -1663.66 | -1.5764 | 25.34% | REJECT |
| GBP_USD | H1 | 250 | 43.76 | 0.0476 | 11.17% | PASS |
| GBP_USD | M15 | 796 | -1294.16 | -1.0261 | 26.18% | REJECT |
| USD_JPY | H1 | 290 | 264739.28 | 2.2806 | 71.95% | REJECT |
| USD_JPY | M15 | 889 | 183375.24 | 1.0938 | 585.31% | REJECT |
| XAU_USD | H1 | 360 | 4118479.92 | 1.3779 | 624.09% | REJECT |
| XAU_USD | M15 | 1085 | 2946550.07 | 0.5667 | 558.66% | REJECT |

## Interpretation

1. High absolute PnL on `USD_JPY` and `XAU_USD` is not deployable because drawdown is catastrophic.
2. The only setup passing risk-adjusted gate in this sweep is:
   1. `GBP_USD H1` (small but controlled edge).
3. Current ensemble is over-allocating risk in some instrument/timeframe combinations.

## Current Working Strategy (Strict Gate)

1. Primary: `GBP_USD H1` with regime-runtime ensemble.
2. Deployment status: `research/paper only`, not production-live yet.
3. Next hardening priorities:
   1. instrument-specific max exposure and max position limits,
   2. tighter per-trade risk budget for `XAU_USD` and `USD_JPY`,
   3. re-run this exact sweep with unchanged gate to confirm improved survivability.
