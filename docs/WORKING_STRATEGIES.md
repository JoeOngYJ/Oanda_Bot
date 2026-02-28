# Working Strategies (Regime Runtime Sweep)

This document summarizes the latest cross-instrument, cross-timeframe regime-runtime validation.

## Scope

1. Decision mode: `ensemble`
2. Fill mode: `next_open`
3. Model build window: `2024-01-01` to `2024-12-31`
4. Untouched evaluation window: `2025-01-01` to `2025-12-31`
5. Instruments: `EUR_USD`, `GBP_USD`, `USD_JPY`, `XAU_USD`
6. Timeframes: `M15`, `H1`
7. Runtime param overrides: `data/research/universe_research_20260228_140910_shortlist.csv`
8. Runtime risk controls:
   1. `risk_per_trade_pct=0.01`
   2. `max_notional_exposure_pct=1.0`
   3. `min_quantity=1`, `max_quantity=100000`
9. Runtime guardrails:
   1. `max_drawdown_stop_pct=0.20`
   2. `daily_loss_limit_pct=0.05`

Raw sweep file:
`data/research/regime_runtime_sweep_20260228_143406_tuned_params_risksized_guardrails.csv`

## Promotion Gate

A setup is marked `PASS` only if all conditions hold:

1. `net_pnl > 0`
2. `max_drawdown <= 20%`
3. `trades >= 50`
4. `sharpe > 0`

## Results

| Instrument | TF | Trades | Net PnL | Sharpe | Max DD | Status |
|---|---|---:|---:|---:|---:|---|
| EUR_USD | H1 | 246 | -934.64 | -1.3256 | 12.76% | REJECT |
| EUR_USD | M15 | 624 | -1395.88 | -1.5650 | 20.05% | REJECT |
| GBP_USD | H1 | 250 | 34.18 | 0.0476 | 8.77% | PASS |
| GBP_USD | M15 | 441 | -1467.29 | -1.9180 | 20.43% | REJECT |
| USD_JPY | H1 | 299 | 4372.85 | 2.3212 | 9.89% | PASS |
| USD_JPY | M15 | 120 | -1885.67 | -2.3397 | 21.29% | REJECT |
| XAU_USD | H1 | 336 | 1184.75 | 0.8161 | 8.07% | PASS |
| XAU_USD | M15 | 1037 | 3684.40 | 1.4377 | 18.18% | PASS |

## Interpretation

1. Risk-normalized sizing materially reduced catastrophic drawdowns on `USD_JPY` and `XAU_USD`.
2. Passing setups under unchanged gate are now:
   1. `GBP_USD H1`
   2. `USD_JPY H1`
   3. `XAU_USD M15`
   4. `XAU_USD H1`
3. `M15` on `EUR_USD`, `GBP_USD`, and `USD_JPY` remains unstable under this gate, even after guardrails.
4. Risk sizing is helping survivability; strategy edge quality still varies by instrument/timeframe.

## Current Working Strategy (Strict Gate)

1. Primary candidates:
   1. `USD_JPY H1`
   2. `XAU_USD M15`
   3. `XAU_USD H1`
   4. `GBP_USD H1`
2. Deployment status: `research/paper only`, not production-live yet.
3. Next hardening priorities:
   1. add portfolio-level kill switches (`daily_loss_limit`, rolling DD stop),
   2. replace static reference prices in sizing with dataset/live-derived prices,
   3. evaluate stricter drawdown threshold for `M15` variants (`<= 15%` stress gate).
