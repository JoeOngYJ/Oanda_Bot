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
`data/research/regime_runtime_sweep_20260228_144022_tuned_params_risksized_guardrails_dataprice.csv`

## Promotion Gate

A setup is marked `PASS` only if all conditions hold:

1. `net_pnl > 0`
2. `max_drawdown <= 20%`
3. `trades >= 50`
4. `sharpe > 0`

## Results

| Instrument | TF | Trades | Net PnL | Sharpe | Max DD | Status |
|---|---|---:|---:|---:|---:|---|
| EUR_USD | H1 | 246 | -891.35 | -1.3256 | 12.19% | REJECT |
| EUR_USD | M15 | 659 | -1399.42 | -1.6278 | 19.99% | REJECT |
| GBP_USD | H1 | 250 | 32.82 | 0.0476 | 8.43% | PASS |
| GBP_USD | M15 | 574 | -1526.09 | -1.8488 | 20.81% | REJECT |
| USD_JPY | H1 | 299 | 4424.59 | 2.3229 | 9.97% | PASS |
| USD_JPY | M15 | 102 | -2174.95 | -2.6414 | 24.12% | REJECT |
| XAU_USD | H1 | 336 | 473.90 | 0.8161 | 3.40% | PASS |
| XAU_USD | M15 | 1037 | 1473.76 | 1.4377 | 9.56% | PASS |

## Interpretation

1. Data-driven reference pricing kept `PASS` count at `4/8` while tightening risk realism versus static pricing.
2. Passing setups under unchanged gate are now:
   1. `GBP_USD H1`
   2. `USD_JPY H1`
   3. `XAU_USD M15`
   4. `XAU_USD H1`
3. `M15` on `EUR_USD`, `GBP_USD`, and `USD_JPY` remains unstable under this gate, even after guardrails.
4. `XAU_USD` drawdowns improved further under data-priced sizing (`H1: 3.40%`, `M15: 9.56%`).
5. Strategy edge quality still varies by instrument/timeframe.

## Current Working Strategy (Strict Gate)

1. Primary candidates:
   1. `USD_JPY H1`
   2. `XAU_USD M15`
   3. `XAU_USD H1`
   4. `GBP_USD H1`
2. Deployment status: `research/paper only`, not production-live yet.
3. Next hardening priorities:
   1. persist per-bar guardrail rejection diagnostics to artifacts,
   2. align USD conversion for non-USD-quoted crosses in sizing,
   3. evaluate stricter drawdown threshold for `M15` variants (`<= 15%` stress gate).
