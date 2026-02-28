# XAUUSD Multi-Timeframe Session-Aware Model Architecture

This is the implementation baseline for developing the next-generation multi-timeframe model.

## 1) Data Preprocessing

1. Instruments (phase-1): `XAU_USD` only.
2. Timeframes:
   1. Higher-TF: `D1`, `H1`
   2. Lower-TF: `M15` (baseline), optional pilots `M5`, `M1`
3. Input schema:
   1. Full OHLCV candles (`open`, `high`, `low`, `close`, `volume`) for each timeframe.
4. Rules:
   1. UTC timestamps only.
   2. Strictly closed-bar usage; no lookahead.
   3. Gaps/irregularities tracked and handled before training.

## 2) Multi-Timeframe Feature Alignment

1. Anchor rows on execution timeframe (`M15` baseline).
2. Join higher-timeframe bars by latest closed bar at-or-before anchor timestamp (`asof` alignment):
   1. `H1` -> anchor `M15`
   2. `D1` -> anchor `M15`
3. Runtime and training use the same feature names so models are executable in backtests.

## 3) Session-Aware Features

Session logic uses UTC:

1. Flags:
   1. `sess_asia` (`00:00-06:59`)
   2. `sess_europe` (`07:00-12:59`)
   3. `sess_us` (`13:00-21:59`)
   4. `sess_eu_us_overlap` (`13:00-16:59`)
2. Time embeddings:
   1. `hour_sin`, `hour_cos`
   2. `wday_sin`, `wday_cos`
3. OHLCV-derived features by timeframe:
   1. returns (`ret1`, `ret4`)
   2. volatility/structure (`atr_pct`, `bbw`, `range_pct`)
   3. candle body (`body_pct`)
   4. volume context (`vol_z`)

## 4) Model Structure

Current implementation follows a hierarchical regime design:

1. Regime/trend encoder (higher+execution TF features):
   1. KMeans regime states from multiframe OHLCV + session features.
2. Runtime predictor:
   1. Nearest-centroid regime classification online.
3. Regime-conditioned execution:
   1. Ensemble/router strategy modules use detected regime.
   2. Risk sizing and guardrails constrain lower-TF decisions.

## 5) Higher-TF Conditioning of Lower-TF Decisions

Conditioning path:

1. `D1/H1/M15` features -> regime predictor
2. regime -> strategy-style weighting / routing
3. execution signal on lower TF (`M15` baseline; `M1` pilot path)
4. risk manager applies:
   1. per-trade risk sizing
   2. exposure caps
   3. drawdown/daily loss guardrails

## 6) Training & Validation Workflow

1. Fine-tune stage:
   1. `2-3` recent years for fast iteration.
2. Full-history stage:
   1. expand to `~10` years.
3. Final evaluation:
   1. strict unseen out-of-sample window.
4. Promotion gate unchanged:
   1. `net_pnl > 0`
   2. `max_drawdown <= 20%`
   3. `trades >= 50`
   4. `sharpe > 0`

## 7) XAUUSD-First Implementation Plan

1. Train multiframe regime model on `XAU_USD` with `M15/H1/H4/D1`.
2. Validate runtime on `XAU_USD` OOS with risk controls.
3. Run optional `M1` pilot as execution-only extension.
4. Expand to other instruments only after XAUUSD OOS pass is stable.
