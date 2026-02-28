# Multi-Timeframe Baseline Framework

This document defines the default strategy architecture going forward.

## Baseline Structure

1. Higher-timeframe regime/trend layer:
   1. `D1` + `H1` identify market regime, trend direction, volatility state.
   2. This layer controls which strategy family is allowed.
2. Lower-timeframe execution layer:
   1. `H1` + `M15` are the default execution stack.
   2. `M1` is optional and used only after a separate pilot pass.
3. Risk/governance layer:
   1. risk-based sizing,
   2. exposure caps,
   3. drawdown and daily-loss guardrails.

## Training Policy

1. Fine-tune window:
   1. Start with recent `2-3` years (`fine_start`..`fine_end`) for fast iteration.
2. Full-history window:
   1. Retrain on `~10` years (`full_start`..`full_end`) after architecture stabilizes.
3. Out-of-sample:
   1. Evaluate on strictly unseen period (`oos_start`..`oos_end`).
   2. Keep promotion gate unchanged unless explicitly revised.

## Large-Data Handling

1. Data patching/chunking:
   1. Cache data in month/quarter chunks (`--chunk-months`) to avoid giant one-shot pulls.
2. Artifact retention:
   1. Keep raw data + final model artifacts + final eval CSVs.
   2. Archive/prune heavy intermediates (`bar_regimes`, label dumps, window-level debug CSVs).

## Operational Commands

1. Prepare and cache chunked data:
```bash
make mtf-prepare-data \
  INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD \
  BASE_TF=M15 HTF1=H1 HTF2=H4 PREP_EXTRA_TFS=D1 \
  FULL_START=2015-01-01 OOS_END=2025-12-31 CHUNK_MONTHS=3
```

2. Train baseline multiframe model (fine-tune + full-history):
```bash
make mtf-train \
  INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD \
  BASE_TF=M15 HTF1=H1 HTF2=H4 \
  FINE_START=2022-01-01 FINE_END=2024-12-31 \
  FULL_START=2015-01-01 FULL_END=2024-12-31 GPU=on
```

3. Run strict out-of-sample evaluation:
```bash
make mtf-eval \
  INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD \
  EVAL_TFS=H1,M15 OOS_START=2025-01-01 OOS_END=2025-12-31 \
  STRATEGY_PARAMS_CSV=data/research/<latest_shortlist.csv>
```

4. End-to-end one command:
```bash
make mtf-pipeline \
  INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,XAU_USD \
  BASE_TF=M15 HTF1=H1 HTF2=H4 EVAL_TFS=H1,M15 \
  FINE_START=2022-01-01 FINE_END=2024-12-31 \
  FULL_START=2015-01-01 FULL_END=2024-12-31 \
  OOS_START=2025-01-01 OOS_END=2025-12-31 GPU=on
```

5. Archive/prune old intermediate research files:
```bash
make mtf-clean OLDER_THAN_DAYS=7 DELETE_AFTER_ARCHIVE=1
```

## Notes on `M1`

1. `M1` is now supported by core timeframe mapping and CLI validators.
2. For baseline regime framework, keep model training at `M15/H1/H4` first.
3. Run `M1` as an execution pilot track, then promote only if it passes the same risk gate.
