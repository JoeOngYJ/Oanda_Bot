# Deployment Readiness TODO (Prioritized)

## P0: Wire Optimized Params into Runtime (Done)

1. Runtime backtest now supports `--strategy-params-csv` and applies per-instrument strategy overrides from shortlist CSV.
2. Make target `regime-runtime-backtest` now accepts `STRATEGY_PARAMS_CSV=<path>`.

## P1: Re-run Strict Holdout with Tuned Runtime Params (Done)

1. Re-ran `regime-runtime-backtest` sweep for `EUR_USD,GBP_USD,USD_JPY,XAU_USD` x `M15,H1` using latest per-instrument runtime models and:
   `STRATEGY_PARAMS_CSV=data/research/universe_research_20260228_140910_shortlist.csv`.
2. Sweep artifact: `data/research/regime_runtime_sweep_20260228_142339_tuned_params.csv`.
3. Kept promotion gate unchanged:
   1. `net_pnl > 0`
   2. `max_drawdown <= 20%`
   3. `trades >= 50`
   4. `sharpe > 0`
4. Updated `docs/WORKING_STRATEGIES.md` from that sweep.

## P2: Instrument-Normalized Risk Sizing (Done - First Version)

1. Implemented in runtime backtest:
   1. `risk_per_trade_pct`,
   2. `max_notional_exposure_pct`,
   3. `min_quantity` / `max_quantity`,
   4. instrument-normalized USD notional-per-unit approximation.
2. Remaining refinement:
   1. align USD conversion for non-USD-quoted crosses.
3. Completed effect check:
   1. strict holdout with tuned params + risk sizing now yields 4/8 PASS (`data/research/regime_runtime_sweep_20260228_142934_tuned_params_risksized.csv`).
   2. data-driven reference pricing is now enabled in runtime sizing (`data_median_close`), validated via
      `data/research/regime_runtime_sweep_20260228_144022_tuned_params_risksized_guardrails_dataprice.csv` (4/8 PASS).
4. Replace fixed `quantity=10000` with risk-based sizing in runtime:
   1. per-trade risk pct,
   2. ATR or stop-distance sizing,
   3. instrument-specific contract/pip normalization.
5. Add hard caps:
   1. max quantity,
   2. max notional exposure,
   3. max concurrent positions per instrument.

## P3: Portfolio Risk Guardrails in Runtime (Done - First Version)

1. Implemented runtime guardrails before order approval:
   1. daily max loss,
   2. rolling drawdown kill switch,
   3. explicit rejection counters in runtime output.
2. Effect check:
   1. strict holdout with tuned params + risk sizing + guardrails: `data/research/regime_runtime_sweep_20260228_143406_tuned_params_risksized_guardrails.csv`.
   2. pass count remained `4/8`, with weaker M15 variants seeing lower trade counts (guardrail clipping).
3. Remaining:
   1. add volatility-targeting adaptation to risk manager decisions,
   2. persist per-bar reject reasons to artifact CSV for diagnostics.

## P4: Regime-to-Parameter Mapping

1. Extend regime model payload to include strategy params per regime (not only strategy name).
2. Use regime-specific parameter sets in router/ensemble modules.

## P5: Deployment Qualification Workflow

1. Require sequence before paper/live promotion:
   1. purged walk-forward pass,
   2. strict untouched holdout pass,
   3. shadow-mode consistency pass.
2. Enforce scorecard update for each candidate strategy.
