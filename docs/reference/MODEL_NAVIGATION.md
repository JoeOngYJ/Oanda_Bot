# Model Navigation Guide

## Core Model Workflows

- Multi-timeframe model training:
  - `scripts/train_multiframe_regime_model.py`
- Regime GPU research and model generation:
  - `scripts/run_regime_gpu_research.py`
- Runtime model evaluation/backtest:
  - `scripts/run_regime_runtime_backtest.py`
- Baseline pipeline orchestration:
  - `scripts/run_mtf_baseline_pipeline.py`
- GPU pre-screening for strategy candidates:
  - `scripts/run_gpu_prescreener.py`
- Regime->strategy routing research (D1/H1 regimes, LTF entries):
  - `scripts/regime_strategy_research.py`
  - config snippet: `config/regime_strategy_research.snippet.yaml`

## Runtime Model Consumption

- Live/runtime strategy agent:
  - `agents/strategy/regime_runtime_agent.py`
- Runtime regime core:
  - `backtesting/core/regime_runtime.py`
- Supervisor auto-select latest model:
  - `scripts/trading_supervisor.py`

## Model Artifacts

- Trained model JSONs:
  - `data/research/multiframe_regime_model_*.json`
- Promoted/active model JSONs:
  - `models/active/*.json`
- Optional label dumps:
  - `data/research/multiframe_regime_labels_*.csv`
- Research sweeps and summaries:
  - `data/research/*.csv`

## Model Documentation

- Architecture baseline:
  - `docs/MTF_MODEL_ARCHITECTURE.md`
- Baseline framework and run flow:
  - `docs/MTF_BASELINE.md`
- Working strategy results:
  - `docs/WORKING_STRATEGIES.md`
- Strategy evidence:
  - `docs/STRATEGY_EVIDENCE.md`

## Quick Commands

- Train model:
  - `python scripts/train_multiframe_regime_model.py --instruments XAU_USD,EUR_USD,GBP_USD --start 2022-01-01 --end 2024-12-31 --gpu auto`
- Run runtime backtest:
  - `python scripts/run_regime_runtime_backtest.py --model-json data/research/<model>.json --instrument XAU_USD --tf M15 --start 2025-01-01 --end 2025-12-31 --fill-mode next_open --decision-mode ensemble`
- Run GPU research:
  - `python scripts/run_regime_gpu_research.py --instrument EUR_USD --tf M15 --regimes 4 --gpu auto --demo-bars 3000`

## Conventions

- Store model outputs under `data/research/`.
- Keep training/runtime feature names aligned before promoting a model.
- Prefer timestamped artifact names and avoid overwriting prior model JSONs.
